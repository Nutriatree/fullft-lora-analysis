from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch, Mock
import torch

from helpers.tiny_training import make_config, environment
from pubmedqa.full_finetune import PubMedQAFullFineTuner
from pubmedqa.data.supervised import PubMedQASupervisedDataset
from pubmedqa.eval.inference import load_local_jsonl


class TrainingEvaluationTest(unittest.TestCase):
    def test_both_ranks_receive_same_evaluation_result_or_failure(self):
        from helpers.distributed import FakeRanks
        from pubmedqa.train.distributed import TrainingSession

        ranks = FakeRanks()
        with patch("torch.distributed.broadcast_object_list", ranks.broadcast):
            results = ranks.run(
                lambda rank: TrainingSession.run_on_main_process(
                    SimpleNamespace(
                        is_main_process=rank == 0, control_group=ranks.group
                    ),
                    lambda: {"loss": 3.0, "loss_reduction": "token_mean"},
                    operation_name="long evaluation",
                )
            )
        self.assertEqual([{"loss": 3.0, "loss_reduction": "token_mean"}] * 2, results)

    def test_real_checkpoint_cpu_evaluation_restores_owner_device(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = make_config(
                Path(directory), distributed_mode="fsdp", dtype=torch.bfloat16
            )
            owner = PubMedQAFullFineTuner(cfg, environment())
            owner.device = torch.device("cuda:0")
            with patch("torch.cuda.empty_cache"):
                result = owner._evaluate_checkpoint_on_main(
                    checkpoint_dir=Path(cfg.model_name),
                    examples=load_local_jsonl(cfg.validation_path),
                    split_name="checkpoint",
                )
            self.assertEqual(3, result.metrics.num_examples)
            self.assertEqual("cuda", owner.device.type)
            self.assertEqual(
                "torch.float32", owner._distributed_metadata()["evaluation_dtype"]
            )

    def test_success_and_each_failure_restore_training_and_padding(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = make_config(Path(directory))
            owner = PubMedQAFullFineTuner(cfg, environment())
            tokenizer, model = owner.load_model_and_tokenizer(cfg.model_name)
            examples = load_local_jsonl(cfg.validation_path)
            loader = owner.build_eval_dataloader(
                PubMedQASupervisedDataset(examples, tokenizer), tokenizer
            )
            for initial in (False, True):
                for failure in (None, "forward", "generate", "write"):
                    with self.subTest(initial=initial, failure=failure):
                        model.train(initial)
                        tokenizer.padding_side = "right"
                        context = nullcontext()
                        if failure in ("forward", "generate"):
                            context = patch.object(
                                model, failure, side_effect=RuntimeError("injected")
                            )
                        elif failure == "write":
                            context = patch(
                                "pubmedqa.eval.validation.write_eval_result",
                                side_effect=OSError("injected"),
                            )
                        with context:
                            if failure:
                                with self.assertRaisesRegex(Exception, "injected"):
                                    owner.evaluate_split(
                                        model=model,
                                        tokenizer=tokenizer,
                                        supervised_loader=loader,
                                        examples=examples,
                                        split_name="probe",
                                    )
                            else:
                                owner.evaluate_split(
                                    model=model,
                                    tokenizer=tokenizer,
                                    supervised_loader=loader,
                                    examples=examples,
                                    split_name="probe",
                                )
                        self.assertEqual(initial, model.training)
                        self.assertEqual("right", tokenizer.padding_side)

    def test_token_weighted_loss_is_batch_partition_invariant(self):
        from pubmedqa.eval.validation import EvaluationSettings, evaluate_split

        class TokenLoss(torch.nn.Module):
            def forward(self, input_ids, labels, **kw):
                values = input_ids[:, 1:][labels[:, 1:] != -100].float()
                return SimpleNamespace(loss=values.mean())

        with tempfile.TemporaryDirectory() as directory:
            cfg = make_config(Path(directory))
            owner = PubMedQAFullFineTuner(cfg, environment())
            model = TokenLoss()
            tokenizer = SimpleNamespace(padding_side="right")
            # Three tokens of NLL 1 plus one of NLL 9 => token mean 3.
            rows = [torch.tensor([[-100, value]]) for value in (1, 1, 1, 9)]
            values = []
            for partitions in ((rows[:3], rows[3:]), (rows,)):
                batches = [
                    dict(input_ids=torch.cat(part), labels=torch.cat(part))
                    for part in partitions
                ]
                result = evaluate_split(
                    settings=EvaluationSettings.from_config(cfg, owner.device),
                    evaluations_dir=owner.files.evaluations_dir,
                    model=model,
                    tokenizer=tokenizer,
                    supervised_loader=batches,
                    examples=[],
                    split_name="loss",
                )
                values.append(result.metrics.loss)
            self.assertEqual([3.0, 3.0], values)

    def test_snapshot_creation_failure_removes_partial_files(self):
        with tempfile.TemporaryDirectory() as directory:
            owner = PubMedQAFullFineTuner(make_config(Path(directory)), environment())
            tokenizer, model = owner.load_model_and_tokenizer(owner.config.model_name)
            with patch.object(
                tokenizer, "save_pretrained", side_effect=OSError("tokenizer failure")
            ):
                with self.assertRaisesRegex(RuntimeError, "tokenizer failure"):
                    owner._create_evaluation_snapshot(
                        model=model, tokenizer=tokenizer, split_name="probe"
                    )
            self.assertEqual(
                [], list((owner.output_root / ".evaluation_snapshots").glob("*"))
            )

    def test_fsdp_reference_and_checkpoint_evaluate_on_cpu_without_device_mutation(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            cfg = make_config(Path(directory), distributed_mode="fsdp")
            owner = PubMedQAFullFineTuner(cfg, environment())
            owner.device = torch.device("cuda:0")
            tokenizer, model = owner.load_model_and_tokenizer(cfg.model_name)
            examples = load_local_jsonl(cfg.validation_path)
            loader = owner.build_eval_dataloader(
                PubMedQASupervisedDataset(examples, tokenizer), tokenizer
            )
            result = owner.evaluate_split(
                model=model,
                tokenizer=tokenizer,
                supervised_loader=loader,
                examples=examples,
                split_name="reference",
            )
            self.assertEqual(3, result.metrics.num_examples)
            self.assertEqual("cuda", owner.device.type)
            self.assertEqual("cpu", next(model.parameters()).device.type)
