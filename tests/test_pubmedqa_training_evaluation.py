import tempfile
import unittest
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from helpers.tiny_training import make_config, model_options

from pubmedqa.data.supervised import PubMedQASupervisedDataset, build_eval_dataloader
from pubmedqa.data.records import load_local_jsonl
from pubmedqa.eval.validation import (
    EvaluationSettings,
    evaluate_checkpoint_on_main,
    evaluate_split,
)
from pubmedqa.model.loading import load_full_model
from pubmedqa.train.artifacts import RunFiles
from pubmedqa.train.checkpoints import create_evaluation_snapshot
from pubmedqa.train.distributed import TrainingSession


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
            session = TrainingSession.from_config(cfg)
            files = RunFiles.from_config(cfg)
            session.device = torch.device("cuda:0")
            with patch("torch.cuda.empty_cache"):
                result = evaluate_checkpoint_on_main(
                    session=session,
                    settings=EvaluationSettings.from_config(cfg, session.device),
                    evaluations_dir=files.evaluations_dir,
                    load_model=partial(
                        load_full_model,
                        options=model_options(cfg, device=session.device),
                    ),
                    checkpoint_dir=Path(cfg.model_name),
                    examples=load_local_jsonl(cfg.validation_path),
                    split_name="checkpoint",
                )
            self.assertEqual(3, result.metrics.num_examples)
            self.assertEqual("cuda", session.device.type)
            self.assertEqual("torch.float32", session.metadata()["evaluation_dtype"])

    def test_success_and_each_failure_restore_training_and_padding(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = make_config(Path(directory))
            session = TrainingSession.from_config(cfg)
            files = RunFiles.from_config(cfg)
            tokenizer, model = load_full_model(
                cfg.model_name, options=model_options(cfg, device=session.device)
            )
            examples = load_local_jsonl(cfg.validation_path)
            loader = build_eval_dataloader(
                PubMedQASupervisedDataset(examples, tokenizer),
                tokenizer,
                batch_size=cfg.eval_batch_size,
                num_workers=cfg.num_workers,
                max_input_tokens=cfg.max_input_tokens,
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
                                    evaluate_split(
                                        settings=EvaluationSettings.from_config(
                                            cfg, session.device
                                        ),
                                        evaluations_dir=files.evaluations_dir,
                                        model=model,
                                        tokenizer=tokenizer,
                                        supervised_loader=loader,
                                        examples=examples,
                                        split_name="probe",
                                    )
                            else:
                                evaluate_split(
                                    settings=EvaluationSettings.from_config(
                                        cfg, session.device
                                    ),
                                    evaluations_dir=files.evaluations_dir,
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
            session = TrainingSession.from_config(cfg)
            files = RunFiles.from_config(cfg)
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
                    settings=EvaluationSettings.from_config(cfg, session.device),
                    evaluations_dir=files.evaluations_dir,
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
            cfg = make_config(Path(directory))
            session = TrainingSession.from_config(cfg)
            files = RunFiles.from_config(cfg)
            tokenizer, model = load_full_model(
                cfg.model_name, options=model_options(cfg, device=session.device)
            )
            with patch.object(
                tokenizer, "save_pretrained", side_effect=OSError("tokenizer failure")
            ):
                with self.assertRaisesRegex(RuntimeError, "tokenizer failure"):
                    create_evaluation_snapshot(
                        session=session,
                        files=files,
                        model=model,
                        tokenizer=tokenizer,
                        split_name="probe",
                    )
            self.assertEqual(
                [], list((files.output_root / ".evaluation_snapshots").glob("*"))
            )

    def test_fsdp_reference_and_checkpoint_evaluate_on_cpu_without_device_mutation(
        self,
    ):
        with tempfile.TemporaryDirectory() as directory:
            cfg = make_config(Path(directory), distributed_mode="fsdp")
            session = TrainingSession.from_config(cfg)
            files = RunFiles.from_config(cfg)
            session.device = torch.device("cuda:0")
            tokenizer, model = load_full_model(
                cfg.model_name, options=model_options(cfg, device=session.device)
            )
            examples = load_local_jsonl(cfg.validation_path)
            loader = build_eval_dataloader(
                PubMedQASupervisedDataset(examples, tokenizer),
                tokenizer,
                batch_size=cfg.eval_batch_size,
                num_workers=cfg.num_workers,
                max_input_tokens=cfg.max_input_tokens,
            )
            result = evaluate_split(
                settings=EvaluationSettings.from_config(cfg, session.device),
                evaluations_dir=files.evaluations_dir,
                model=model,
                tokenizer=tokenizer,
                supervised_loader=loader,
                examples=examples,
                split_name="reference",
            )
            self.assertEqual(3, result.metrics.num_examples)
            self.assertEqual("cuda", session.device.type)
            self.assertEqual("cpu", next(model.parameters()).device.type)
