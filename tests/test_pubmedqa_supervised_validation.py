import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from helpers.tiny_training import environment, make_config, model_options

from pubmedqa.data.supervised import SupervisedDataCollator, SupervisedExample
from pubmedqa.model.loading import load_full_model
from pubmedqa.train.pipeline import run_training


class SupervisedValidationTest(unittest.TestCase):
    def test_optional_limits_and_unsupported_save_config(self):
        from dataclasses import replace

        from pubmedqa.config.train import validate_training_config

        with tempfile.TemporaryDirectory() as directory:
            cfg = make_config(Path(directory))
            cases = [
                (name, 0)
                for name in (
                    "max_train_examples",
                    "max_validation_examples",
                    "max_test_examples",
                    "max_input_tokens",
                )
            ]
            cases += [
                ("num_workers", -1),
                ("learning_rate", float("nan")),
                ("max_grad_norm", float("inf")),
                ("weight_decay", -1),
                ("warmup_ratio", 2),
                ("eval_every_epoch", False),
            ]
            for field, value in cases:
                with self.subTest(field=field), self.assertRaises(ValueError):
                    validate_training_config(replace(cfg, **{field: value}))

    def test_rank_local_bad_batch_prevents_forward_on_every_rank(self):
        from helpers.distributed import FakeRanks
        from test_pubmedqa_training_loop import ScalarLoss

        from pubmedqa.train.distributed import SynchronizedOperationError
        from pubmedqa.train.loop import train_epoch

        for failed_rank in (0, 1):
            ranks = FakeRanks()
            forwards = [Mock(), Mock()]

            def worker(rank):
                model = ScalarLoss()
                model.register_forward_pre_hook(lambda *a: forwards[rank]())
                optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
                labels = torch.tensor([[-100, -100 if rank == failed_rank else 1]])
                batch = dict(
                    input_ids=torch.ones(1, 2),
                    labels=labels,
                    attention_mask=torch.ones(1, 2),
                )
                return list(
                    train_epoch(
                        model=model,
                        loader=[batch],
                        optimizer=optimizer,
                        scheduler=Mock(),
                        epoch=1,
                        global_step=0,
                        accumulation_steps=1,
                        max_grad_norm=1,
                        device=torch.device("cpu"),
                        autocast_context=nullcontext,
                        ddp_enabled=False,
                        control_group=ranks.group,
                    )
                )

            with (
                patch("torch.distributed.all_gather_object", ranks.gather),
                patch("torch.distributed.get_world_size", return_value=2),
            ):
                results = ranks.run(worker)
            self.assertTrue(
                all(
                    isinstance(result, SynchronizedOperationError) for result in results
                )
            )
            self.assertEqual(str(results[0]), str(results[1]))
            for forward in forwards:
                forward.assert_not_called()

    def test_empty_split_and_masked_late_row_fail_before_optimizer(self):
        from pubmedqa.data.supervised import PubMedQASupervisedDataset

        with self.assertRaisesRegex(ValueError, "Empty"):
            PubMedQASupervisedDataset([], None)
        with tempfile.TemporaryDirectory() as directory:
            cfg = make_config(Path(directory), max_input_tokens=1)
            with patch("torch.optim.AdamW") as optimizer:
                with self.assertRaisesRegex(ValueError, "pubid="):
                    run_training(cfg, environment())
            optimizer.assert_not_called()

    def test_masked_row_rejected_with_pubid_and_no_raw_text(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = make_config(Path(directory))
            tokenizer, _ = load_full_model(cfg.model_name, options=model_options(cfg))
            for side in ("left", "right"):
                tokenizer.padding_side = side
                for maximum in (1, 2):
                    collate = SupervisedDataCollator(tokenizer, maximum)
                    with self.assertRaisesRegex(
                        ValueError, "pubid=bad.*length"
                    ) as caught:
                        collate(
                            [
                                SupervisedExample(
                                    "bad", "test test", "test test yes", "yes"
                                )
                            ]
                        )
                    self.assertNotIn("test test", str(caught.exception))

    def test_valid_one_target_and_mixed_invalid_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = make_config(Path(directory))
            tokenizer, _ = load_full_model(cfg.model_name, options=model_options(cfg))
            good = SupervisedExample("good", "test", "test yes", "yes")
            bad = SupervisedExample("bad", "test", "test", "yes")
            for side in ("left", "right"):
                tokenizer.padding_side = side
                collate = SupervisedDataCollator(tokenizer, None)
                self.assertEqual(
                    1, (collate([good])["labels"][:, 1:] != -100).sum().item()
                )
                with self.assertRaisesRegex(ValueError, "pubid=bad"):
                    collate([good, bad])

    def test_bad_configuration_fails_before_model_or_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = make_config(Path(directory))
            from dataclasses import replace

            for field in (
                "num_epochs",
                "train_batch_size",
                "eval_batch_size",
                "gradient_accumulation_steps",
            ):
                for value in (0, -1):
                    candidate = replace(cfg, **{field: value})
                    with (
                        patch("pubmedqa.train.pipeline.load_full_model") as load,
                        patch("torch.optim.AdamW") as optimizer,
                    ):
                        with self.assertRaisesRegex(ValueError, field):
                            run_training(candidate, environment())
                    load.assert_not_called()
                    optimizer.assert_not_called()

    def test_nonfinite_loss_never_backward_or_steps(self):
        from test_pubmedqa_training_loop import ScalarLoss

        from pubmedqa.train.loop import train_epoch

        model = ScalarLoss()
        model.weight.data.fill_(float("nan"))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = Mock()
        batch = dict(
            input_ids=torch.ones(1, 2),
            labels=torch.ones(1, 2),
            attention_mask=torch.ones(1, 2),
        )
        with patch.object(optimizer, "step") as step:
            with self.assertRaisesRegex(ValueError, "nonfinite"):
                list(
                    train_epoch(
                        model=model,
                        loader=[batch],
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=1,
                        global_step=0,
                        accumulation_steps=1,
                        max_grad_norm=1,
                        device=torch.device("cpu"),
                        autocast_context=nullcontext,
                        ddp_enabled=False,
                    )
                )
        self.assertIsNone(model.weight.grad)
        step.assert_not_called()
        scheduler.step.assert_not_called()
