"""End-to-end contracts for the functional training pipeline."""

import json
import tempfile
import unittest
from dataclasses import asdict, fields
from pathlib import Path
from unittest.mock import patch

from helpers.offline import offline_cpu
from helpers.tiny_training import environment, make_config


class TrainingPipelineTest(unittest.TestCase):
    def test_training_model_is_released_before_best_checkpoint_test_load(self):
        import weakref

        import pubmedqa.train.pipeline as pipeline

        with tempfile.TemporaryDirectory() as directory, offline_cpu():
            cfg = make_config(Path(directory))
            load_model = pipeline.load_full_model
            loaded = []

            def load(source, **options):
                # Reference load, scheduled evaluation, then best-checkpoint test.
                if len(loaded) == 2:
                    self.assertIsNone(loaded[0]())
                    self.assertIsNone(loaded[1]())
                tokenizer, model = load_model(source, **options)
                loaded.append(weakref.ref(model))
                return tokenizer, model

            with patch.object(pipeline, "load_full_model", load):
                pipeline.run_training(cfg, environment())
            self.assertEqual(3, len(loaded))
            self.assertTrue(all(reference() is None for reference in loaded))

    def test_direct_function_runs_full_and_lora(self):
        from pubmedqa.train.pipeline import run_training
        from helpers.tiny_training import with_lora

        for method in ("full", "lora"):
            with (
                self.subTest(method=method),
                tempfile.TemporaryDirectory() as directory,
                offline_cpu(),
            ):
                config = make_config(Path(directory))
                if method == "lora":
                    config = with_lora(config)
                summary = run_training(config, environment())
                self.assertEqual(2, summary.optimizer_steps)
                self.assertEqual(
                    1464 if method == "full" else 128, summary.trainable_params
                )
                self.assertTrue(Path(summary.best_checkpoint_dir).is_dir())
                from pubmedqa.config.train import TrainingConfig

                persisted = json.loads(
                    (
                        Path(summary.best_checkpoint_dir).parents[1] / "config.json"
                    ).read_text()
                )
                common = {item.name for item in fields(type(config))} - {"adapter"}
                adapter_fields = {
                    "target_modules",
                    "target_layers",
                    "layer_scope",
                    "lora_rank",
                    "lora_alpha",
                    "lora_dropout",
                }
                self.assertEqual(
                    common | adapter_fields | {"environment", "distributed"},
                    set(persisted),
                )

    def test_model_load_failure_closes_session_and_does_not_start_updates(self):
        from pubmedqa.train.distributed import TrainingSession
        from pubmedqa.train.pipeline import run_training

        with tempfile.TemporaryDirectory() as directory, offline_cpu():
            config = make_config(Path(directory))
            session = TrainingSession.from_config(config)
            trace = []
            with (
                patch.object(
                    session, "initialize", wraps=session.initialize
                ) as initialize,
                patch.object(session, "close", wraps=session.close) as close,
                patch(
                    "pubmedqa.train.pipeline.load_full_model",
                    side_effect=ValueError("load failed"),
                ),
                patch("pubmedqa.train.pipeline.train_epoch") as train,
            ):
                initialize.side_effect = lambda: trace.append("initialize")
                close.side_effect = lambda: trace.append("close")
                with self.assertRaisesRegex(ValueError, "load failed"):
                    run_training(config, environment(), session=session)
                train.assert_not_called()
                self.assertEqual(["initialize", "close"], trace)
                close.assert_called_once_with()
