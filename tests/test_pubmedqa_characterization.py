from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from pubmedqa.evaluation import resolve_metric_labels
from pubmedqa.full_finetune import FullFineTuneCliConfig, PubMedQAFullFineTuner


class PubMedQACharacterizationTest(unittest.TestCase):
    def test_validation_metrics_use_only_labels_present_in_gold_data(self) -> None:
        self.assertEqual(("yes", "no"), resolve_metric_labels(["yes", "no", "yes"]))
        self.assertEqual(
            ("yes", "no", "maybe"),
            resolve_metric_labels(["yes", "maybe", "no"]),
        )

    def test_environment_values_override_full_fine_tune_defaults(self) -> None:
        environment = {
            "PUBMEDQA_TRAIN_PATH": "train.jsonl",
            "PUBMEDQA_VALIDATION_PATH": "validation.jsonl",
            "PUBMEDQA_MODEL_NAME": "local/test-model",
            "PUBMEDQA_NUM_EPOCHS": "7",
            "PUBMEDQA_LEARNING_RATE": "3e-5",
            "PUBMEDQA_CHECKPOINT_PERCENTS": "25,100",
        }
        with patch.dict(os.environ, environment, clear=True):
            cli_config = FullFineTuneCliConfig.from_env()

        self.assertEqual("local/test-model", cli_config.config.model_name)
        self.assertEqual(7, cli_config.config.num_epochs)
        self.assertEqual(3e-5, cli_config.config.learning_rate)
        self.assertEqual((25, 100), cli_config.config.checkpoint_percents)

    def test_checkpoint_directory_schema_remains_stable(self) -> None:
        trainer = object.__new__(PubMedQAFullFineTuner)
        trainer.checkpoints_dir = Path("run/checkpoints")

        checkpoint_dir = trainer._checkpoint_directory(
            checkpoint_kind="scheduled",
            checkpoint_percent=25.0,
            epoch=1,
            global_step=313,
        )

        self.assertEqual(
            Path("run/checkpoints/scheduled_pct_025_epoch_001_step_000313"),
            checkpoint_dir,
        )

    def test_importing_package_does_not_eagerly_import_ml_frameworks(self) -> None:
        process = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; import pubmedqa; "
                    "assert 'torch' not in sys.modules; "
                    "assert 'transformers' not in sys.modules; "
                    "assert 'peft' not in sys.modules"
                ),
            ],
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": "src"},
        )

        self.assertEqual(0, process.returncode, process.stderr)

    def test_legacy_top_level_trainer_import_remains_available(self) -> None:
        process = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from pubmedqa import PubMedQAFullFineTuner, PubMedQALoRAFineTuner; "
                    "assert PubMedQAFullFineTuner.__name__ == 'PubMedQAFullFineTuner'; "
                    "assert PubMedQALoRAFineTuner.__name__ == 'PubMedQALoRAFineTuner'"
                ),
            ],
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": "src"},
        )

        self.assertEqual(0, process.returncode, process.stderr)


if __name__ == "__main__":
    unittest.main()
