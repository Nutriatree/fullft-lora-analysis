from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1] / "src" / "pubmedqa"


class PubMedQAArchitectureTest(unittest.TestCase):
    def test_source_layout_has_five_flat_pipeline_packages(self) -> None:
        packages = {
            path.name
            for path in ROOT.iterdir()
            if path.is_dir() and path.name != "__pycache__"
        }
        self.assertEqual({"config", "data", "eval", "model", "train"}, packages)
        self.assertEqual({"__init__.py"}, {path.name for path in ROOT.glob("*.py")})
        for package in packages:
            nested = [
                path
                for path in (ROOT / package).iterdir()
                if path.is_dir() and path.name != "__pycache__"
            ]
            self.assertEqual([], nested, package)

    def test_run_registry_is_data_only(self) -> None:
        process = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; import pubmedqa.config.experiments; "
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

    def test_standalone_eval_cli_builds_current_runtime_contract(self) -> None:
        from scripts.run_pubmedqa_eval import build_runtime, parse_args

        argv = [
            "run_pubmedqa_eval.py",
            "--data-file",
            "test.jsonl",
            "--run-id",
            "smoke",
            "--model-name",
            "Qwen/Qwen3-1.7B",
            "--condition",
            "baseline",
            "--device",
            "cuda:0",
            "--torch-dtype",
            "bfloat16",
        ]
        with patch("sys.argv", argv):
            runtime = build_runtime(parse_args())

        self.assertEqual("torch", runtime.backend)
        self.assertEqual("cuda:0", runtime.device)
        self.assertEqual("bfloat16", runtime.dtype)

    def test_lora_cli_uses_lora_specific_environment(self) -> None:
        from pubmedqa.config.train import TrainingCliConfig

        environment = {
            "PUBMEDQA_TRAIN_PATH": "train.jsonl",
            "PUBMEDQA_VALIDATION_PATH": "validation.jsonl",
            "PUBMEDQA_LORA_LEARNING_RATE": "0.0003",
            "PUBMEDQA_LORA_GRADIENT_CHECKPOINTING": "true",
            "PUBMEDQA_DISTRIBUTED_MODE": "ddp",
        }
        with patch.dict(os.environ, environment, clear=True):
            config = TrainingCliConfig.from_env("lora").config

        self.assertEqual(0.0003, config.learning_rate)
        self.assertTrue(config.gradient_checkpointing)
        self.assertEqual("ddp", config.distributed_mode)


if __name__ == "__main__":
    unittest.main()
