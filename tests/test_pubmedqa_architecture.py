from __future__ import annotations

import ast
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


class PubMedQAArchitectureTest(unittest.TestCase):
    def test_run_registry_is_data_only(self) -> None:
        process = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; import pubmedqa.experiments.specs; "
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

    def test_full_ft_and_lora_use_common_training_engine(self) -> None:
        from pubmedqa.training.engine import PubMedQATrainingEngine
        from pubmedqa.training.strategies.full import PubMedQAFullFineTuner
        from pubmedqa.training.strategies.lora import PubMedQALoRAFineTuner

        self.assertTrue(issubclass(PubMedQAFullFineTuner, PubMedQATrainingEngine))
        self.assertTrue(issubclass(PubMedQALoRAFineTuner, PubMedQATrainingEngine))
        self.assertFalse(issubclass(PubMedQALoRAFineTuner, PubMedQAFullFineTuner))

    def test_lora_strategy_does_not_import_private_engine_symbols(self) -> None:
        source_path = Path("src/pubmedqa/training/strategies/lora.py")
        module = ast.parse(source_path.read_text(encoding="utf-8"))
        private_imports: list[str] = []
        for node in ast.walk(module):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.module != "pubmedqa.training.engine":
                continue
            private_imports.extend(alias.name for alias in node.names if alias.name.startswith("_"))

        self.assertEqual([], private_imports)

    def test_legacy_training_imports_are_compatibility_aliases(self) -> None:
        from pubmedqa.full_finetune import PubMedQAFullFineTuner as LegacyFull
        from pubmedqa.lora_finetune import PubMedQALoRAFineTuner as LegacyLoRA
        from pubmedqa.training.strategies.full import PubMedQAFullFineTuner
        from pubmedqa.training.strategies.lora import PubMedQALoRAFineTuner

        self.assertIs(PubMedQAFullFineTuner, LegacyFull)
        self.assertIs(PubMedQALoRAFineTuner, LegacyLoRA)

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
        from pubmedqa.training.strategies.lora import LoRAFineTuneCliConfig

        environment = {
            "PUBMEDQA_TRAIN_PATH": "train.jsonl",
            "PUBMEDQA_VALIDATION_PATH": "validation.jsonl",
            "PUBMEDQA_LORA_LEARNING_RATE": "0.0003",
            "PUBMEDQA_LORA_GRADIENT_CHECKPOINTING": "true",
            "PUBMEDQA_DISTRIBUTED_MODE": "ddp",
        }
        with patch.dict(os.environ, environment, clear=True):
            config = LoRAFineTuneCliConfig.from_env().config

        self.assertEqual(0.0003, config.learning_rate)
        self.assertTrue(config.gradient_checkpointing)
        self.assertEqual("ddp", config.distributed_mode)


if __name__ == "__main__":
    unittest.main()
