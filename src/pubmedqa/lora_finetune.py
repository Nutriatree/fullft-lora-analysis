"""Public notebook adapter for LoRA; all training uses the shared pipeline."""

from __future__ import annotations

from pubmedqa.config import EnvironmentConfig
from pubmedqa.config.lora import (
    LoRAFineTuneCliConfig,
    LoRAFineTuneConfig,
    load_target_layers,
    normalize_lora_target_modules,
)
from pubmedqa.full_finetune import PubMedQATrainingEngine
from pubmedqa.train.lora import LoRAAdapterMetric
from pubmedqa.train.pipeline import run_training


class PubMedQALoRAFineTuner(PubMedQATrainingEngine):
    """Legacy LoRA entry point; adapter preparation lives beside model loading."""

    def __init__(self, config: LoRAFineTuneConfig, environment: EnvironmentConfig):
        super().__init__(config, environment)
        self.lora_config = config


def main() -> None:
    cli_config = LoRAFineTuneCliConfig.from_env()
    summary = run_training(cli_config.config, cli_config.environment)
    print(summary.title)
    print(f"Best checkpoint: {summary.best_checkpoint_dir}")
    print(f"Best validation ACC: {summary.best_validation_accuracy:.4f}")
    print(f"Best validation Macro F1: {summary.best_validation_macro_f1:.4f}")
    if summary.test_accuracy is not None:
        print(f"Test ACC: {summary.test_accuracy:.4f}")
        print(f"Test Macro F1: {summary.test_macro_f1:.4f}")


# Temporary aliases for notebooks and scripts written before the public strategy API.
_load_target_layers = load_target_layers
_normalize_lora_target_modules = normalize_lora_target_modules


if __name__ == "__main__":
    main()

__all__ = [
    "LoRAAdapterMetric",
    "LoRAFineTuneCliConfig",
    "LoRAFineTuneConfig",
    "PubMedQALoRAFineTuner",
    "load_target_layers",
    "main",
    "normalize_lora_target_modules",
]
