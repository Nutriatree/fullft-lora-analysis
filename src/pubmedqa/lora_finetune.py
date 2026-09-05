"""Compatibility facade for the LoRA training strategy."""

from pubmedqa.training.strategies.lora import (
    LoRAAdapterMetric,
    LoRAFineTuneCliConfig,
    LoRAFineTuneConfig,
    PubMedQALoRAFineTuner,
    load_target_layers,
    main,
    normalize_lora_target_modules,
)

_load_target_layers = load_target_layers
_normalize_lora_target_modules = normalize_lora_target_modules

__all__ = [
    "LoRAAdapterMetric",
    "LoRAFineTuneCliConfig",
    "LoRAFineTuneConfig",
    "PubMedQALoRAFineTuner",
    "load_target_layers",
    "main",
    "normalize_lora_target_modules",
]


if __name__ == "__main__":
    main()
