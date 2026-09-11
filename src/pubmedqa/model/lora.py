"""LoRA model and adapter preparation; optimizer updates use the shared train loop."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from pubmedqa.model.loading import _load_pretrained, model_load_device


@dataclass(frozen=True)
class AdapterOptions:
    lora_rank: int
    lora_alpha: float
    lora_dropout: float
    target_modules: tuple[str, ...]
    target_layers: tuple[int, ...] = ()
    layer_scope: str = "all"
    lora_bias: str = "none"
    lora_task_type: str = "CAUSAL_LM"
    modules_to_save: tuple[str, ...] = ()
    merge_for_eval: bool = False
    gradient_checkpointing: bool = False

    @classmethod
    def from_config(cls, config):
        return cls(**{item.name: getattr(config, item.name) for item in fields(cls)})


def _import_peft():
    try:
        from peft import LoraConfig, PeftConfig, PeftModel, TaskType, get_peft_model
    except ImportError as exc:
        raise RuntimeError(
            "peft is required for LoRA fine-tuning. Activate the configured conda environment first."
        ) from exc
    return LoraConfig, PeftConfig, PeftModel, TaskType, get_peft_model


def _build_peft_task_type(adapter, task_type_enum: Any) -> Any:
    if hasattr(task_type_enum, adapter.lora_task_type):
        return getattr(task_type_enum, adapter.lora_task_type)
    return adapter.lora_task_type


def load_lora_model(
    model_name_or_path: str, *, options, adapter: AdapterOptions
) -> tuple[Any, torch.nn.Module]:
    LoraConfig, PeftConfig, PeftModel, TaskType, get_peft_model = _import_peft()
    path = Path(model_name_or_path)
    adapter_config_path = path / "adapter_config.json"

    if path.is_dir() and adapter_config_path.is_file():
        peft_config = PeftConfig.from_pretrained(str(path))
        tokenizer_source = (
            str(path)
            if (path / "tokenizer_config.json").is_file()
            else peft_config.base_model_name_or_path
        )
        tokenizer, base_model = _load_pretrained(
            peft_config.base_model_name_or_path, options=options
        )
        if tokenizer_source != peft_config.base_model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_source, token=options.hf_token
            )
            if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"
        model = PeftModel.from_pretrained(base_model, str(path), is_trainable=False)
        if adapter.merge_for_eval:
            model = model.merge_and_unload()
            model.to(model_load_device(options))
        else:
            model.to(model_load_device(options))
        return tokenizer, model

    tokenizer, base_model = _load_pretrained(model_name_or_path, options=options)
    if adapter.gradient_checkpointing:
        base_model.gradient_checkpointing_enable()
        if hasattr(base_model.config, "use_cache"):
            base_model.config.use_cache = False

    peft_task_type = _build_peft_task_type(adapter, TaskType)
    lora_arguments: dict[str, Any] = {
        "r": adapter.lora_rank,
        "lora_alpha": adapter.lora_alpha,
        "lora_dropout": adapter.lora_dropout,
        "bias": adapter.lora_bias,
        "task_type": peft_task_type,
        "target_modules": list(adapter.target_modules),
    }
    if adapter.target_layers:
        lora_arguments["layers_to_transform"] = list(adapter.target_layers)
        lora_arguments["layers_pattern"] = ["layers"]
    if adapter.modules_to_save:
        lora_arguments["modules_to_save"] = list(adapter.modules_to_save)

    lora_config = LoraConfig(**lora_arguments)
    model = get_peft_model(base_model, lora_config)
    if adapter.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.to(model_load_device(options))
    return tokenizer, model
