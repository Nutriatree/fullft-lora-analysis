"""Shared base-model loading policy; training methods choose their own preparation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class ModelLoadOptions:
    device: torch.device
    dtype: Any
    distributed_mode: str = "single"
    attn_implementation: str | None = None
    trust_remote_code: bool = False
    hf_token: str | None = None


def model_load_device(options) -> torch.device:
    """FSDP loads/evaluates on CPU; only nested FSDP moves training units to GPU."""
    return (
        torch.device("cpu")
        if getattr(options, "distributed_mode", "single") == "fsdp"
        else options.device
    )


def model_load_dtype(options):
    # Flattened FSDP units require uniform dtype, including PEFT's float32
    # adapters. FP32 masters also keep optimizer updates out of low precision.
    return (
        torch.float32
        if getattr(options, "distributed_mode", "single") == "fsdp"
        else options.dtype
    )


def _load_pretrained(
    model_name_or_path: str,
    *,
    options: ModelLoadOptions,
    allow_multimodal_fallback: bool = False,
) -> tuple[Any, torch.nn.Module]:
    common_kwargs = {
        "token": options.hf_token,
        "trust_remote_code": options.trust_remote_code,
    }
    model_kwargs = {
        **common_kwargs,
        "torch_dtype": model_load_dtype(options),
    }
    if options.attn_implementation is not None:
        model_kwargs["attn_implementation"] = options.attn_implementation
    if getattr(options, "distributed_mode", "single") == "fsdp":
        model_kwargs["attn_implementation"] = "eager"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **common_kwargs)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError(
                f"Tokenizer for {model_name_or_path} has neither pad_token nor eos_token."
            )
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    try:
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    except ValueError:
        # Only historical Full FT supports the image/text model fallback.
        if not allow_multimodal_fallback:
            raise
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(
            model_name_or_path, **model_kwargs
        )

    model.to(model_load_device(options))
    return tokenizer, model
