"""PyTorch-specific device, precision, and memory helpers."""

from __future__ import annotations

import torch


def resolve_dtype(name: str) -> torch.dtype:
    normalized = name.strip().lower()
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype {name!r}. Use bf16, fp16, or fp32.")
    return mapping[normalized]


def resolve_device(requested: str) -> torch.device:
    normalized = requested.strip().lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if normalized == "cuda" and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device(normalized)


def supports_cuda_amp(dtype: torch.dtype, device: torch.device) -> bool:
    return device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}


def count_parameters(model: torch.nn.Module) -> tuple[int, int, float]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable, 0.0 if total == 0 else trainable / total


def memory_snapshot(device: torch.device) -> dict[str, float | None]:
    if device.type != "cuda":
        return {
            "allocated_gb": None,
            "reserved_gb": None,
            "max_allocated_gb": None,
            "max_reserved_gb": None,
        }
    return {
        "allocated_gb": torch.cuda.memory_allocated(device) / (1024**3),
        "reserved_gb": torch.cuda.memory_reserved(device) / (1024**3),
        "max_allocated_gb": torch.cuda.max_memory_allocated(device) / (1024**3),
        "max_reserved_gb": torch.cuda.max_memory_reserved(device) / (1024**3),
    }


def configure_parallelism(cpu_threads: int) -> None:
    import os

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    torch.set_num_threads(max(1, cpu_threads))
    try:
        torch.set_num_interop_threads(max(1, min(4, cpu_threads)))
    except RuntimeError:
        pass
