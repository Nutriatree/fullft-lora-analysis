"""Public API for PubMedQA experiments.

Trainer classes are loaded lazily so prompt and parsing utilities can be used without
initializing the PyTorch, Transformers, and PEFT stacks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pubmedqa.full_finetune import PubMedQAFullFineTuner
    from pubmedqa.lora_finetune import PubMedQALoRAFineTuner

__all__ = [
    "PubMedQAFullFineTuner",
    "PubMedQALoRAFineTuner",
]


def __getattr__(name: str) -> Any:
    if name == "PubMedQAFullFineTuner":
        from pubmedqa.full_finetune import PubMedQAFullFineTuner

        return PubMedQAFullFineTuner
    if name == "PubMedQALoRAFineTuner":
        from pubmedqa.lora_finetune import PubMedQALoRAFineTuner

        return PubMedQALoRAFineTuner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
