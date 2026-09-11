"""Inference configuration and lazily exposed result records."""

from importlib import import_module

from pubmedqa.config.eval import ModelRuntimeConfig

__all__ = ["EvalItem", "EvalSummary", "ModelRuntimeConfig"]


def __getattr__(name):
    if name in {"EvalItem", "EvalSummary"}:
        return getattr(import_module("pubmedqa.eval.inference"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
