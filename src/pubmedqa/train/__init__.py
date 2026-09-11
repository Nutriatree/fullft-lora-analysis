"""Training concepts, with lazy public aliases resolved at their actual owner."""

from importlib import import_module

_RECORD_MODULES = {
    "TrainStepLog": "pubmedqa.train.loop",
    "EvalPrediction": "pubmedqa.eval.validation",
    "EvalMetrics": "pubmedqa.eval.validation",
    "EvalResult": "pubmedqa.eval.validation",
    "CheckpointRecord": "pubmedqa.train.checkpoints",
    "LayerwiseReference": "pubmedqa.train.full_ft",
    "LayerwiseUpdateRecord": "pubmedqa.train.full_ft",
    "TrainingSummary": "pubmedqa.train.artifacts",
}
__all__ = list(_RECORD_MODULES)


def __getattr__(name):
    # Keep package-only imports lightweight, without depending on old contracts.
    if name in _RECORD_MODULES:
        return getattr(import_module(_RECORD_MODULES[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
