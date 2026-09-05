"""Compatibility facade for :mod:`pubmedqa.inference`."""

from pubmedqa.config import EnvironmentConfig
from pubmedqa.domain.labels import VALID_LABELS
from pubmedqa.domain.metrics import accuracy, macro_f1, resolve_metric_labels
from pubmedqa.inference.contracts import EvalItem, EvalSummary, ModelRuntimeConfig
from pubmedqa.inference.runner import (
    DEFAULT_BASE_MODELS,
    DEFAULT_MLX_MODEL_MAP,
    DEFAULT_MODEL_NAME,
    CliBatchConfig,
    InferenceBackend,
    MLXBackend,
    PubMedQAEvaluationRunner,
    TorchBackend,
    build_title,
    create_backend,
    empty_device_cache,
    get_device_memory_gb,
    load_local_jsonl,
    main,
    print_runtime,
    reset_device_memory_stats,
    resolve_attention,
    resolve_backend,
    resolve_batch_size,
    resolve_torch_device,
    resolve_torch_dtype,
    save_results,
    synchronize_device,
)
from pubmedqa.runtime.io import current_time_iso, safe_name, write_json, write_jsonl
from pubmedqa.runtime.torch_runtime import configure_parallelism

__all__ = [
    "DEFAULT_BASE_MODELS",
    "DEFAULT_MODEL_NAME",
    "CliBatchConfig",
    "EnvironmentConfig",
    "EvalItem",
    "EvalSummary",
    "InferenceBackend",
    "ModelRuntimeConfig",
    "PubMedQAEvaluationRunner",
    "accuracy",
    "load_local_jsonl",
    "macro_f1",
    "resolve_metric_labels",
    "save_results",
]


if __name__ == "__main__":
    main()
