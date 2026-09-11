"""Established evaluation API; implementation is owned by pubmedqa.eval."""

from pubmedqa.config import EnvironmentConfig
from pubmedqa.config.eval import ModelRuntimeConfig
from pubmedqa.data.records import current_time_iso, safe_name, write_json, write_jsonl
from pubmedqa.eval.inference import (
    DEFAULT_BASE_MODELS,
    DEFAULT_MLX_MODEL_MAP,
    DEFAULT_MODEL_NAME,
    CliBatchConfig,
    EvalItem,
    EvalSummary,
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
from pubmedqa.eval.metrics import accuracy, macro_f1, resolve_metric_labels
from pubmedqa.labels import VALID_LABELS
from pubmedqa.model.device import configure_parallelism

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
