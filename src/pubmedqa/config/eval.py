"""Inference environment keys and defaults; importing these never loads a model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pubmedqa.config import default_cpu_threads


@dataclass(frozen=True)
class EvalSettings:
    """Evaluation environment keys and runtime defaults."""

    env_test_path: str = "PUBMEDQA_TEST_PATH"
    env_expected_test_size: str = "PUBMEDQA_EXPECTED_TEST_SIZE"
    env_models: str = "PUBMEDQA_MODELS"
    env_backend: str = "PUBMEDQA_BACKEND"
    env_batch_size: str = "PUBMEDQA_BATCH_SIZE"
    env_device: str = "PUBMEDQA_DEVICE"
    env_dtype: str = "PUBMEDQA_DTYPE"
    env_attn_implementation: str = "PUBMEDQA_ATTN_IMPLEMENTATION"
    env_max_new_tokens: str = "PUBMEDQA_MAX_NEW_TOKENS"
    env_max_input_tokens: str = "PUBMEDQA_MAX_INPUT_TOKENS"
    env_trust_remote_code: str = "PUBMEDQA_TRUST_REMOTE_CODE"
    env_cpu_threads: str = "PUBMEDQA_CPU_THREADS"
    env_strict_parser: str = "PUBMEDQA_STRICT_PARSER"
    env_output_dir: str = "PUBMEDQA_OUTPUT_DIR"
    env_condition: str = "PUBMEDQA_CONDITION"
    env_run_id: str = "PUBMEDQA_RUN_ID"

    default_model_name: str = "Qwen/Qwen3-1.7B"
    default_base_models: tuple[str, ...] = ("Qwen/Qwen3-1.7B",)
    default_output_dir: Path = Path("outputs/pubmedqa_eval")
    default_condition: str = "baseline"
    default_backend: str = "torch"
    default_device: str = "cuda"
    default_dtype: str = "bf16"
    default_attn_implementation: str = "sdpa"
    default_batch_size: int = 4
    default_max_new_tokens: int = 4
    default_cpu_threads: int = default_cpu_threads()
    default_strict_parser: bool = False
    default_expected_test_size: int = 500


EVAL_CONFIG = EvalSettings()


@dataclass(frozen=True)
class ModelRuntimeConfig:
    model_name: str
    backend: str = EVAL_CONFIG.default_backend  # auto | torch | mlx
    device: str = EVAL_CONFIG.default_device  # auto | cuda:0 | mps | cpu
    dtype: str = EVAL_CONFIG.default_dtype  # auto | bf16 | fp16 | fp32
    max_new_tokens: int = EVAL_CONFIG.default_max_new_tokens
    batch_size: int | None = None
    max_input_tokens: int | None = None
    attn_implementation: str | None = (
        EVAL_CONFIG.default_attn_implementation
    )  # auto | sdpa | eager | ... | none
    trust_remote_code: bool = False
    cpu_threads: int = EVAL_CONFIG.default_cpu_threads
    strict_parser: bool = EVAL_CONFIG.default_strict_parser
