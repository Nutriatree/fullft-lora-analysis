"""Public configuration and result records for model inference."""

from __future__ import annotations

from dataclasses import dataclass

from pubmedqa.config import EVAL_CONFIG


@dataclass(frozen=True)
class ModelRuntimeConfig:
    model_name: str
    backend: str = EVAL_CONFIG.default_backend  # auto | torch | mlx
    device: str = EVAL_CONFIG.default_device  # auto | cuda:0 | mps | cpu
    dtype: str = EVAL_CONFIG.default_dtype  # auto | bf16 | fp16 | fp32
    max_new_tokens: int = EVAL_CONFIG.default_max_new_tokens
    batch_size: int | None = None
    max_input_tokens: int | None = None
    attn_implementation: str | None = EVAL_CONFIG.default_attn_implementation  # auto | sdpa | eager | ... | none
    trust_remote_code: bool = False
    cpu_threads: int = EVAL_CONFIG.default_cpu_threads
    strict_parser: bool = EVAL_CONFIG.default_strict_parser


@dataclass(frozen=True)
class EvalItem:
    index: int
    run_id: str
    model_name: str
    resolved_model_name: str
    backend: str
    condition: str
    title: str
    start_time: str
    end_time: str
    pubid: str
    question: str
    context: list[str]
    gold_label: str
    prompt: str
    response_text: str
    predicted_label: str | None
    parse_method: str | None
    parse_error: str | None


@dataclass(frozen=True)
class EvalSummary:
    run_id: str
    model_name: str
    resolved_model_name: str
    backend: str
    condition: str
    title: str
    start_time: str
    end_time: str
    elapsed_seconds: float
    num_examples: int
    num_parsed: int
    accuracy: float
    macro_f1: float
    invalid_rate: float
    examples_per_second: float
    device: str
    dtype: str
    device_memory_gb: float | None
