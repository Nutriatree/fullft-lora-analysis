"""Shared runtime setting objects for PubMedQA evaluation and training."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _default_cpu_threads() -> int:
    return max(1, (os.cpu_count() or 1) - 2)


def parse_checkpoint_percents(
    raw: str | None,
    default: tuple[int, ...],
) -> tuple[int, ...]:
    if raw is None:
        return default
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    return values or default


@dataclass(frozen=True)
class EvalSettings:
    env_test_path: str = "PUBMEDQA_TEST_PATH"
    env_expected_test_size: str = "PUBMEDQA_EXPECTED_TEST_SIZE"
    env_models: str = "PUBMEDQA_MODELS"
    env_attn_implementation: str = "PUBMEDQA_ATTN_IMPLEMENTATION"
    env_batch_size: str = "PUBMEDQA_BATCH_SIZE"
    env_backend: str = "PUBMEDQA_BACKEND"
    env_device: str = "PUBMEDQA_DEVICE"
    env_dtype: str = "PUBMEDQA_DTYPE"
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
    default_backend: str = "auto"
    default_device: str = "auto"
    default_dtype: str = "auto"
    default_max_new_tokens: int = 4
    default_attn_implementation: str = "auto"
    default_cpu_threads: int = _default_cpu_threads()
    default_strict_parser: bool = False
    default_expected_test_size: int = 500


@dataclass(frozen=True)
class TrainFullFineTuneSettings:
    env_run_id: str = "PUBMEDQA_RUN_ID"
    env_model_name: str = "PUBMEDQA_MODEL_NAME"
    env_condition: str = "PUBMEDQA_CONDITION"
    env_train_path: str = "PUBMEDQA_TRAIN_PATH"
    env_validation_path: str = "PUBMEDQA_VALIDATION_PATH"
    env_test_path: str = "PUBMEDQA_TEST_PATH"
    env_output_dir: str = "PUBMEDQA_OUTPUT_DIR"
    env_num_epochs: str = "PUBMEDQA_NUM_EPOCHS"
    env_train_batch_size: str = "PUBMEDQA_TRAIN_BATCH_SIZE"
    env_eval_batch_size: str = "PUBMEDQA_EVAL_BATCH_SIZE"
    env_grad_accum_steps: str = "PUBMEDQA_GRAD_ACCUM_STEPS"
    env_learning_rate: str = "PUBMEDQA_LEARNING_RATE"
    env_weight_decay: str = "PUBMEDQA_WEIGHT_DECAY"
    env_warmup_ratio: str = "PUBMEDQA_WARMUP_RATIO"
    env_max_grad_norm: str = "PUBMEDQA_MAX_GRAD_NORM"
    env_max_input_tokens: str = "PUBMEDQA_MAX_INPUT_TOKENS"
    env_max_new_tokens: str = "PUBMEDQA_MAX_NEW_TOKENS"
    env_device: str = "PUBMEDQA_DEVICE"
    env_dtype: str = "PUBMEDQA_DTYPE"
    env_attn_implementation: str = "PUBMEDQA_ATTN_IMPLEMENTATION"
    env_trust_remote_code: str = "PUBMEDQA_TRUST_REMOTE_CODE"
    env_cpu_threads: str = "PUBMEDQA_CPU_THREADS"
    env_log_every_steps: str = "PUBMEDQA_LOG_EVERY_STEPS"
    env_save_every_epoch: str = "PUBMEDQA_SAVE_EVERY_EPOCH"
    env_eval_every_epoch: str = "PUBMEDQA_EVAL_EVERY_EPOCH"
    env_max_train_examples: str = "PUBMEDQA_MAX_TRAIN_EXAMPLES"
    env_max_validation_examples: str = "PUBMEDQA_MAX_VALIDATION_EXAMPLES"
    env_max_test_examples: str = "PUBMEDQA_MAX_TEST_EXAMPLES"
    env_num_workers: str = "PUBMEDQA_NUM_WORKERS"
    env_gradient_checkpointing: str = "PUBMEDQA_GRADIENT_CHECKPOINTING"
    env_save_optimizer_state: str = "PUBMEDQA_SAVE_OPTIMIZER_STATE"
    env_strict_parser: str = "PUBMEDQA_STRICT_PARSER"
    env_seed: str = "PUBMEDQA_SEED"
    env_method_name: str = "PUBMEDQA_METHOD_NAME"
    env_run_tag: str = "PUBMEDQA_RUN_TAG"
    env_data_regime: str = "PUBMEDQA_DATA_REGIME"
    env_data_fraction: str = "PUBMEDQA_DATA_FRACTION"
    env_distributed_mode: str = "PUBMEDQA_DISTRIBUTED_MODE"
    env_fsdp_cpu_offload: str = "PUBMEDQA_FSDP_CPU_OFFLOAD"
    default_model_name: str = "Qwen/Qwen3-1.7B"
    default_condition: str = "full-ft"
    default_output_dir: Path = Path("outputs/pubmedqa_train")
    default_num_epochs: int = 3
    default_train_batch_size: int = 2
    default_eval_batch_size: int = 4
    default_grad_accum_steps: int = 8
    default_learning_rate: float = 2e-5
    default_weight_decay: float = 0.01
    default_warmup_ratio: float = 0.03
    default_max_grad_norm: float = 1.0
    default_max_new_tokens: int = 4
    default_device: str = "auto"
    default_dtype: str = "bf16"
    default_attn_implementation: str = "sdpa"
    default_cpu_threads: int = _default_cpu_threads()
    default_log_every_steps: int = 10
    default_num_workers: int = 0
    default_save_every_epoch: bool = True
    default_eval_every_epoch: bool = True
    default_gradient_checkpointing: bool = False
    default_save_optimizer_state: bool = True
    default_strict_parser: bool = False
    default_seed: int = 42
    default_method_name: str = "full-ft"
    default_run_tag: str = "F1"
    default_data_regime: str = "full-data"
    default_data_fraction: float = 1.0
    default_distributed_mode: str = "single"
    default_fsdp_cpu_offload: bool = False


@dataclass(frozen=True)
class TrainLoraSettings:
    env_target_modules: str = "PUBMEDQA_LORA_TARGET_MODULES"
    env_target_layers: str = "PUBMEDQA_LORA_TARGET_LAYERS"
    env_target_layers_file: str = "PUBMEDQA_LORA_TARGET_LAYERS_FILE"
    env_layer_scope: str = "PUBMEDQA_LORA_LAYER_SCOPE"
    env_lora_rank: str = "PUBMEDQA_LORA_RANK"
    env_lora_alpha: str = "PUBMEDQA_LORA_ALPHA"
    env_lora_dropout: str = "PUBMEDQA_LORA_DROPOUT"
    env_lora_bias: str = "PUBMEDQA_LORA_BIAS"
    env_lora_task_type: str = "PUBMEDQA_LORA_TASK_TYPE"
    env_modules_to_save: str = "PUBMEDQA_LORA_MODULES_TO_SAVE"
    env_merge_for_eval: str = "PUBMEDQA_LORA_MERGE_FOR_EVAL"
    default_method_name: str = "lora"
    default_condition: str = "lora"
    default_run_tag: str = "L1"
    default_target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    default_target_layers: tuple[int, ...] = ()
    default_layer_scope: str = "all"
    default_lora_rank: int = 8
    default_lora_alpha: float = 16.0
    default_lora_dropout: float = 0.0
    default_lora_bias: str = "none"
    default_lora_task_type: str = "CAUSAL_LM"
    default_modules_to_save: tuple[str, ...] = ()
    default_merge_for_eval: bool = False


@dataclass(frozen=True)
class TrainLayerSettings:
    env_target_modules: str = "PUBMEDQA_TARGET_MODULES"
    env_target_layers: str = "PUBMEDQA_TARGET_LAYERS"
    env_layer_scope: str = "PUBMEDQA_LAYER_SCOPE"
    env_lora_rank: str = "PUBMEDQA_LORA_RANK"
    env_lora_alpha: str = "PUBMEDQA_LORA_ALPHA"
    env_lora_dropout: str = "PUBMEDQA_LORA_DROPOUT"
    env_notes: str = "PUBMEDQA_NOTES"
    env_track_layerwise_updates: str = "PUBMEDQA_TRACK_LAYERWISE_UPDATES"
    env_checkpoint_percents: str = "PUBMEDQA_CHECKPOINT_PERCENTS"
    default_target_modules: tuple[str, ...] = ()
    default_target_layers: tuple[int, ...] = ()
    default_layer_scope: str = "all"
    default_lora_rank: int | None = None
    default_lora_alpha: float | None = None
    default_lora_dropout: float | None = None
    default_notes: str | None = None
    default_track_layerwise_updates: bool = True
    default_checkpoint_percents: tuple[int, ...] = (25, 50, 75, 100)
    default_tracked_module_suffixes: tuple[tuple[str, tuple[str, str]], ...] = (
        ("self_attn.q_proj.weight", ("Q", "attention")),
        ("self_attn.k_proj.weight", ("K", "attention")),
        ("self_attn.v_proj.weight", ("V", "attention")),
        ("self_attn.o_proj.weight", ("O", "attention")),
        ("mlp.gate_proj.weight", ("gate", "mlp")),
        ("mlp.up_proj.weight", ("up", "mlp")),
        ("mlp.down_proj.weight", ("down", "mlp")),
    )


EVAL_CONFIG = EvalSettings()
TRAIN_FULL_FINE_TUNE_CONFIG = TrainFullFineTuneSettings()
TRAIN_LORA_CONFIG = TrainLoraSettings()
TRAIN_LAYER_CONFIG = TrainLayerSettings()
