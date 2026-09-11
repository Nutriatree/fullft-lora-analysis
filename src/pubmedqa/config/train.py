"""Shared training configuration with optional LoRA adapter settings."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from pubmedqa.config.env import (
    EnvironmentConfig,
    default_cpu_threads,
)
from pubmedqa.config.env import (
    env_bool as _env_bool,
)
from pubmedqa.config.env import (
    env_float as _env_float,
)
from pubmedqa.config.env import (
    env_int as _env_int,
)
from pubmedqa.config.env import (
    env_int_tuple as _env_int_tuple,
)
from pubmedqa.config.env import (
    env_optional_int as _env_optional_int,
)
from pubmedqa.config.env import (
    env_optional_str as _env_optional_str,
)
from pubmedqa.config.env import (
    env_tuple as _env_tuple,
)

if TYPE_CHECKING:
    import torch


def parse_checkpoint_percents(
    raw: str | None,
    default: tuple[int, ...],
) -> tuple[int, ...]:
    if raw is None:
        return default
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    return values or default


@dataclass(frozen=True)
class TrainingSettings:
    """Environment keys and defaults shared by all training methods."""

    env_run_id: str = "PUBMEDQA_RUN_ID"
    env_model_name: str = "PUBMEDQA_MODEL_NAME"
    env_condition: str = "PUBMEDQA_CONDITION"
    env_method_name: str = "PUBMEDQA_METHOD_NAME"
    env_run_tag: str = "PUBMEDQA_RUN_TAG"

    env_train_path: str = "PUBMEDQA_TRAIN_PATH"
    env_validation_path: str = "PUBMEDQA_VALIDATION_PATH"
    env_test_path: str = "PUBMEDQA_TEST_PATH"
    env_data_regime: str = "PUBMEDQA_DATA_REGIME"
    env_data_fraction: str = "PUBMEDQA_DATA_FRACTION"
    env_max_train_examples: str = "PUBMEDQA_MAX_TRAIN_EXAMPLES"
    env_max_validation_examples: str = "PUBMEDQA_MAX_VALIDATION_EXAMPLES"
    env_max_test_examples: str = "PUBMEDQA_MAX_TEST_EXAMPLES"

    env_num_epochs: str = "PUBMEDQA_NUM_EPOCHS"
    env_train_batch_size: str = "PUBMEDQA_TRAIN_BATCH_SIZE"
    env_eval_batch_size: str = "PUBMEDQA_EVAL_BATCH_SIZE"
    env_grad_accum_steps: str = "PUBMEDQA_GRAD_ACCUM_STEPS"
    env_learning_rate: str = "PUBMEDQA_LEARNING_RATE"
    env_weight_decay: str = "PUBMEDQA_WEIGHT_DECAY"
    env_warmup_ratio: str = "PUBMEDQA_WARMUP_RATIO"
    env_max_grad_norm: str = "PUBMEDQA_MAX_GRAD_NORM"
    env_gradient_checkpointing: str = "PUBMEDQA_GRADIENT_CHECKPOINTING"
    env_max_input_tokens: str = "PUBMEDQA_MAX_INPUT_TOKENS"
    env_max_new_tokens: str = "PUBMEDQA_MAX_NEW_TOKENS"

    env_device: str = "PUBMEDQA_DEVICE"
    env_dtype: str = "PUBMEDQA_DTYPE"
    env_attn_implementation: str = "PUBMEDQA_ATTN_IMPLEMENTATION"
    env_trust_remote_code: str = "PUBMEDQA_TRUST_REMOTE_CODE"
    env_cpu_threads: str = "PUBMEDQA_CPU_THREADS"
    env_num_workers: str = "PUBMEDQA_NUM_WORKERS"
    env_seed: str = "PUBMEDQA_SEED"
    env_distributed_mode: str = "PUBMEDQA_DISTRIBUTED_MODE"
    env_fsdp_cpu_offload: str = "PUBMEDQA_FSDP_CPU_OFFLOAD"

    env_output_dir: str = "PUBMEDQA_OUTPUT_DIR"
    env_log_every_steps: str = "PUBMEDQA_LOG_EVERY_STEPS"
    env_save_every_epoch: str = "PUBMEDQA_SAVE_EVERY_EPOCH"
    env_eval_every_epoch: str = "PUBMEDQA_EVAL_EVERY_EPOCH"
    env_save_optimizer_state: str = "PUBMEDQA_SAVE_OPTIMIZER_STATE"
    env_strict_parser: str = "PUBMEDQA_STRICT_PARSER"

    default_model_name: str = "Qwen/Qwen3-1.7B"
    default_condition: str = "full-ft"
    default_method_name: str = "full-ft"
    default_run_tag: str = "F1"
    default_distributed_mode: str = "single"
    default_fsdp_cpu_offload: bool = False
    default_output_dir: Path = Path("outputs/pubmedqa_train")

    default_num_epochs: int = 1
    default_train_batch_size: int = 1
    default_eval_batch_size: int = 4
    default_grad_accum_steps: int = 8
    default_learning_rate: float = 2e-5
    default_weight_decay: float = 0.01
    default_warmup_ratio: float = 0.03
    default_max_grad_norm: float = 1.0
    default_gradient_checkpointing: bool = True

    default_device: str = "cuda"
    default_dtype: str = "bf16"
    default_attn_implementation: str = "sdpa"
    default_cpu_threads: int = default_cpu_threads()
    default_num_workers: int = 0
    default_max_new_tokens: int = 4

    default_log_every_steps: int = 10
    default_save_every_epoch: bool = True
    default_eval_every_epoch: bool = True
    default_save_optimizer_state: bool = False
    default_strict_parser: bool = False
    default_seed: int = 42
    default_data_regime: str = "full-data"
    default_data_fraction: float = 1.0


@dataclass(frozen=True)
class TrainLayerSettings:
    """Layer-wise and temporal-analysis environment keys and defaults."""

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


TRAIN_CONFIG = TrainingSettings()


TRAIN_LAYER_CONFIG = TrainLayerSettings()


@dataclass(frozen=True)
class LoRASettings:
    env_learning_rate: str = "PUBMEDQA_LORA_LEARNING_RATE"
    env_gradient_checkpointing: str = "PUBMEDQA_LORA_GRADIENT_CHECKPOINTING"
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

    default_condition: str = "lora"
    default_method_name: str = "lora"
    default_run_tag: str = "L1"
    default_target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    default_target_layers: tuple[int, ...] = ()
    default_layer_scope: str = "all"
    default_lora_rank: int = 8
    default_lora_alpha: float = 16.0
    default_lora_dropout: float = 0.0
    default_lora_bias: str = "none"
    default_lora_task_type: str = "CAUSAL_LM"
    default_learning_rate: float = 1e-4
    default_gradient_checkpointing: bool = False
    default_modules_to_save: tuple[str, ...] = ()
    default_merge_for_eval: bool = False


TRAIN_LORA_CONFIG = LoRASettings()


def load_target_layers(raw: str | None, file_path: str | None) -> tuple[int, ...]:
    layers = (
        [int(part.strip()) for part in raw.split(",") if part.strip()] if raw else []
    )
    if file_path:
        payload = json.loads(Path(file_path).read_text(encoding="utf-8"))
        file_layers = payload.get("target_layers", [])
        if not isinstance(file_layers, list):
            raise ValueError(
                "target_layers file must contain a list in 'target_layers'."
            )
        layers.extend(int(value) for value in file_layers)
    return tuple(sorted(set(layers)))


def normalize_lora_target_modules(values: Sequence[str]) -> tuple[str, ...]:
    alias_map = {
        "q": "q_proj",
        "k": "k_proj",
        "v": "v_proj",
        "o": "o_proj",
        "q_proj": "q_proj",
        "k_proj": "k_proj",
        "v_proj": "v_proj",
        "o_proj": "o_proj",
        "gate": "gate_proj",
        "up": "up_proj",
        "down": "down_proj",
        "gate_proj": "gate_proj",
        "up_proj": "up_proj",
        "down_proj": "down_proj",
    }
    normalized = []
    for value in values:
        key = value.strip().lower()
        if not key:
            continue
        if key not in alias_map:
            raise ValueError(f"Unsupported LoRA target module {value!r}.")
        resolved = alias_map[key]
        if resolved not in normalized:
            normalized.append(resolved)
    return tuple(normalized)


def normalize_checkpoint_percents(values: Sequence[int]) -> tuple[int, ...]:
    normalized = sorted({value for value in values if 0 < value <= 100})
    if not normalized:
        return TRAIN_LAYER_CONFIG.default_checkpoint_percents
    if normalized[-1] != 100:
        normalized.append(100)
    return tuple(normalized)


def validate_training_config(config):
    for name in (
        "num_epochs",
        "train_batch_size",
        "eval_batch_size",
        "gradient_accumulation_steps",
        "max_new_tokens",
        "cpu_threads",
    ):
        value = getattr(config, name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    for name in (
        "max_train_examples",
        "max_validation_examples",
        "max_test_examples",
        "max_input_tokens",
    ):
        value = getattr(config, name)
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive when set")
    if config.num_workers < 0:
        raise ValueError("num_workers must be nonnegative")
    for name in ("learning_rate", "max_grad_norm"):
        value = getattr(config, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not math.isfinite(config.weight_decay) or config.weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if not 0 <= config.warmup_ratio <= 1:
        raise ValueError("warmup_ratio must be between 0 and 1")
    if config.save_every_epoch and not config.eval_every_epoch:
        raise ValueError(
            "save_every_epoch requires eval_every_epoch for fresh checkpoint metrics"
        )


@dataclass(frozen=True)
class LoRAOptions:
    rank: int
    alpha: float
    dropout: float
    target_modules: tuple[str, ...]
    target_layers: tuple[int, ...] = ()
    layer_scope: str = "all"
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    modules_to_save: tuple[str, ...] = ()
    merge_for_eval: bool = False


@dataclass(frozen=True)
class TrainingConfig:
    run_id: str
    run_tag: str
    method_name: str
    model_name: str
    condition: str
    data_regime: str
    data_fraction: float
    train_path: Path
    validation_path: Path
    test_path: Path | None
    output_dir: Path
    num_epochs: int
    train_batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    max_grad_norm: float
    max_input_tokens: int | None
    max_new_tokens: int
    device: str
    dtype: torch.dtype
    attn_implementation: str | None
    trust_remote_code: bool
    cpu_threads: int
    log_every_steps: int
    save_every_epoch: bool
    eval_every_epoch: bool
    max_train_examples: int | None
    max_validation_examples: int | None
    max_test_examples: int | None
    num_workers: int
    gradient_checkpointing: bool
    save_optimizer_state: bool
    strict_parser: bool
    seed: int
    notes: str | None
    track_layerwise_updates: bool
    checkpoint_percents: tuple[int, ...]
    adapter: LoRAOptions | None = None
    distributed_mode: str = "single"
    fsdp_cpu_offload: bool = False

    @property
    def target_modules(self) -> tuple[str, ...]:
        return () if self.adapter is None else self.adapter.target_modules

    @property
    def target_layers(self) -> tuple[int, ...]:
        return () if self.adapter is None else self.adapter.target_layers

    @property
    def layer_scope(self) -> str:
        return "all" if self.adapter is None else self.adapter.layer_scope

    @property
    def lora_rank(self) -> int | None:
        return None if self.adapter is None else self.adapter.rank

    @property
    def lora_alpha(self) -> float | None:
        return None if self.adapter is None else self.adapter.alpha

    @property
    def lora_dropout(self) -> float | None:
        return None if self.adapter is None else self.adapter.dropout

    @property
    def lora_bias(self) -> str:
        return "none" if self.adapter is None else self.adapter.bias

    @property
    def lora_task_type(self) -> str:
        return "CAUSAL_LM" if self.adapter is None else self.adapter.task_type

    @property
    def modules_to_save(self) -> tuple[str, ...]:
        return () if self.adapter is None else self.adapter.modules_to_save

    @property
    def merge_for_eval(self) -> bool:
        return False if self.adapter is None else self.adapter.merge_for_eval


@dataclass(frozen=True)
class TrainingCliConfig:
    config: TrainingConfig
    environment: EnvironmentConfig

    @classmethod
    def from_env(cls, method: str = "full-ft") -> "TrainingCliConfig":
        from pubmedqa.model.device import resolve_dtype as _resolve_dtype

        if method not in {"full-ft", "lora"}:
            raise ValueError("method must be 'full-ft' or 'lora'")
        is_lora = method == "lora"

        train_path = os.getenv(TRAIN_CONFIG.env_train_path)
        validation_path = os.getenv(TRAIN_CONFIG.env_validation_path)
        if not train_path or not validation_path:
            raise RuntimeError(
                f"{TRAIN_CONFIG.env_train_path} and {TRAIN_CONFIG.env_validation_path} are required for training."
            )

        attn = (
            os.getenv(
                TRAIN_CONFIG.env_attn_implementation,
                TRAIN_CONFIG.default_attn_implementation,
            )
            .strip()
            .lower()
        )
        if attn in {"", "none", "auto"}:
            attn = None

        adapter = None
        if is_lora:
            adapter = LoRAOptions(
                rank=_env_int(
                    TRAIN_LORA_CONFIG.env_lora_rank,
                    TRAIN_LORA_CONFIG.default_lora_rank,
                ),
                alpha=_env_float(
                    TRAIN_LORA_CONFIG.env_lora_alpha,
                    TRAIN_LORA_CONFIG.default_lora_alpha,
                ),
                dropout=_env_float(
                    TRAIN_LORA_CONFIG.env_lora_dropout,
                    TRAIN_LORA_CONFIG.default_lora_dropout,
                ),
                target_modules=normalize_lora_target_modules(
                    _env_tuple(TRAIN_LORA_CONFIG.env_target_modules)
                    or TRAIN_LORA_CONFIG.default_target_modules
                ),
                target_layers=load_target_layers(
                    os.getenv(TRAIN_LORA_CONFIG.env_target_layers),
                    os.getenv(TRAIN_LORA_CONFIG.env_target_layers_file),
                )
                or TRAIN_LORA_CONFIG.default_target_layers,
                layer_scope=os.getenv(
                    TRAIN_LORA_CONFIG.env_layer_scope,
                    TRAIN_LORA_CONFIG.default_layer_scope,
                ),
                bias=os.getenv(
                    TRAIN_LORA_CONFIG.env_lora_bias,
                    TRAIN_LORA_CONFIG.default_lora_bias,
                ),
                task_type=os.getenv(
                    TRAIN_LORA_CONFIG.env_lora_task_type,
                    TRAIN_LORA_CONFIG.default_lora_task_type,
                ),
                modules_to_save=_env_tuple(TRAIN_LORA_CONFIG.env_modules_to_save)
                or TRAIN_LORA_CONFIG.default_modules_to_save,
                merge_for_eval=_env_bool(
                    TRAIN_LORA_CONFIG.env_merge_for_eval,
                    TRAIN_LORA_CONFIG.default_merge_for_eval,
                ),
            )

        config = TrainingConfig(
            run_id=os.getenv(TRAIN_CONFIG.env_run_id) or time.strftime("%Y%m%d_%H%M%S"),
            run_tag=os.getenv(
                TRAIN_CONFIG.env_run_tag,
                TRAIN_LORA_CONFIG.default_run_tag
                if is_lora
                else TRAIN_CONFIG.default_run_tag,
            ),
            method_name=os.getenv(
                TRAIN_CONFIG.env_method_name,
                method,
            ),
            model_name=os.getenv(
                TRAIN_CONFIG.env_model_name,
                TRAIN_CONFIG.default_model_name,
            ),
            condition=os.getenv(
                TRAIN_CONFIG.env_condition,
                TRAIN_LORA_CONFIG.default_condition
                if is_lora
                else TRAIN_CONFIG.default_condition,
            ),
            data_regime=os.getenv(
                TRAIN_CONFIG.env_data_regime,
                TRAIN_CONFIG.default_data_regime,
            ),
            data_fraction=_env_float(
                TRAIN_CONFIG.env_data_fraction,
                TRAIN_CONFIG.default_data_fraction,
            ),
            train_path=Path(train_path),
            validation_path=Path(validation_path),
            test_path=Path(os.getenv(TRAIN_CONFIG.env_test_path))
            if os.getenv(TRAIN_CONFIG.env_test_path)
            else None,
            output_dir=Path(
                os.getenv(
                    TRAIN_CONFIG.env_output_dir,
                    str(TRAIN_CONFIG.default_output_dir),
                )
            ),
            num_epochs=_env_int(
                TRAIN_CONFIG.env_num_epochs,
                TRAIN_CONFIG.default_num_epochs,
            ),
            train_batch_size=_env_int(
                TRAIN_CONFIG.env_train_batch_size,
                TRAIN_CONFIG.default_train_batch_size,
            ),
            eval_batch_size=_env_int(
                TRAIN_CONFIG.env_eval_batch_size,
                TRAIN_CONFIG.default_eval_batch_size,
            ),
            gradient_accumulation_steps=_env_int(
                TRAIN_CONFIG.env_grad_accum_steps,
                TRAIN_CONFIG.default_grad_accum_steps,
            ),
            learning_rate=_env_float(
                TRAIN_LORA_CONFIG.env_learning_rate
                if is_lora
                else TRAIN_CONFIG.env_learning_rate,
                TRAIN_LORA_CONFIG.default_learning_rate
                if is_lora
                else TRAIN_CONFIG.default_learning_rate,
            ),
            weight_decay=_env_float(
                TRAIN_CONFIG.env_weight_decay,
                TRAIN_CONFIG.default_weight_decay,
            ),
            warmup_ratio=_env_float(
                TRAIN_CONFIG.env_warmup_ratio,
                TRAIN_CONFIG.default_warmup_ratio,
            ),
            max_grad_norm=_env_float(
                TRAIN_CONFIG.env_max_grad_norm,
                TRAIN_CONFIG.default_max_grad_norm,
            ),
            max_input_tokens=_env_optional_int(TRAIN_CONFIG.env_max_input_tokens),
            max_new_tokens=_env_int(
                TRAIN_CONFIG.env_max_new_tokens,
                TRAIN_CONFIG.default_max_new_tokens,
            ),
            device=os.getenv(
                TRAIN_CONFIG.env_device,
                TRAIN_CONFIG.default_device,
            ),
            dtype=_resolve_dtype(
                os.getenv(
                    TRAIN_CONFIG.env_dtype,
                    TRAIN_CONFIG.default_dtype,
                )
            ),
            attn_implementation=attn,
            trust_remote_code=_env_bool(TRAIN_CONFIG.env_trust_remote_code, False),
            cpu_threads=_env_int(
                TRAIN_CONFIG.env_cpu_threads,
                TRAIN_CONFIG.default_cpu_threads,
            ),
            log_every_steps=_env_int(
                TRAIN_CONFIG.env_log_every_steps,
                TRAIN_CONFIG.default_log_every_steps,
            ),
            save_every_epoch=_env_bool(
                TRAIN_CONFIG.env_save_every_epoch,
                TRAIN_CONFIG.default_save_every_epoch,
            ),
            eval_every_epoch=_env_bool(
                TRAIN_CONFIG.env_eval_every_epoch,
                TRAIN_CONFIG.default_eval_every_epoch,
            ),
            max_train_examples=_env_optional_int(TRAIN_CONFIG.env_max_train_examples),
            max_validation_examples=_env_optional_int(
                TRAIN_CONFIG.env_max_validation_examples
            ),
            max_test_examples=_env_optional_int(TRAIN_CONFIG.env_max_test_examples),
            num_workers=_env_int(
                TRAIN_CONFIG.env_num_workers,
                TRAIN_CONFIG.default_num_workers,
            ),
            gradient_checkpointing=_env_bool(
                TRAIN_LORA_CONFIG.env_gradient_checkpointing
                if is_lora
                else TRAIN_CONFIG.env_gradient_checkpointing,
                TRAIN_LORA_CONFIG.default_gradient_checkpointing
                if is_lora
                else TRAIN_CONFIG.default_gradient_checkpointing,
            ),
            save_optimizer_state=_env_bool(
                TRAIN_CONFIG.env_save_optimizer_state,
                TRAIN_CONFIG.default_save_optimizer_state,
            ),
            strict_parser=_env_bool(
                TRAIN_CONFIG.env_strict_parser,
                TRAIN_CONFIG.default_strict_parser,
            ),
            seed=_env_int(
                TRAIN_CONFIG.env_seed,
                TRAIN_CONFIG.default_seed,
            ),
            notes=_env_optional_str(TRAIN_LAYER_CONFIG.env_notes),
            track_layerwise_updates=_env_bool(
                TRAIN_LAYER_CONFIG.env_track_layerwise_updates,
                TRAIN_LAYER_CONFIG.default_track_layerwise_updates,
            ),
            checkpoint_percents=normalize_checkpoint_percents(
                _env_int_tuple(TRAIN_LAYER_CONFIG.env_checkpoint_percents)
                or TRAIN_LAYER_CONFIG.default_checkpoint_percents
            ),
            adapter=adapter,
            distributed_mode=os.getenv(TRAIN_CONFIG.env_distributed_mode, "single"),
            fsdp_cpu_offload=_env_bool(TRAIN_CONFIG.env_fsdp_cpu_offload, False),
        )
        return cls(config=config, environment=EnvironmentConfig.from_env())
