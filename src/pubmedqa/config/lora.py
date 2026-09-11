"""LoRA-specific settings, target selection and executable configuration."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from pubmedqa.config import (
    EnvironmentConfig,
)
from pubmedqa.config import (
    env_bool as _env_bool,
)
from pubmedqa.config import (
    env_float as _env_float,
)
from pubmedqa.config import (
    env_int as _env_int,
)
from pubmedqa.config import (
    env_int_tuple as _env_int_tuple,
)
from pubmedqa.config import (
    env_optional_int as _env_optional_int,
)
from pubmedqa.config import (
    env_optional_str as _env_optional_str,
)
from pubmedqa.config import (
    env_tuple as _env_tuple,
)
from pubmedqa.config.full_ft import (
    TRAIN_FULL_FINE_TUNE_CONFIG,
    TRAIN_LAYER_CONFIG,
    normalize_checkpoint_percents,
)
from pubmedqa.data.records import current_time_iso

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True)
class TrainLoraSettings:
    """LoRA-specific environment keys and defaults."""

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
    default_learning_rate: float = 1e-4
    default_gradient_checkpointing: bool = False
    default_modules_to_save: tuple[str, ...] = ()
    default_merge_for_eval: bool = False


TRAIN_LORA_CONFIG = TrainLoraSettings()


def load_target_layers(raw: str | None, file_path: str | None) -> tuple[int, ...]:
    layers: list[int] = []
    if raw:
        layers.extend(int(part.strip()) for part in raw.split(",") if part.strip())
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
    normalized: list[str] = []
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


@dataclass(frozen=True)
class LoRAFineTuneConfig:
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
    target_modules: tuple[str, ...]
    target_layers: tuple[int, ...]
    layer_scope: str
    lora_rank: int
    lora_alpha: float
    lora_dropout: float
    lora_bias: str
    lora_task_type: str
    modules_to_save: tuple[str, ...]
    merge_for_eval: bool
    notes: str | None
    track_layerwise_updates: bool
    checkpoint_percents: tuple[int, ...]
    distributed_mode: str = TRAIN_FULL_FINE_TUNE_CONFIG.default_distributed_mode
    fsdp_cpu_offload: bool = False


@dataclass(frozen=True)
class LoRAFineTuneCliConfig:
    config: LoRAFineTuneConfig
    environment: EnvironmentConfig

    @classmethod
    def from_env(cls) -> "LoRAFineTuneCliConfig":
        from pubmedqa.model.device import resolve_dtype as _resolve_dtype

        train_path = os.getenv(TRAIN_FULL_FINE_TUNE_CONFIG.env_train_path)
        validation_path = os.getenv(TRAIN_FULL_FINE_TUNE_CONFIG.env_validation_path)
        if not train_path or not validation_path:
            raise RuntimeError(
                f"{TRAIN_FULL_FINE_TUNE_CONFIG.env_train_path} and {TRAIN_FULL_FINE_TUNE_CONFIG.env_validation_path} are required for LoRA fine-tuning."
            )

        attn = (
            os.getenv(
                TRAIN_FULL_FINE_TUNE_CONFIG.env_attn_implementation,
                TRAIN_FULL_FINE_TUNE_CONFIG.default_attn_implementation,
            )
            .strip()
            .lower()
        )
        if attn in {"", "none", "auto"}:
            attn = None

        config = LoRAFineTuneConfig(
            run_id=os.getenv(TRAIN_FULL_FINE_TUNE_CONFIG.env_run_id)
            or current_time_iso().replace(":", "").replace("-", ""),
            run_tag=os.getenv(
                TRAIN_FULL_FINE_TUNE_CONFIG.env_run_tag,
                TRAIN_LORA_CONFIG.default_run_tag,
            ),
            method_name=os.getenv(
                TRAIN_FULL_FINE_TUNE_CONFIG.env_method_name,
                TRAIN_LORA_CONFIG.default_method_name,
            ),
            model_name=os.getenv(
                TRAIN_FULL_FINE_TUNE_CONFIG.env_model_name,
                TRAIN_FULL_FINE_TUNE_CONFIG.default_model_name,
            ),
            condition=os.getenv(
                TRAIN_FULL_FINE_TUNE_CONFIG.env_condition,
                TRAIN_LORA_CONFIG.default_condition,
            ),
            data_regime=os.getenv(
                TRAIN_FULL_FINE_TUNE_CONFIG.env_data_regime,
                TRAIN_FULL_FINE_TUNE_CONFIG.default_data_regime,
            ),
            data_fraction=_env_float(
                TRAIN_FULL_FINE_TUNE_CONFIG.env_data_fraction,
                TRAIN_FULL_FINE_TUNE_CONFIG.default_data_fraction,
            ),
            train_path=Path(train_path),
            validation_path=Path(validation_path),
            test_path=Path(os.getenv(TRAIN_FULL_FINE_TUNE_CONFIG.env_test_path))
            if os.getenv(TRAIN_FULL_FINE_TUNE_CONFIG.env_test_path)
            else None,
            output_dir=Path(
                os.getenv(
                    TRAIN_FULL_FINE_TUNE_CONFIG.env_output_dir,
                    str(TRAIN_FULL_FINE_TUNE_CONFIG.default_output_dir),
                )
            ),
            num_epochs=_env_int(
                TRAIN_FULL_FINE_TUNE_CONFIG.env_num_epochs,
                TRAIN_FULL_FINE_TUNE_CONFIG.default_num_epochs,
            ),
            train_batch_size=_env_int(
                TRAIN_FULL_FINE_TUNE_CONFIG.env_train_batch_size,
                TRAIN_FULL_FINE_TUNE_CONFIG.default_train_batch_size,
            ),
            eval_batch_size=_env_int(
                TRAIN_FULL_FINE_TUNE_CONFIG.env_eval_batch_size,
                TRAIN_FULL_FINE_TUNE_CONFIG.default_eval_batch_size,
            ),
            gradient_accumulation_steps=_env_int(
                TRAIN_FULL_FINE_TUNE_CONFIG.env_grad_accum_steps,
                TRAIN_FULL_FINE_TUNE_CONFIG.default_grad_accum_steps,
            ),
            learning_rate=_env_float(
                TRAIN_LORA_CONFIG.env_learning_rate,
                TRAIN_LORA_CONFIG.default_learning_rate,
            ),
            weight_decay=_env_float(
                TRAIN_FULL_FINE_TUNE_CONFIG.env_weight_decay,
                TRAIN_FULL_FINE_TUNE_CONFIG.default_weight_decay,
            ),
            warmup_ratio=_env_float(
                TRAIN_FULL_FINE_TUNE_CONFIG.env_warmup_ratio,
                TRAIN_FULL_FINE_TUNE_CONFIG.default_warmup_ratio,
            ),
            max_grad_norm=_env_float(
                TRAIN_FULL_FINE_TUNE_CONFIG.env_max_grad_norm,
                TRAIN_FULL_FINE_TUNE_CONFIG.default_max_grad_norm,
            ),
            max_input_tokens=_env_optional_int(
                TRAIN_FULL_FINE_TUNE_CONFIG.env_max_input_tokens
            ),
            max_new_tokens=_env_int(
                TRAIN_FULL_FINE_TUNE_CONFIG.env_max_new_tokens,
                TRAIN_FULL_FINE_TUNE_CONFIG.default_max_new_tokens,
            ),
            device=os.getenv(
                TRAIN_FULL_FINE_TUNE_CONFIG.env_device,
                TRAIN_FULL_FINE_TUNE_CONFIG.default_device,
            ),
            dtype=_resolve_dtype(
                os.getenv(
                    TRAIN_FULL_FINE_TUNE_CONFIG.env_dtype,
                    TRAIN_FULL_FINE_TUNE_CONFIG.default_dtype,
                )
            ),
            attn_implementation=attn,
            trust_remote_code=_env_bool(
                TRAIN_FULL_FINE_TUNE_CONFIG.env_trust_remote_code, False
            ),
            cpu_threads=_env_int(
                TRAIN_FULL_FINE_TUNE_CONFIG.env_cpu_threads,
                TRAIN_FULL_FINE_TUNE_CONFIG.default_cpu_threads,
            ),
            log_every_steps=_env_int(
                TRAIN_FULL_FINE_TUNE_CONFIG.env_log_every_steps,
                TRAIN_FULL_FINE_TUNE_CONFIG.default_log_every_steps,
            ),
            save_every_epoch=_env_bool(
                TRAIN_FULL_FINE_TUNE_CONFIG.env_save_every_epoch, True
            ),
            eval_every_epoch=_env_bool(
                TRAIN_FULL_FINE_TUNE_CONFIG.env_eval_every_epoch, True
            ),
            max_train_examples=_env_optional_int(
                TRAIN_FULL_FINE_TUNE_CONFIG.env_max_train_examples
            ),
            max_validation_examples=_env_optional_int(
                TRAIN_FULL_FINE_TUNE_CONFIG.env_max_validation_examples
            ),
            max_test_examples=_env_optional_int(
                TRAIN_FULL_FINE_TUNE_CONFIG.env_max_test_examples
            ),
            num_workers=_env_int(
                TRAIN_FULL_FINE_TUNE_CONFIG.env_num_workers,
                TRAIN_FULL_FINE_TUNE_CONFIG.default_num_workers,
            ),
            gradient_checkpointing=_env_bool(
                TRAIN_LORA_CONFIG.env_gradient_checkpointing, False
            ),
            save_optimizer_state=_env_bool(
                TRAIN_FULL_FINE_TUNE_CONFIG.env_save_optimizer_state,
                TRAIN_FULL_FINE_TUNE_CONFIG.default_save_optimizer_state,
            ),
            strict_parser=_env_bool(
                TRAIN_FULL_FINE_TUNE_CONFIG.env_strict_parser, False
            ),
            seed=_env_int(
                TRAIN_FULL_FINE_TUNE_CONFIG.env_seed,
                TRAIN_FULL_FINE_TUNE_CONFIG.default_seed,
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
                TRAIN_LORA_CONFIG.env_layer_scope, TRAIN_LORA_CONFIG.default_layer_scope
            ),
            lora_rank=_env_int(
                TRAIN_LORA_CONFIG.env_lora_rank, TRAIN_LORA_CONFIG.default_lora_rank
            ),
            lora_alpha=_env_float(
                TRAIN_LORA_CONFIG.env_lora_alpha, TRAIN_LORA_CONFIG.default_lora_alpha
            ),
            lora_dropout=_env_float(
                TRAIN_LORA_CONFIG.env_lora_dropout,
                TRAIN_LORA_CONFIG.default_lora_dropout,
            ),
            lora_bias=os.getenv(
                TRAIN_LORA_CONFIG.env_lora_bias, TRAIN_LORA_CONFIG.default_lora_bias
            ),
            lora_task_type=os.getenv(
                TRAIN_LORA_CONFIG.env_lora_task_type,
                TRAIN_LORA_CONFIG.default_lora_task_type,
            ),
            modules_to_save=_env_tuple(TRAIN_LORA_CONFIG.env_modules_to_save)
            or TRAIN_LORA_CONFIG.default_modules_to_save,
            merge_for_eval=_env_bool(
                TRAIN_LORA_CONFIG.env_merge_for_eval,
                TRAIN_LORA_CONFIG.default_merge_for_eval,
            ),
            notes=_env_optional_str(TRAIN_LAYER_CONFIG.env_notes),
            track_layerwise_updates=_env_bool(
                TRAIN_LAYER_CONFIG.env_track_layerwise_updates, True
            ),
            checkpoint_percents=normalize_checkpoint_percents(
                _env_int_tuple(TRAIN_LAYER_CONFIG.env_checkpoint_percents)
                or TRAIN_LAYER_CONFIG.default_checkpoint_percents
            ),
            distributed_mode=os.getenv(
                TRAIN_FULL_FINE_TUNE_CONFIG.env_distributed_mode,
                TRAIN_FULL_FINE_TUNE_CONFIG.default_distributed_mode,
            ),
            fsdp_cpu_offload=_env_bool(
                TRAIN_FULL_FINE_TUNE_CONFIG.env_fsdp_cpu_offload, False
            ),
        )
        return cls(config=config, environment=EnvironmentConfig.from_env())
