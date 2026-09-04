"""Torch-based LoRA fine-tuning pipeline for PubMedQA QA experiments."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pubmedqa.evaluation import EnvironmentConfig, current_time_iso, safe_name, write_json, write_jsonl
from pubmedqa.full_finetune import (
    DEFAULT_ATTN_IMPLEMENTATION,
    DEFAULT_CHECKPOINT_PERCENTS,
    DEFAULT_CONDITION,
    DEFAULT_CPU_THREADS,
    DEFAULT_DATA_FRACTION,
    DEFAULT_DATA_REGIME,
    DEFAULT_DISTRIBUTED_MODE,
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    DEFAULT_EVAL_BATCH_SIZE,
    DEFAULT_GRAD_ACCUM_STEPS,
    DEFAULT_LAYER_SCOPE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LOG_EVERY_STEPS,
    DEFAULT_MAX_GRAD_NORM,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_METHOD_NAME,
    DEFAULT_MODEL_NAME,
    DEFAULT_NOTES,
    DEFAULT_NUM_EPOCHS,
    DEFAULT_NUM_WORKERS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RUN_TAG,
    DEFAULT_SAVE_OPTIMIZER_STATE,
    DEFAULT_SEED,
    DEFAULT_TRAIN_BATCH_SIZE,
    DEFAULT_WARMUP_RATIO,
    DEFAULT_WEIGHT_DECAY,
    CheckpointRecord,
    FullFineTuneConfig,
    LayerwiseReference,
    LayerwiseUpdateRecord,
    PubMedQAFullFineTuner,
    TrainingSummary,
    _count_directory_size_bytes,
    _count_parameters,
    _cosine_similarity,
    _env_bool,
    _env_float,
    _env_int,
    _env_int_tuple,
    _env_optional_int,
    _env_optional_float,
    _env_optional_str,
    _env_tuple,
    _match_tracked_module,
    _memory_snapshot,
    _normalize_checkpoint_percents,
    _parse_layer_index,
    _resolve_device,
    _resolve_dtype,
)
from pubmedqa.runtime_settings import TRAIN_FULL_FINE_TUNE_CONFIG, TRAIN_LAYER_CONFIG, TRAIN_LORA_CONFIG

ENV_RUN_ID = TRAIN_FULL_FINE_TUNE_CONFIG.env_run_id
ENV_MODEL_NAME = TRAIN_FULL_FINE_TUNE_CONFIG.env_model_name
ENV_CONDITION = TRAIN_FULL_FINE_TUNE_CONFIG.env_condition
ENV_TRAIN_PATH = TRAIN_FULL_FINE_TUNE_CONFIG.env_train_path
ENV_VALIDATION_PATH = TRAIN_FULL_FINE_TUNE_CONFIG.env_validation_path
ENV_TEST_PATH = TRAIN_FULL_FINE_TUNE_CONFIG.env_test_path
ENV_OUTPUT_DIR = TRAIN_FULL_FINE_TUNE_CONFIG.env_output_dir
ENV_NUM_EPOCHS = TRAIN_FULL_FINE_TUNE_CONFIG.env_num_epochs
ENV_TRAIN_BATCH_SIZE = TRAIN_FULL_FINE_TUNE_CONFIG.env_train_batch_size
ENV_EVAL_BATCH_SIZE = TRAIN_FULL_FINE_TUNE_CONFIG.env_eval_batch_size
ENV_GRAD_ACCUM_STEPS = TRAIN_FULL_FINE_TUNE_CONFIG.env_grad_accum_steps
ENV_LEARNING_RATE = TRAIN_FULL_FINE_TUNE_CONFIG.env_learning_rate
ENV_WEIGHT_DECAY = TRAIN_FULL_FINE_TUNE_CONFIG.env_weight_decay
ENV_WARMUP_RATIO = TRAIN_FULL_FINE_TUNE_CONFIG.env_warmup_ratio
ENV_MAX_GRAD_NORM = TRAIN_FULL_FINE_TUNE_CONFIG.env_max_grad_norm
ENV_MAX_INPUT_TOKENS = TRAIN_FULL_FINE_TUNE_CONFIG.env_max_input_tokens
ENV_MAX_NEW_TOKENS = TRAIN_FULL_FINE_TUNE_CONFIG.env_max_new_tokens
ENV_DEVICE = TRAIN_FULL_FINE_TUNE_CONFIG.env_device
ENV_DTYPE = TRAIN_FULL_FINE_TUNE_CONFIG.env_dtype
ENV_ATTN_IMPLEMENTATION = TRAIN_FULL_FINE_TUNE_CONFIG.env_attn_implementation
ENV_TRUST_REMOTE_CODE = TRAIN_FULL_FINE_TUNE_CONFIG.env_trust_remote_code
ENV_CPU_THREADS = TRAIN_FULL_FINE_TUNE_CONFIG.env_cpu_threads
ENV_LOG_EVERY_STEPS = TRAIN_FULL_FINE_TUNE_CONFIG.env_log_every_steps
ENV_SAVE_EVERY_EPOCH = TRAIN_FULL_FINE_TUNE_CONFIG.env_save_every_epoch
ENV_EVAL_EVERY_EPOCH = TRAIN_FULL_FINE_TUNE_CONFIG.env_eval_every_epoch
ENV_MAX_TRAIN_EXAMPLES = TRAIN_FULL_FINE_TUNE_CONFIG.env_max_train_examples
ENV_MAX_VALIDATION_EXAMPLES = TRAIN_FULL_FINE_TUNE_CONFIG.env_max_validation_examples
ENV_MAX_TEST_EXAMPLES = TRAIN_FULL_FINE_TUNE_CONFIG.env_max_test_examples
ENV_NUM_WORKERS = TRAIN_FULL_FINE_TUNE_CONFIG.env_num_workers
ENV_GRADIENT_CHECKPOINTING = TRAIN_FULL_FINE_TUNE_CONFIG.env_gradient_checkpointing
ENV_SAVE_OPTIMIZER_STATE = TRAIN_FULL_FINE_TUNE_CONFIG.env_save_optimizer_state
ENV_STRICT_PARSER = TRAIN_FULL_FINE_TUNE_CONFIG.env_strict_parser
ENV_SEED = TRAIN_FULL_FINE_TUNE_CONFIG.env_seed
ENV_METHOD_NAME = TRAIN_FULL_FINE_TUNE_CONFIG.env_method_name
ENV_RUN_TAG = TRAIN_FULL_FINE_TUNE_CONFIG.env_run_tag
ENV_DATA_REGIME = TRAIN_FULL_FINE_TUNE_CONFIG.env_data_regime
ENV_DATA_FRACTION = TRAIN_FULL_FINE_TUNE_CONFIG.env_data_fraction
ENV_NOTES = TRAIN_LAYER_CONFIG.env_notes
ENV_TRACK_LAYERWISE_UPDATES = TRAIN_LAYER_CONFIG.env_track_layerwise_updates
ENV_CHECKPOINT_PERCENTS = TRAIN_LAYER_CONFIG.env_checkpoint_percents
ENV_DISTRIBUTED_MODE = "PUBMEDQA_DISTRIBUTED_MODE"
ENV_FSDP_CPU_OFFLOAD = "PUBMEDQA_FSDP_CPU_OFFLOAD"

ENV_LORA_TARGET_MODULES = TRAIN_LORA_CONFIG.env_target_modules
ENV_LORA_TARGET_LAYERS = TRAIN_LORA_CONFIG.env_target_layers
ENV_LORA_TARGET_LAYERS_FILE = TRAIN_LORA_CONFIG.env_target_layers_file
ENV_LORA_LAYER_SCOPE = TRAIN_LORA_CONFIG.env_layer_scope
ENV_LORA_RANK = TRAIN_LORA_CONFIG.env_lora_rank
ENV_LORA_ALPHA = TRAIN_LORA_CONFIG.env_lora_alpha
ENV_LORA_DROPOUT = TRAIN_LORA_CONFIG.env_lora_dropout
ENV_LORA_BIAS = TRAIN_LORA_CONFIG.env_lora_bias
ENV_LORA_TASK_TYPE = TRAIN_LORA_CONFIG.env_lora_task_type
ENV_LORA_MODULES_TO_SAVE = TRAIN_LORA_CONFIG.env_modules_to_save
ENV_LORA_MERGE_FOR_EVAL = TRAIN_LORA_CONFIG.env_merge_for_eval

DEFAULT_METHOD_NAME = TRAIN_LORA_CONFIG.default_method_name
DEFAULT_CONDITION = TRAIN_LORA_CONFIG.default_condition
DEFAULT_RUN_TAG = TRAIN_LORA_CONFIG.default_run_tag
DEFAULT_LORA_TARGET_MODULES = TRAIN_LORA_CONFIG.default_target_modules
DEFAULT_LORA_TARGET_LAYERS = TRAIN_LORA_CONFIG.default_target_layers
DEFAULT_LORA_RANK = TRAIN_LORA_CONFIG.default_lora_rank
DEFAULT_LORA_ALPHA = TRAIN_LORA_CONFIG.default_lora_alpha
DEFAULT_LORA_DROPOUT = TRAIN_LORA_CONFIG.default_lora_dropout
DEFAULT_LORA_BIAS = TRAIN_LORA_CONFIG.default_lora_bias
DEFAULT_LORA_TASK_TYPE = TRAIN_LORA_CONFIG.default_lora_task_type
DEFAULT_LORA_MODULES_TO_SAVE = TRAIN_LORA_CONFIG.default_modules_to_save
DEFAULT_LORA_MERGE_FOR_EVAL = TRAIN_LORA_CONFIG.default_merge_for_eval


def _load_target_layers(raw: str | None, file_path: str | None) -> tuple[int, ...]:
    layers: list[int] = []
    if raw:
        layers.extend(int(part.strip()) for part in raw.split(",") if part.strip())
    if file_path:
        payload = json.loads(Path(file_path).read_text(encoding="utf-8"))
        file_layers = payload.get("target_layers", [])
        if not isinstance(file_layers, list):
            raise ValueError("target_layers file must contain a list in 'target_layers'.")
        layers.extend(int(value) for value in file_layers)
    return tuple(sorted(set(layers)))


def _normalize_lora_target_modules(values: Sequence[str]) -> tuple[str, ...]:
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


def _effective_rank_from_singular_values(singular_values: torch.Tensor) -> float:
    if singular_values.numel() == 0:
        return 0.0
    total = float(singular_values.sum().item())
    if total == 0.0:
        return 0.0
    probabilities = singular_values / total
    entropy = float((-(probabilities * probabilities.clamp_min(1e-12).log())).sum().item())
    return float(math.exp(entropy))


def _tensor_list(tensor: torch.Tensor) -> list[float]:
    return [float(value) for value in tensor.detach().cpu().tolist()]


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
    distributed_mode: str = DEFAULT_DISTRIBUTED_MODE
    fsdp_cpu_offload: bool = False


@dataclass(frozen=True)
class LoRAFineTuneCliConfig:
    config: LoRAFineTuneConfig
    environment: EnvironmentConfig

    @classmethod
    def from_env(cls) -> "LoRAFineTuneCliConfig":
        train_path = os.getenv(ENV_TRAIN_PATH)
        validation_path = os.getenv(ENV_VALIDATION_PATH)
        if not train_path or not validation_path:
            raise RuntimeError(
                f"{ENV_TRAIN_PATH} and {ENV_VALIDATION_PATH} are required for LoRA fine-tuning."
            )

        attn = os.getenv(ENV_ATTN_IMPLEMENTATION, DEFAULT_ATTN_IMPLEMENTATION).strip().lower()
        if attn in {"", "none", "auto"}:
            attn = None

        config = LoRAFineTuneConfig(
            run_id=os.getenv(ENV_RUN_ID) or current_time_iso().replace(":", "").replace("-", ""),
            run_tag=os.getenv(ENV_RUN_TAG, DEFAULT_RUN_TAG),
            method_name=os.getenv(ENV_METHOD_NAME, DEFAULT_METHOD_NAME),
            model_name=os.getenv(ENV_MODEL_NAME, DEFAULT_MODEL_NAME),
            condition=os.getenv(ENV_CONDITION, DEFAULT_CONDITION),
            data_regime=os.getenv(ENV_DATA_REGIME, DEFAULT_DATA_REGIME),
            data_fraction=_env_float(ENV_DATA_FRACTION, DEFAULT_DATA_FRACTION),
            train_path=Path(train_path),
            validation_path=Path(validation_path),
            test_path=Path(os.getenv(ENV_TEST_PATH)) if os.getenv(ENV_TEST_PATH) else None,
            output_dir=Path(os.getenv(ENV_OUTPUT_DIR, str(DEFAULT_OUTPUT_DIR))),
            num_epochs=_env_int(ENV_NUM_EPOCHS, DEFAULT_NUM_EPOCHS),
            train_batch_size=_env_int(ENV_TRAIN_BATCH_SIZE, DEFAULT_TRAIN_BATCH_SIZE),
            eval_batch_size=_env_int(ENV_EVAL_BATCH_SIZE, DEFAULT_EVAL_BATCH_SIZE),
            gradient_accumulation_steps=_env_int(ENV_GRAD_ACCUM_STEPS, DEFAULT_GRAD_ACCUM_STEPS),
            learning_rate=_env_float(ENV_LEARNING_RATE, DEFAULT_LEARNING_RATE),
            weight_decay=_env_float(ENV_WEIGHT_DECAY, DEFAULT_WEIGHT_DECAY),
            warmup_ratio=_env_float(ENV_WARMUP_RATIO, DEFAULT_WARMUP_RATIO),
            max_grad_norm=_env_float(ENV_MAX_GRAD_NORM, DEFAULT_MAX_GRAD_NORM),
            max_input_tokens=_env_optional_int(ENV_MAX_INPUT_TOKENS),
            max_new_tokens=_env_int(ENV_MAX_NEW_TOKENS, DEFAULT_MAX_NEW_TOKENS),
            device=os.getenv(ENV_DEVICE, DEFAULT_DEVICE),
            dtype=_resolve_dtype(os.getenv(ENV_DTYPE, DEFAULT_DTYPE)),
            attn_implementation=attn,
            trust_remote_code=_env_bool(ENV_TRUST_REMOTE_CODE, False),
            cpu_threads=_env_int(ENV_CPU_THREADS, DEFAULT_CPU_THREADS),
            log_every_steps=_env_int(ENV_LOG_EVERY_STEPS, DEFAULT_LOG_EVERY_STEPS),
            save_every_epoch=_env_bool(ENV_SAVE_EVERY_EPOCH, True),
            eval_every_epoch=_env_bool(ENV_EVAL_EVERY_EPOCH, True),
            max_train_examples=_env_optional_int(ENV_MAX_TRAIN_EXAMPLES),
            max_validation_examples=_env_optional_int(ENV_MAX_VALIDATION_EXAMPLES),
            max_test_examples=_env_optional_int(ENV_MAX_TEST_EXAMPLES),
            num_workers=_env_int(ENV_NUM_WORKERS, DEFAULT_NUM_WORKERS),
            gradient_checkpointing=_env_bool(ENV_GRADIENT_CHECKPOINTING, False),
            save_optimizer_state=_env_bool(ENV_SAVE_OPTIMIZER_STATE, DEFAULT_SAVE_OPTIMIZER_STATE),
            strict_parser=_env_bool(ENV_STRICT_PARSER, False),
            seed=_env_int(ENV_SEED, DEFAULT_SEED),
            target_modules=_normalize_lora_target_modules(
                _env_tuple(ENV_LORA_TARGET_MODULES) or DEFAULT_LORA_TARGET_MODULES
            ),
            target_layers=_load_target_layers(
                os.getenv(ENV_LORA_TARGET_LAYERS),
                os.getenv(ENV_LORA_TARGET_LAYERS_FILE),
            ) or DEFAULT_LORA_TARGET_LAYERS,
            layer_scope=os.getenv(ENV_LORA_LAYER_SCOPE, DEFAULT_LAYER_SCOPE),
            lora_rank=_env_int(ENV_LORA_RANK, DEFAULT_LORA_RANK),
            lora_alpha=_env_float(ENV_LORA_ALPHA, DEFAULT_LORA_ALPHA),
            lora_dropout=_env_float(ENV_LORA_DROPOUT, DEFAULT_LORA_DROPOUT),
            lora_bias=os.getenv(ENV_LORA_BIAS, DEFAULT_LORA_BIAS),
            lora_task_type=os.getenv(ENV_LORA_TASK_TYPE, DEFAULT_LORA_TASK_TYPE),
            modules_to_save=_env_tuple(ENV_LORA_MODULES_TO_SAVE) or DEFAULT_LORA_MODULES_TO_SAVE,
            merge_for_eval=_env_bool(ENV_LORA_MERGE_FOR_EVAL, DEFAULT_LORA_MERGE_FOR_EVAL),
            notes=_env_optional_str(ENV_NOTES),
            track_layerwise_updates=_env_bool(ENV_TRACK_LAYERWISE_UPDATES, True),
            checkpoint_percents=_normalize_checkpoint_percents(
                _env_int_tuple(ENV_CHECKPOINT_PERCENTS) or DEFAULT_CHECKPOINT_PERCENTS
            ),
            distributed_mode=DEFAULT_DISTRIBUTED_MODE,
            fsdp_cpu_offload=False,
        )
        return cls(config=config, environment=EnvironmentConfig.from_env())


@dataclass(frozen=True)
class LoRAAdapterMetric:
    checkpoint_id: str
    checkpoint_kind: str
    checkpoint_percent: float
    epoch: int
    global_step: int
    layer_index: int | None
    module_name: str
    component_name: str
    parameter_name: str
    base_weight_norm: float
    a_norm: float
    b_norm: float
    scaling: float
    delta_w_norm: float
    delta_w_relative_norm: float
    incremental_delta_w_norm: float
    incremental_delta_w_relative_norm: float
    incremental_delta_w_cosine_similarity: float | None
    effective_rank: float
    singular_values: list[float]
    cumulative_update_share: float
    incremental_update_share: float


class PubMedQALoRAFineTuner(PubMedQAFullFineTuner):
    def __init__(self, config: LoRAFineTuneConfig, environment: EnvironmentConfig) -> None:
        super().__init__(
            FullFineTuneConfig(
                run_id=config.run_id,
                run_tag=config.run_tag,
                method_name=config.method_name,
                model_name=config.model_name,
                condition=config.condition,
                data_regime=config.data_regime,
                data_fraction=config.data_fraction,
                train_path=config.train_path,
                validation_path=config.validation_path,
                test_path=config.test_path,
                output_dir=config.output_dir,
                num_epochs=config.num_epochs,
                train_batch_size=config.train_batch_size,
                eval_batch_size=config.eval_batch_size,
                gradient_accumulation_steps=config.gradient_accumulation_steps,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
                warmup_ratio=config.warmup_ratio,
                max_grad_norm=config.max_grad_norm,
                max_input_tokens=config.max_input_tokens,
                max_new_tokens=config.max_new_tokens,
                device=config.device,
                dtype=config.dtype,
                attn_implementation=config.attn_implementation,
                trust_remote_code=config.trust_remote_code,
                cpu_threads=config.cpu_threads,
                log_every_steps=config.log_every_steps,
                save_every_epoch=config.save_every_epoch,
                eval_every_epoch=config.eval_every_epoch,
                max_train_examples=config.max_train_examples,
                max_validation_examples=config.max_validation_examples,
                max_test_examples=config.max_test_examples,
                num_workers=config.num_workers,
                gradient_checkpointing=config.gradient_checkpointing,
                save_optimizer_state=config.save_optimizer_state,
                strict_parser=config.strict_parser,
                seed=config.seed,
                target_modules=config.target_modules,
                target_layers=config.target_layers,
                layer_scope=config.layer_scope,
                lora_rank=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                notes=config.notes,
                track_layerwise_updates=config.track_layerwise_updates,
                checkpoint_percents=config.checkpoint_percents,
                distributed_mode=config.distributed_mode,
                fsdp_cpu_offload=config.fsdp_cpu_offload,
            ),
            environment,
        )
        self.lora_config = config
        self.adapter_metrics_history: list[dict[str, Any]] = []
        self.module_share_history: list[dict[str, Any]] = []
        self.adapter_checkpoint_sizes: dict[str, int] = {}
        self.target_modules_normalized = config.target_modules
        self.device = _resolve_device(config.device)

    def _import_peft(self):
        try:
            from peft import LoraConfig, PeftConfig, PeftModel, TaskType, get_peft_model
        except ImportError as exc:
            raise RuntimeError(
                "peft is required for LoRA fine-tuning. Activate the configured conda environment first."
            ) from exc
        return LoraConfig, PeftConfig, PeftModel, TaskType, get_peft_model

    def _build_peft_task_type(self, task_type_enum: Any) -> Any:
        if hasattr(task_type_enum, self.lora_config.lora_task_type):
            return getattr(task_type_enum, self.lora_config.lora_task_type)
        return self.lora_config.lora_task_type

    def _load_base_model_and_tokenizer(self, model_name_or_path: str) -> tuple[Any, torch.nn.Module]:
        common_kwargs = {
            "token": self.environment.hf_token,
            "trust_remote_code": self.lora_config.trust_remote_code,
        }
        model_kwargs = {**common_kwargs, "torch_dtype": self.lora_config.dtype}
        if self.lora_config.attn_implementation is not None:
            model_kwargs["attn_implementation"] = self.lora_config.attn_implementation

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **common_kwargs)
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise RuntimeError(f"Tokenizer for {model_name_or_path} has neither pad_token nor eos_token.")
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
        model.to(self.device)
        return tokenizer, model

    def load_model_and_tokenizer(self, model_name_or_path: str) -> tuple[Any, torch.nn.Module]:
        _, PeftConfig, PeftModel, TaskType, get_peft_model = self._import_peft()
        path = Path(model_name_or_path)
        adapter_config_path = path / "adapter_config.json"

        if path.is_dir() and adapter_config_path.is_file():
            peft_config = PeftConfig.from_pretrained(str(path))
            tokenizer_source = str(path) if (path / "tokenizer_config.json").is_file() else peft_config.base_model_name_or_path
            tokenizer, base_model = self._load_base_model_and_tokenizer(peft_config.base_model_name_or_path)
            if tokenizer_source != peft_config.base_model_name_or_path:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, token=self.environment.hf_token)
                if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                tokenizer.padding_side = "right"
            model = PeftModel.from_pretrained(base_model, str(path), is_trainable=False)
            if self.lora_config.merge_for_eval:
                model = model.merge_and_unload()
                model.to(self.device)
            else:
                model.to(self.device)
            return tokenizer, model

        tokenizer, base_model = self._load_base_model_and_tokenizer(model_name_or_path)
        if self.lora_config.gradient_checkpointing:
            base_model.gradient_checkpointing_enable()
            if hasattr(base_model.config, "use_cache"):
                base_model.config.use_cache = False

        peft_task_type = self._build_peft_task_type(TaskType)
        lora_arguments: dict[str, Any] = {
            "r": self.lora_config.lora_rank,
            "lora_alpha": self.lora_config.lora_alpha,
            "lora_dropout": self.lora_config.lora_dropout,
            "bias": self.lora_config.lora_bias,
            "task_type": peft_task_type,
            "target_modules": list(self.target_modules_normalized),
        }
        if self.lora_config.target_layers:
            lora_arguments["layers_to_transform"] = list(self.lora_config.target_layers)
            lora_arguments["layers_pattern"] = ["layers"]
        if self.lora_config.modules_to_save:
            lora_arguments["modules_to_save"] = list(self.lora_config.modules_to_save)

        lora_config = LoraConfig(**lora_arguments)
        model = get_peft_model(base_model, lora_config)
        if self.lora_config.gradient_checkpointing and hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        model.to(self.device)
        return tokenizer, model

    def save_checkpoint(
        self,
        *,
        model: torch.nn.Module,
        tokenizer: Any,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        checkpoint_kind: str,
        checkpoint_percent: float,
        epoch: int,
        step_in_epoch: int,
        global_step: int,
        elapsed_seconds: float,
        validation_metrics: Any,
        save_model_files: bool,
    ) -> Path:
        checkpoint_dir = self.checkpoints_dir / (
            f"{checkpoint_kind}_pct_{int(round(checkpoint_percent)):03d}_"
            f"epoch_{epoch:03d}_step_{global_step:06d}"
        )
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if save_model_files:
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
        if save_model_files and self.lora_config.save_optimizer_state:
            torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
            torch.save(scheduler.state_dict(), checkpoint_dir / "scheduler.pt")

        checkpoint_id = checkpoint_dir.name
        write_json(
            checkpoint_dir / "training_state.json",
            {
                "checkpoint_id": checkpoint_id,
                "checkpoint_kind": checkpoint_kind,
                "checkpoint_percent": checkpoint_percent,
                "epoch": epoch,
                "step_in_epoch": step_in_epoch,
                "global_step": global_step,
                "elapsed_seconds": elapsed_seconds,
                "validation_metrics": asdict(validation_metrics),
                "title": self.title,
                "run_id": self.lora_config.run_id,
                "run_tag": self.lora_config.run_tag,
                "model_name": self.lora_config.model_name,
                "condition": self.lora_config.condition,
                "lora": {
                    "target_modules": list(self.lora_config.target_modules),
                    "target_layers": list(self.lora_config.target_layers),
                    "layer_scope": self.lora_config.layer_scope,
                    "rank": self.lora_config.lora_rank,
                    "alpha": self.lora_config.lora_alpha,
                    "dropout": self.lora_config.lora_dropout,
                    "bias": self.lora_config.lora_bias,
                    "task_type": self.lora_config.lora_task_type,
                    "modules_to_save": list(self.lora_config.modules_to_save),
                    "merge_for_eval": self.lora_config.merge_for_eval,
                },
                "saved_at": current_time_iso(),
            },
        )
        return checkpoint_dir

    def _named_lora_modules(self, model: torch.nn.Module) -> list[tuple[str, Any]]:
        modules: list[tuple[str, Any]] = []
        for module_name, module in model.named_modules():
            if hasattr(module, "lora_A") and hasattr(module, "lora_B") and hasattr(module, "base_layer"):
                modules.append((module_name, module))
        return modules

    def _compute_lora_delta_tensor(self, module: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        adapter_names = list(module.lora_A.keys())
        if not adapter_names:
            raise RuntimeError("LoRA module has no registered adapter.")
        adapter_name = adapter_names[0]
        a_weight = module.lora_A[adapter_name].weight.detach().cpu().float()
        b_weight = module.lora_B[adapter_name].weight.detach().cpu().float()
        scaling = float(module.scaling[adapter_name])
        delta = scaling * (b_weight @ a_weight)
        return a_weight, b_weight, delta, scaling

    def capture_layerwise_references(self, model: torch.nn.Module) -> list[LayerwiseReference]:
        references: list[LayerwiseReference] = []
        tracked_parameters: list[dict[str, Any]] = []
        for module_name, module in self._named_lora_modules(model):
            parameter_name = f"{module_name}.weight"
            match = _match_tracked_module(parameter_name)
            if match is None:
                continue
            module_alias, component_name = match
            base_weight = module.base_layer.weight.detach().cpu().float()
            references.append(
                LayerwiseReference(
                    parameter_name=parameter_name,
                    layer_index=_parse_layer_index(parameter_name),
                    module_name=module_alias,
                    component_name=component_name,
                    shape=tuple(base_weight.shape),
                    num_parameters=base_weight.numel(),
                    base_weight_norm=float(base_weight.norm().item()),
                    base_tensor=torch.zeros_like(base_weight, dtype=torch.float16),
                )
            )
            tracked_parameters.append(
                {
                    "parameter_name": parameter_name,
                    "layer_index": _parse_layer_index(parameter_name),
                    "module_name": module_alias,
                    "component_name": component_name,
                    "shape": list(base_weight.shape),
                    "num_parameters": base_weight.numel(),
                    "base_weight_norm": float(base_weight.norm().item()),
                    "tracked_as": "lora_delta",
                }
            )
        write_json(
            self.layerwise_dir / "base_reference_summary.json",
            {
                "run_id": self.lora_config.run_id,
                "run_tag": self.lora_config.run_tag,
                "model_name": self.lora_config.model_name,
                "tracked_parameters": tracked_parameters,
            },
        )
        return references

    def write_layerwise_update_artifacts(
        self,
        *,
        model: torch.nn.Module,
        checkpoint_kind: str,
        checkpoint_percent: float,
        epoch: int,
        global_step: int,
        checkpoint_dir: Path,
        references: Sequence[LayerwiseReference],
        previous_snapshots: dict[str, torch.Tensor],
        previous_incremental_updates: dict[str, torch.Tensor],
    ) -> None:
        if not self.lora_config.track_layerwise_updates or not references:
            return

        reference_map = {reference.parameter_name: reference for reference in references}
        module_map = {name: module for name, module in self._named_lora_modules(model)}
        layerwise_records: list[LayerwiseUpdateRecord] = []
        adapter_records: list[LoRAAdapterMetric] = []
        current_snapshots: dict[str, torch.Tensor] = {}

        for module_name, module in module_map.items():
            parameter_name = f"{module_name}.weight"
            reference = reference_map.get(parameter_name)
            if reference is None:
                continue

            a_weight, b_weight, delta_tensor, scaling = self._compute_lora_delta_tensor(module)
            previous_tensor = previous_snapshots.get(parameter_name)
            if previous_tensor is None:
                previous_tensor = torch.zeros_like(delta_tensor)
            previous_tensor = previous_tensor.float()
            previous_incremental_tensor = previous_incremental_updates.get(parameter_name)

            incremental_delta = delta_tensor - previous_tensor
            delta_norm = float(delta_tensor.norm().item())
            incremental_delta_norm = float(incremental_delta.norm().item())
            delta_relative_norm = 0.0 if reference.base_weight_norm == 0.0 else delta_norm / reference.base_weight_norm
            incremental_relative_norm = (
                0.0 if reference.base_weight_norm == 0.0 else incremental_delta_norm / reference.base_weight_norm
            )
            singular_values = torch.linalg.svdvals(delta_tensor)
            effective_rank = _effective_rank_from_singular_values(singular_values)
            cosine_similarity = None
            if previous_incremental_tensor is not None:
                cosine_similarity = _cosine_similarity(incremental_delta, previous_incremental_tensor.float())

            layerwise_records.append(
                LayerwiseUpdateRecord(
                    checkpoint_kind=checkpoint_kind,
                    checkpoint_percent=checkpoint_percent,
                    epoch=epoch,
                    global_step=global_step,
                    layer_index=reference.layer_index,
                    module_name=reference.module_name,
                    component_name=reference.component_name,
                    parameter_name=parameter_name,
                    shape=reference.shape,
                    num_parameters=reference.num_parameters,
                    base_weight_norm=reference.base_weight_norm,
                    current_weight_norm=delta_norm,
                    update_norm=delta_norm,
                    relative_update_norm=delta_relative_norm,
                    incremental_update_norm=incremental_delta_norm,
                    relative_incremental_update_norm=incremental_relative_norm,
                    incremental_update_cosine_similarity=cosine_similarity,
                    cumulative_update_share=0.0,
                    incremental_update_share=0.0,
                )
            )
            adapter_records.append(
                LoRAAdapterMetric(
                    checkpoint_id=checkpoint_dir.name,
                    checkpoint_kind=checkpoint_kind,
                    checkpoint_percent=checkpoint_percent,
                    epoch=epoch,
                    global_step=global_step,
                    layer_index=reference.layer_index,
                    module_name=reference.module_name,
                    component_name=reference.component_name,
                    parameter_name=parameter_name,
                    base_weight_norm=reference.base_weight_norm,
                    a_norm=float(a_weight.norm().item()),
                    b_norm=float(b_weight.norm().item()),
                    scaling=scaling,
                    delta_w_norm=delta_norm,
                    delta_w_relative_norm=delta_relative_norm,
                    incremental_delta_w_norm=incremental_delta_norm,
                    incremental_delta_w_relative_norm=incremental_relative_norm,
                    incremental_delta_w_cosine_similarity=cosine_similarity,
                    effective_rank=effective_rank,
                    singular_values=_tensor_list(singular_values),
                    cumulative_update_share=0.0,
                    incremental_update_share=0.0,
                )
            )
            current_snapshots[parameter_name] = delta_tensor.to(torch.float16)
            previous_incremental_updates[parameter_name] = incremental_delta.to(torch.float16)

        total_cumulative = sum(record.update_norm for record in layerwise_records)
        total_incremental = sum(record.incremental_update_norm for record in layerwise_records)
        normalized_layerwise: list[LayerwiseUpdateRecord] = []
        normalized_adapter: list[LoRAAdapterMetric] = []
        for layer_record, adapter_record in zip(layerwise_records, adapter_records):
            cumulative_share = 0.0 if total_cumulative == 0.0 else layer_record.update_norm / total_cumulative
            incremental_share = (
                0.0 if total_incremental == 0.0 else layer_record.incremental_update_norm / total_incremental
            )
            normalized_layerwise.append(
                LayerwiseUpdateRecord(
                    checkpoint_kind=layer_record.checkpoint_kind,
                    checkpoint_percent=layer_record.checkpoint_percent,
                    epoch=layer_record.epoch,
                    global_step=layer_record.global_step,
                    layer_index=layer_record.layer_index,
                    module_name=layer_record.module_name,
                    component_name=layer_record.component_name,
                    parameter_name=layer_record.parameter_name,
                    shape=layer_record.shape,
                    num_parameters=layer_record.num_parameters,
                    base_weight_norm=layer_record.base_weight_norm,
                    current_weight_norm=layer_record.current_weight_norm,
                    update_norm=layer_record.update_norm,
                    relative_update_norm=layer_record.relative_update_norm,
                    incremental_update_norm=layer_record.incremental_update_norm,
                    relative_incremental_update_norm=layer_record.relative_incremental_update_norm,
                    incremental_update_cosine_similarity=layer_record.incremental_update_cosine_similarity,
                    cumulative_update_share=cumulative_share,
                    incremental_update_share=incremental_share,
                )
            )
            normalized_adapter.append(
                LoRAAdapterMetric(
                    checkpoint_id=adapter_record.checkpoint_id,
                    checkpoint_kind=adapter_record.checkpoint_kind,
                    checkpoint_percent=adapter_record.checkpoint_percent,
                    epoch=adapter_record.epoch,
                    global_step=adapter_record.global_step,
                    layer_index=adapter_record.layer_index,
                    module_name=adapter_record.module_name,
                    component_name=adapter_record.component_name,
                    parameter_name=adapter_record.parameter_name,
                    base_weight_norm=adapter_record.base_weight_norm,
                    a_norm=adapter_record.a_norm,
                    b_norm=adapter_record.b_norm,
                    scaling=adapter_record.scaling,
                    delta_w_norm=adapter_record.delta_w_norm,
                    delta_w_relative_norm=adapter_record.delta_w_relative_norm,
                    incremental_delta_w_norm=adapter_record.incremental_delta_w_norm,
                    incremental_delta_w_relative_norm=adapter_record.incremental_delta_w_relative_norm,
                    incremental_delta_w_cosine_similarity=adapter_record.incremental_delta_w_cosine_similarity,
                    effective_rank=adapter_record.effective_rank,
                    singular_values=adapter_record.singular_values,
                    cumulative_update_share=cumulative_share,
                    incremental_update_share=incremental_share,
                )
            )

        previous_snapshots.update(current_snapshots)
        file_stem = checkpoint_dir.name
        write_jsonl(self.layerwise_dir / f"{file_stem}.jsonl", (asdict(row) for row in normalized_layerwise))
        write_jsonl(checkpoint_dir / "layerwise_updates.jsonl", (asdict(row) for row in normalized_layerwise))
        write_jsonl(checkpoint_dir / "lora_adapter_metrics.jsonl", (asdict(row) for row in normalized_adapter))

        component_summary = self._summarize_layerwise_by_component(normalized_layerwise)
        layer_summary = self._summarize_layerwise_by_layer(normalized_layerwise)
        module_share_summary = self._summarize_module_share(normalized_adapter)
        avg_cosine = self._average_cosine_similarity(normalized_layerwise)
        summary_payload = {
            "checkpoint_id": checkpoint_dir.name,
            "checkpoint_kind": checkpoint_kind,
            "checkpoint_percent": checkpoint_percent,
            "epoch": epoch,
            "global_step": global_step,
            "num_records": len(normalized_adapter),
            "total_cumulative_update_norm": total_cumulative,
            "total_incremental_update_norm": total_incremental,
            "avg_incremental_update_cosine_similarity": avg_cosine,
            "by_component": component_summary,
            "by_layer": layer_summary,
            "by_module": module_share_summary,
        }
        write_json(self.layerwise_dir / f"{file_stem}_summary.json", summary_payload)

        adapter_summary = {
            "checkpoint_id": checkpoint_dir.name,
            "checkpoint_kind": checkpoint_kind,
            "checkpoint_percent": checkpoint_percent,
            "epoch": epoch,
            "global_step": global_step,
            "adapter_metrics_jsonl": str(checkpoint_dir / "lora_adapter_metrics.jsonl"),
            "module_update_share": module_share_summary,
            "avg_effective_rank": (
                sum(record.effective_rank for record in normalized_adapter) / len(normalized_adapter)
                if normalized_adapter
                else 0.0
            ),
        }
        write_json(checkpoint_dir / "lora_adapter_summary.json", adapter_summary)

        self.adapter_metrics_history.append(
            {
                "checkpoint_id": checkpoint_dir.name,
                "checkpoint_kind": checkpoint_kind,
                "checkpoint_percent": checkpoint_percent,
                "epoch": epoch,
                "global_step": global_step,
                "records": [asdict(row) for row in normalized_adapter],
            }
        )
        self.module_share_history.append(
            {
                "checkpoint_id": checkpoint_dir.name,
                "checkpoint_kind": checkpoint_kind,
                "checkpoint_percent": checkpoint_percent,
                "epoch": epoch,
                "global_step": global_step,
                "by_module": module_share_summary,
            }
        )

    @staticmethod
    def _summarize_module_share(records: Sequence[LoRAAdapterMetric]) -> dict[str, dict[str, float | int]]:
        summary: dict[str, dict[str, float | int]] = {}
        for record in records:
            key = record.module_name
            bucket = summary.setdefault(
                key,
                {
                    "count": 0,
                    "sum_delta_w_norm": 0.0,
                    "sum_delta_w_relative_norm": 0.0,
                    "sum_incremental_delta_w_norm": 0.0,
                    "sum_incremental_delta_w_relative_norm": 0.0,
                    "sum_cumulative_update_share": 0.0,
                    "sum_incremental_update_share": 0.0,
                },
            )
            bucket["count"] += 1
            bucket["sum_delta_w_norm"] += record.delta_w_norm
            bucket["sum_delta_w_relative_norm"] += record.delta_w_relative_norm
            bucket["sum_incremental_delta_w_norm"] += record.incremental_delta_w_norm
            bucket["sum_incremental_delta_w_relative_norm"] += record.incremental_delta_w_relative_norm
            bucket["sum_cumulative_update_share"] += record.cumulative_update_share
            bucket["sum_incremental_update_share"] += record.incremental_update_share
        for bucket in summary.values():
            count = int(bucket["count"])
            bucket["avg_delta_w_relative_norm"] = (
                bucket["sum_delta_w_relative_norm"] / count if count > 0 else 0.0
            )
            bucket["avg_incremental_delta_w_relative_norm"] = (
                bucket["sum_incremental_delta_w_relative_norm"] / count if count > 0 else 0.0
            )
        return summary

    def run(self) -> TrainingSummary:
        summary = super().run()
        if not self.is_main_process:
            return summary
        output_summary = {
            **asdict(summary),
            "adapter_params": summary.trainable_params,
            "adapter_checkpoint_size_bytes": self.adapter_checkpoint_sizes.get(summary.best_checkpoint_dir, 0),
            "merged_checkpoint_size_bytes": None,
            "lora_bias": self.lora_config.lora_bias,
            "lora_task_type": self.lora_config.lora_task_type,
            "modules_to_save": list(self.lora_config.modules_to_save),
            "merge_for_eval": self.lora_config.merge_for_eval,
        }
        write_json(self.output_root / "summary.json", output_summary)
        run_metadata_path = self.output_root / "run_metadata.json"
        if run_metadata_path.is_file():
            metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
        else:
            metadata = {}
        metadata.update(
            {
                "lora_bias": self.lora_config.lora_bias,
                "lora_task_type": self.lora_config.lora_task_type,
                "modules_to_save": list(self.lora_config.modules_to_save),
                "merge_for_eval": self.lora_config.merge_for_eval,
            }
        )
        write_json(run_metadata_path, metadata)
        return summary

    def record_checkpoint(
        self,
        *,
        model: torch.nn.Module,
        tokenizer: Any,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        checkpoint_kind: str,
        checkpoint_percent: float,
        epoch: int,
        step_in_epoch: int,
        global_step: int,
        elapsed_seconds: float,
        train_loss: float | None,
        gradient_norm: float | None,
        validation_metrics: Any,
        references: Sequence[LayerwiseReference],
        previous_snapshots: dict[str, torch.Tensor],
        previous_incremental_updates: dict[str, torch.Tensor],
        save_model_files: bool,
    ) -> CheckpointRecord:
        checkpoint = super().record_checkpoint(
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            checkpoint_kind=checkpoint_kind,
            checkpoint_percent=checkpoint_percent,
            epoch=epoch,
            step_in_epoch=step_in_epoch,
            global_step=global_step,
            elapsed_seconds=elapsed_seconds,
            train_loss=train_loss,
            gradient_norm=gradient_norm,
            validation_metrics=validation_metrics,
            references=references,
            previous_snapshots=previous_snapshots,
            previous_incremental_updates=previous_incremental_updates,
            save_model_files=save_model_files,
        )
        self.adapter_checkpoint_sizes[checkpoint.checkpoint_dir] = checkpoint.checkpoint_size_bytes
        return checkpoint

    def write_analysis_groups(
        self,
        summary: TrainingSummary,
        checkpoint_records: Sequence[CheckpointRecord],
        step_logs: Sequence[Any],
    ) -> None:
        super().write_analysis_groups(summary, checkpoint_records, step_logs)

        adapter_records: list[dict[str, Any]] = []
        singular_timeline: list[dict[str, Any]] = []
        effective_rank_timeline: list[dict[str, Any]] = []
        val_update_alignment: list[dict[str, Any]] = []
        checkpoint_to_val = {
            record.global_step: {
                "checkpoint_id": Path(record.checkpoint_dir).name,
                "checkpoint_kind": record.checkpoint_kind,
                "checkpoint_percent": record.checkpoint_percent,
                "validation_accuracy": record.validation_accuracy,
                "validation_macro_f1": record.validation_macro_f1,
                "validation_loss": record.validation_loss,
            }
            for record in checkpoint_records
        }

        for entry in self.adapter_metrics_history:
            adapter_records.extend(entry["records"])
            delta_total = sum(record["delta_w_norm"] for record in entry["records"])
            val_row = checkpoint_to_val.get(entry["global_step"], {})
            val_update_alignment.append(
                {
                    "checkpoint_id": entry["checkpoint_id"],
                    "checkpoint_kind": entry["checkpoint_kind"],
                    "checkpoint_percent": entry["checkpoint_percent"],
                    "epoch": entry["epoch"],
                    "global_step": entry["global_step"],
                    "delta_w_norm_total": delta_total,
                    "validation_accuracy": val_row.get("validation_accuracy"),
                    "validation_macro_f1": val_row.get("validation_macro_f1"),
                    "validation_loss": val_row.get("validation_loss"),
                }
            )
            singular_timeline.append(
                {
                    "checkpoint_id": entry["checkpoint_id"],
                    "checkpoint_kind": entry["checkpoint_kind"],
                    "checkpoint_percent": entry["checkpoint_percent"],
                    "epoch": entry["epoch"],
                    "global_step": entry["global_step"],
                    "records": [
                        {
                            "parameter_name": record["parameter_name"],
                            "layer_index": record["layer_index"],
                            "module_name": record["module_name"],
                            "component_name": record["component_name"],
                            "singular_values": record["singular_values"],
                        }
                        for record in entry["records"]
                    ],
                }
            )
            effective_rank_timeline.append(
                {
                    "checkpoint_id": entry["checkpoint_id"],
                    "checkpoint_kind": entry["checkpoint_kind"],
                    "checkpoint_percent": entry["checkpoint_percent"],
                    "epoch": entry["epoch"],
                    "global_step": entry["global_step"],
                    "records": [
                        {
                            "parameter_name": record["parameter_name"],
                            "layer_index": record["layer_index"],
                            "module_name": record["module_name"],
                            "component_name": record["component_name"],
                            "effective_rank": record["effective_rank"],
                        }
                        for record in entry["records"]
                    ],
                }
            )

        write_json(
            self.output_root / "analysis" / "lora_configuration.json",
            {
                "run_tag": summary.run_tag,
                "method_name": summary.method_name,
                "target_modules": list(self.lora_config.target_modules),
                "target_layers": list(self.lora_config.target_layers),
                "layer_scope": self.lora_config.layer_scope,
                "lora_rank": self.lora_config.lora_rank,
                "lora_alpha": self.lora_config.lora_alpha,
                "lora_dropout": self.lora_config.lora_dropout,
                "lora_bias": self.lora_config.lora_bias,
                "lora_task_type": self.lora_config.lora_task_type,
                "modules_to_save": list(self.lora_config.modules_to_save),
                "merge_for_eval": self.lora_config.merge_for_eval,
                "data_regime": summary.data_regime,
                "data_fraction": summary.data_fraction,
            },
        )
        write_json(
            self.output_root / "analysis" / "lora_dynamics.json",
            {
                "run_tag": summary.run_tag,
                "method_name": summary.method_name,
                "checkpoint_timeline": self.adapter_metrics_history,
            },
        )
        write_json(
            self.output_root / "analysis" / "lora_singular_values.json",
            {
                "run_tag": summary.run_tag,
                "method_name": summary.method_name,
                "checkpoint_timeline": singular_timeline,
            },
        )
        write_json(
            self.output_root / "analysis" / "lora_effective_rank.json",
            {
                "run_tag": summary.run_tag,
                "method_name": summary.method_name,
                "checkpoint_timeline": effective_rank_timeline,
            },
        )
        write_json(
            self.output_root / "analysis" / "lora_adapter_norms.json",
            {
                "run_tag": summary.run_tag,
                "method_name": summary.method_name,
                "checkpoint_timeline": [
                    {
                        "checkpoint_id": entry["checkpoint_id"],
                        "checkpoint_kind": entry["checkpoint_kind"],
                        "checkpoint_percent": entry["checkpoint_percent"],
                        "epoch": entry["epoch"],
                        "global_step": entry["global_step"],
                        "records": [
                            {
                                "parameter_name": record["parameter_name"],
                                "a_norm": record["a_norm"],
                                "b_norm": record["b_norm"],
                                "scaling": record["scaling"],
                                "delta_w_norm": record["delta_w_norm"],
                                "delta_w_relative_norm": record["delta_w_relative_norm"],
                            }
                            for record in entry["records"]
                        ],
                    }
                    for entry in self.adapter_metrics_history
                ],
            },
        )
        write_json(
            self.output_root / "analysis" / "lora_update_direction.json",
            {
                "run_tag": summary.run_tag,
                "method_name": summary.method_name,
                "checkpoint_timeline": [
                    {
                        "checkpoint_id": entry["checkpoint_id"],
                        "checkpoint_kind": entry["checkpoint_kind"],
                        "checkpoint_percent": entry["checkpoint_percent"],
                        "epoch": entry["epoch"],
                        "global_step": entry["global_step"],
                        "records": [
                            {
                                "parameter_name": record["parameter_name"],
                                "module_name": record["module_name"],
                                "layer_index": record["layer_index"],
                                "incremental_delta_w_cosine_similarity": record[
                                    "incremental_delta_w_cosine_similarity"
                                ],
                            }
                            for record in entry["records"]
                        ],
                    }
                    for entry in self.adapter_metrics_history
                ],
            },
        )
        write_json(
            self.output_root / "analysis" / "module_update_share.json",
            {
                "run_tag": summary.run_tag,
                "method_name": summary.method_name,
                "checkpoint_timeline": self.module_share_history,
            },
        )
        write_json(
            self.output_root / "analysis" / "val_update_alignment.json",
            {
                "run_tag": summary.run_tag,
                "method_name": summary.method_name,
                "checkpoint_timeline": val_update_alignment,
            },
        )
        write_jsonl(
            self.output_root / "analysis" / "lora_adapter_metrics.jsonl",
            (row for row in adapter_records),
        )


def main() -> None:
    cli_config = LoRAFineTuneCliConfig.from_env()
    trainer = PubMedQALoRAFineTuner(cli_config.config, cli_config.environment)
    try:
        summary = trainer.run()
    finally:
        trainer.close()
    print(summary.title)
    print(f"Best checkpoint: {summary.best_checkpoint_dir}")
    print(f"Best validation ACC: {summary.best_validation_accuracy:.4f}")
    print(f"Best validation Macro F1: {summary.best_validation_macro_f1:.4f}")
    if summary.test_accuracy is not None:
        print(f"Test ACC: {summary.test_accuracy:.4f}")
        print(f"Test Macro F1: {summary.test_macro_f1:.4f}")


if __name__ == "__main__":
    main()
