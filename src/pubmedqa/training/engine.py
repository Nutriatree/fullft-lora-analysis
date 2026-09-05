"""Shared fine-tuning engine with a Full FT strategy implementation."""

from __future__ import annotations

import gc
import math
import os
import shutil
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence, TypeVar, cast

import torch
import torch.distributed as dist
from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from pubmedqa.answer_parser import parse_pubmedqa_answer
from pubmedqa.config import EnvironmentConfig, TRAIN_FULL_FINE_TUNE_CONFIG, TRAIN_LAYER_CONFIG
from pubmedqa.config.env import (
    env_bool as _env_bool,
    env_float as _env_float,
    env_int as _env_int,
    env_int_tuple as _env_int_tuple,
    env_optional_float as _env_optional_float,
    env_optional_int as _env_optional_int,
    env_optional_str as _env_optional_str,
    env_tuple as _env_tuple,
)
from pubmedqa.domain.prompts import PubMedQAExample, build_tokenizer_prompt
from pubmedqa.inference.runner import load_local_jsonl
from pubmedqa.runtime.io import (
    count_directory_size_bytes as _count_directory_size_bytes,
    current_time_iso,
    safe_name,
    write_json,
    write_jsonl,
)
from pubmedqa.runtime.torch_runtime import (
    count_parameters as _count_parameters,
    memory_snapshot as _memory_snapshot,
    resolve_device as _resolve_device,
    resolve_dtype as _resolve_dtype,
    supports_cuda_amp as _supports_cuda_amp,
)
from pubmedqa.training.analysis import (
    average_incremental_cosine_similarity,
    cosine_similarity as _cosine_similarity,
    match_tracked_module as _match_tracked_module,
    parse_layer_index as _parse_layer_index,
    summarize_by_component,
    summarize_by_layer,
)
from pubmedqa.training.checkpoints import checkpoint_directory, write_training_state
from pubmedqa.training.contracts import (
    CheckpointRecord,
    EvalMetrics,
    EvalPrediction,
    EvalResult,
    LayerwiseReference,
    LayerwiseUpdateRecord,
    TrainingSummary,
    TrainStepLog,
)
from pubmedqa.training.data import (
    PubMedQASupervisedDataset,
    SupervisedDataCollator,
    SupervisedExample,
    longest_common_prefix_length as _longest_common_prefix_length,
)
from pubmedqa.training.schedules import (
    build_checkpoint_schedule as _build_checkpoint_schedule,
    normalize_checkpoint_percents as _normalize_checkpoint_percents,
)
from pubmedqa.training.validation import build_eval_result, write_eval_result

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Environment variables and defaults
# ---------------------------------------------------------------------------

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
ENV_TARGET_MODULES = TRAIN_LAYER_CONFIG.env_target_modules
ENV_TARGET_LAYERS = TRAIN_LAYER_CONFIG.env_target_layers
ENV_LAYER_SCOPE = TRAIN_LAYER_CONFIG.env_layer_scope
ENV_LORA_RANK = TRAIN_LAYER_CONFIG.env_lora_rank
ENV_LORA_ALPHA = TRAIN_LAYER_CONFIG.env_lora_alpha
ENV_LORA_DROPOUT = TRAIN_LAYER_CONFIG.env_lora_dropout
ENV_NOTES = TRAIN_LAYER_CONFIG.env_notes
ENV_TRACK_LAYERWISE_UPDATES = TRAIN_LAYER_CONFIG.env_track_layerwise_updates
ENV_CHECKPOINT_PERCENTS = TRAIN_LAYER_CONFIG.env_checkpoint_percents
ENV_DISTRIBUTED_MODE = TRAIN_FULL_FINE_TUNE_CONFIG.env_distributed_mode
ENV_FSDP_CPU_OFFLOAD = TRAIN_FULL_FINE_TUNE_CONFIG.env_fsdp_cpu_offload

DEFAULT_MODEL_NAME = TRAIN_FULL_FINE_TUNE_CONFIG.default_model_name
DEFAULT_CONDITION = TRAIN_FULL_FINE_TUNE_CONFIG.default_condition
DEFAULT_OUTPUT_DIR = TRAIN_FULL_FINE_TUNE_CONFIG.default_output_dir
DEFAULT_NUM_EPOCHS = TRAIN_FULL_FINE_TUNE_CONFIG.default_num_epochs
DEFAULT_TRAIN_BATCH_SIZE = TRAIN_FULL_FINE_TUNE_CONFIG.default_train_batch_size
DEFAULT_EVAL_BATCH_SIZE = TRAIN_FULL_FINE_TUNE_CONFIG.default_eval_batch_size
DEFAULT_GRAD_ACCUM_STEPS = TRAIN_FULL_FINE_TUNE_CONFIG.default_grad_accum_steps
DEFAULT_LEARNING_RATE = TRAIN_FULL_FINE_TUNE_CONFIG.default_learning_rate
DEFAULT_WEIGHT_DECAY = TRAIN_FULL_FINE_TUNE_CONFIG.default_weight_decay
DEFAULT_WARMUP_RATIO = TRAIN_FULL_FINE_TUNE_CONFIG.default_warmup_ratio
DEFAULT_MAX_GRAD_NORM = TRAIN_FULL_FINE_TUNE_CONFIG.default_max_grad_norm
DEFAULT_MAX_NEW_TOKENS = TRAIN_FULL_FINE_TUNE_CONFIG.default_max_new_tokens
DEFAULT_DEVICE = TRAIN_FULL_FINE_TUNE_CONFIG.default_device
DEFAULT_DTYPE = TRAIN_FULL_FINE_TUNE_CONFIG.default_dtype
DEFAULT_ATTN_IMPLEMENTATION = TRAIN_FULL_FINE_TUNE_CONFIG.default_attn_implementation
DEFAULT_CPU_THREADS = TRAIN_FULL_FINE_TUNE_CONFIG.default_cpu_threads
DEFAULT_LOG_EVERY_STEPS = TRAIN_FULL_FINE_TUNE_CONFIG.default_log_every_steps
DEFAULT_NUM_WORKERS = TRAIN_FULL_FINE_TUNE_CONFIG.default_num_workers
DEFAULT_SAVE_EVERY_EPOCH = TRAIN_FULL_FINE_TUNE_CONFIG.default_save_every_epoch
DEFAULT_EVAL_EVERY_EPOCH = TRAIN_FULL_FINE_TUNE_CONFIG.default_eval_every_epoch
DEFAULT_GRADIENT_CHECKPOINTING = TRAIN_FULL_FINE_TUNE_CONFIG.default_gradient_checkpointing
DEFAULT_SAVE_OPTIMIZER_STATE = TRAIN_FULL_FINE_TUNE_CONFIG.default_save_optimizer_state
DEFAULT_STRICT_PARSER = TRAIN_FULL_FINE_TUNE_CONFIG.default_strict_parser
DEFAULT_SEED = TRAIN_FULL_FINE_TUNE_CONFIG.default_seed
DEFAULT_METHOD_NAME = TRAIN_FULL_FINE_TUNE_CONFIG.default_method_name
DEFAULT_RUN_TAG = TRAIN_FULL_FINE_TUNE_CONFIG.default_run_tag
DEFAULT_DATA_REGIME = TRAIN_FULL_FINE_TUNE_CONFIG.default_data_regime
DEFAULT_DATA_FRACTION = TRAIN_FULL_FINE_TUNE_CONFIG.default_data_fraction
DEFAULT_TARGET_MODULES = TRAIN_LAYER_CONFIG.default_target_modules
DEFAULT_TARGET_LAYERS = TRAIN_LAYER_CONFIG.default_target_layers
DEFAULT_LAYER_SCOPE = TRAIN_LAYER_CONFIG.default_layer_scope
DEFAULT_LORA_RANK = TRAIN_LAYER_CONFIG.default_lora_rank
DEFAULT_LORA_ALPHA = TRAIN_LAYER_CONFIG.default_lora_alpha
DEFAULT_LORA_DROPOUT = TRAIN_LAYER_CONFIG.default_lora_dropout
DEFAULT_NOTES = TRAIN_LAYER_CONFIG.default_notes
DEFAULT_TRACK_LAYERWISE_UPDATES = TRAIN_LAYER_CONFIG.default_track_layerwise_updates
DEFAULT_CHECKPOINT_PERCENTS = TRAIN_LAYER_CONFIG.default_checkpoint_percents
DEFAULT_DISTRIBUTED_MODE = "single"
DEFAULT_FSDP_CPU_OFFLOAD = False
try:
    from torch.distributed.fsdp import (
        CPUOffload,
        FullStateDictConfig,
        FullyShardedDataParallel,
        MixedPrecision,
        ShardingStrategy,
        StateDictType,
    )
except ImportError:  # pragma: no cover - older torch builds
    CPUOffload = None
    FullStateDictConfig = None
    FullyShardedDataParallel = None
    MixedPrecision = None
    ShardingStrategy = None
    StateDictType = None




@dataclass(frozen=True)
class FullFineTuneConfig:
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
    lora_rank: int | None
    lora_alpha: float | None
    lora_dropout: float | None
    notes: str | None
    track_layerwise_updates: bool
    checkpoint_percents: tuple[int, ...]
    distributed_mode: str = DEFAULT_DISTRIBUTED_MODE
    fsdp_cpu_offload: bool = DEFAULT_FSDP_CPU_OFFLOAD


@dataclass(frozen=True)
class FullFineTuneCliConfig:
    config: FullFineTuneConfig
    environment: EnvironmentConfig

    @classmethod
    def from_env(cls) -> "FullFineTuneCliConfig":
        train_path = os.getenv(ENV_TRAIN_PATH)
        validation_path = os.getenv(ENV_VALIDATION_PATH)
        if not train_path or not validation_path:
            raise RuntimeError(
                f"{ENV_TRAIN_PATH} and {ENV_VALIDATION_PATH} are required for full fine-tuning."
            )

        attn = os.getenv(ENV_ATTN_IMPLEMENTATION, DEFAULT_ATTN_IMPLEMENTATION).strip().lower()
        if attn in {"", "none", "auto"}:
            attn = None

        config = FullFineTuneConfig(
            run_id=os.getenv(ENV_RUN_ID) or time.strftime("%Y%m%d_%H%M%S"),
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
            save_every_epoch=_env_bool(ENV_SAVE_EVERY_EPOCH, DEFAULT_SAVE_EVERY_EPOCH),
            eval_every_epoch=_env_bool(ENV_EVAL_EVERY_EPOCH, DEFAULT_EVAL_EVERY_EPOCH),
            max_train_examples=_env_optional_int(ENV_MAX_TRAIN_EXAMPLES),
            max_validation_examples=_env_optional_int(ENV_MAX_VALIDATION_EXAMPLES),
            max_test_examples=_env_optional_int(ENV_MAX_TEST_EXAMPLES),
            num_workers=_env_int(ENV_NUM_WORKERS, DEFAULT_NUM_WORKERS),
            gradient_checkpointing=_env_bool(
                ENV_GRADIENT_CHECKPOINTING,
                DEFAULT_GRADIENT_CHECKPOINTING,
            ),
            save_optimizer_state=_env_bool(
                ENV_SAVE_OPTIMIZER_STATE,
                DEFAULT_SAVE_OPTIMIZER_STATE,
            ),
            strict_parser=_env_bool(ENV_STRICT_PARSER, DEFAULT_STRICT_PARSER),
            seed=_env_int(ENV_SEED, DEFAULT_SEED),
            target_modules=_env_tuple(ENV_TARGET_MODULES) or DEFAULT_TARGET_MODULES,
            target_layers=_env_int_tuple(ENV_TARGET_LAYERS) or DEFAULT_TARGET_LAYERS,
            layer_scope=os.getenv(ENV_LAYER_SCOPE, DEFAULT_LAYER_SCOPE),
            lora_rank=_env_optional_int(ENV_LORA_RANK),
            lora_alpha=_env_optional_float(ENV_LORA_ALPHA),
            lora_dropout=_env_optional_float(ENV_LORA_DROPOUT),
            notes=_env_optional_str(ENV_NOTES),
            track_layerwise_updates=_env_bool(
                ENV_TRACK_LAYERWISE_UPDATES,
                DEFAULT_TRACK_LAYERWISE_UPDATES,
            ),
            checkpoint_percents=_normalize_checkpoint_percents(
                _env_int_tuple(ENV_CHECKPOINT_PERCENTS) or DEFAULT_CHECKPOINT_PERCENTS
            ),
            distributed_mode=os.getenv(ENV_DISTRIBUTED_MODE, DEFAULT_DISTRIBUTED_MODE),
            fsdp_cpu_offload=_env_bool(ENV_FSDP_CPU_OFFLOAD, DEFAULT_FSDP_CPU_OFFLOAD),
        )
        return cls(config=config, environment=EnvironmentConfig.from_env())






class PubMedQATrainingEngine:
    def __init__(self, config: FullFineTuneConfig, environment: EnvironmentConfig) -> None:
        self.config = config
        self.environment = environment
        self.device = _resolve_device(config.device)
        self.rank = 0
        self.local_rank = 0
        self.world_size = 1
        self.output_root = (
            config.output_dir / config.run_id / safe_name(config.model_name) / safe_name(config.condition)
        )
        self.checkpoints_dir = self.output_root / "checkpoints"
        self.logs_dir = self.output_root / "logs"
        self.evaluations_dir = self.output_root / "evaluations"
        self.layerwise_dir = self.output_root / "layerwise_updates"
        self.transitions_dir = self.output_root / "prediction_transitions"
        self.train_sampler: DistributedSampler[SupervisedExample] | None = None
        self._owns_process_group = False

    @property
    def title(self) -> str:
        return f"{self.config.run_id}__{safe_name(self.config.model_name)}__{safe_name(self.config.condition)}"

    def run(self) -> TrainingSummary:
        self._initialize_runtime_topology()
        self._prepare_output_dirs()
        self._set_runtime()

        start_time = current_time_iso()
        run_started = time.perf_counter()
        idle_memory = _memory_snapshot(self.device)

        tokenizer, model = self.load_model_and_tokenizer(self.config.model_name)
        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = False
        base_references = self.capture_layerwise_references(model)
        total_params, trainable_params, trainable_ratio = _count_parameters(model)

        train_examples = self._limit_examples(
            load_local_jsonl(self.config.train_path), self.config.max_train_examples
        )
        validation_examples = self._limit_examples(
            load_local_jsonl(self.config.validation_path), self.config.max_validation_examples
        )
        test_examples = self._limit_examples(
            load_local_jsonl(self.config.test_path), self.config.max_test_examples
        ) if self.config.test_path else []

        train_dataset = PubMedQASupervisedDataset(train_examples, tokenizer)
        validation_dataset = PubMedQASupervisedDataset(validation_examples, tokenizer)
        train_loader = self.build_train_dataloader(train_dataset, tokenizer)
        validation_loader = self.build_eval_dataloader(validation_dataset, tokenizer)

        reference_validation_result = self._run_on_main_process(
            lambda: self.evaluate_split(
                model=model,
                tokenizer=tokenizer,
                supervised_loader=validation_loader,
                examples=validation_examples,
                split_name="validation_reference",
            ),
            operation_name="reference validation",
        )

        model = self.wrap_model_for_training(model)

        loaded_memory = _memory_snapshot(self.device)

        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        steps_per_epoch = math.ceil(len(train_loader) / self.config.gradient_accumulation_steps)
        total_optimizer_steps = steps_per_epoch * self.config.num_epochs
        warmup_steps = int(total_optimizer_steps * self.config.warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_optimizer_steps,
        )
        checkpoint_schedule = _build_checkpoint_schedule(
            total_optimizer_steps,
            self.config.checkpoint_percents,
        )
        next_schedule_index = 0

        if self.is_main_process:
            write_json(
                self.output_root / "config.json",
                {
                    **asdict(self.config),
                    "dtype": str(self.config.dtype).replace("torch.", ""),
                    "environment": {"hf_token_set": bool(self.environment.hf_token)},
                    "distributed": self._distributed_metadata(),
                },
            )

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)

        step_logs: list[TrainStepLog] = []
        checkpoint_records: list[CheckpointRecord] = []
        best_checkpoint: CheckpointRecord | None = None
        last_validation_metrics: EvalMetrics | None = None
        global_step = 0
        total_seen_samples = 0
        total_seen_tokens = 0
        train_loss = 0.0
        cumulative_train_loss_total = 0.0
        cumulative_train_loss_count = 0
        saved_checkpoint_steps: set[int] = set()
        previous_snapshots = {
            reference.parameter_name: reference.base_tensor.clone()
            for reference in base_references
        }
        previous_incremental_updates: dict[str, torch.Tensor] = {}

        previous_validation_predictions: list[EvalPrediction] | None = None
        previous_validation_split_name: str | None = None

        checkpoint_records.append(
            self.record_checkpoint(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=0,
                step_in_epoch=0,
                global_step=0,
                checkpoint_kind="reference",
                checkpoint_percent=0.0,
                elapsed_seconds=0.0,
                train_loss=None,
                gradient_norm=None,
                validation_metrics=reference_validation_result.metrics,
                references=base_references,
                previous_snapshots=previous_snapshots,
                previous_incremental_updates={},
                save_model_files=False,
            )
        )
        last_validation_metrics = reference_validation_result.metrics
        previous_validation_predictions = reference_validation_result.predictions
        previous_validation_split_name = "validation_reference"

        for epoch in range(1, self.config.num_epochs + 1):
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            model.train()
            epoch_loss_total = 0.0
            epoch_loss_count = 0
            optimizer_step_started = time.perf_counter()
            accumulated_batch_size = 0
            accumulated_input_tokens = 0
            accumulated_target_tokens = 0
            accumulated_step_loss = 0.0
            accumulated_step_loss_count = 0
            optimizer.zero_grad(set_to_none=True)

            for batch_index, batch in enumerate(train_loader, start=1):
                batch = self._move_batch_to_device(batch)
                should_step = (
                    batch_index % self.config.gradient_accumulation_steps == 0
                    or batch_index == len(train_loader)
                )
                sync_context = nullcontext()
                if self.ddp_enabled and not should_step:
                    sync_context = model.no_sync()

                with sync_context:
                    with self._autocast_context():
                        outputs = model(**batch)
                        raw_loss = outputs.loss
                        loss = raw_loss / self.config.gradient_accumulation_steps

                    loss.backward()
                epoch_loss_total += float(raw_loss.detach().item())
                epoch_loss_count += 1
                cumulative_train_loss_total += float(raw_loss.detach().item())
                cumulative_train_loss_count += 1
                micro_batch_size = int(batch["input_ids"].shape[0])
                micro_input_tokens = int(batch["attention_mask"].sum().item())
                micro_target_tokens = int((batch["labels"] != -100).sum().item())
                accumulated_batch_size += micro_batch_size
                accumulated_input_tokens += micro_input_tokens
                accumulated_target_tokens += micro_target_tokens
                accumulated_step_loss += float(raw_loss.detach().item())
                accumulated_step_loss_count += 1
                total_seen_samples += micro_batch_size
                total_seen_tokens += micro_input_tokens
                del outputs, raw_loss, loss, batch

                if not should_step:
                    continue

                gradient_norm = float(
                    clip_grad_norm_(model.parameters(), self.config.max_grad_norm).detach().item()
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                step_elapsed = time.perf_counter() - optimizer_step_started

                step_log = TrainStepLog(
                    global_step=global_step,
                    epoch=epoch,
                    step_in_epoch=batch_index,
                    learning_rate=float(scheduler.get_last_lr()[0]),
                    train_loss=accumulated_step_loss / max(1, accumulated_step_loss_count),
                    gradient_norm=gradient_norm,
                    batch_size=accumulated_batch_size,
                    input_tokens=accumulated_input_tokens,
                    target_tokens=accumulated_target_tokens,
                    step_time_seconds=step_elapsed,
                    samples_per_second=(accumulated_batch_size / step_elapsed) if step_elapsed > 0 else 0.0,
                    tokens_per_second=(accumulated_input_tokens / step_elapsed) if step_elapsed > 0 else 0.0,
                )
                step_logs.append(step_log)
                accumulated_batch_size = 0
                accumulated_input_tokens = 0
                accumulated_target_tokens = 0
                accumulated_step_loss = 0.0
                accumulated_step_loss_count = 0

                if (
                    self.is_main_process
                    and self.config.log_every_steps > 0
                    and global_step % self.config.log_every_steps == 0
                ):
                    write_jsonl(
                        self.logs_dir / "train_steps.jsonl",
                        (asdict(record) for record in step_logs),
                    )

                while (
                    next_schedule_index < len(checkpoint_schedule)
                    and global_step >= checkpoint_schedule[next_schedule_index][1]
                ):
                    checkpoint_percent, _ = checkpoint_schedule[next_schedule_index]
                    split_name = f"validation_pct_{checkpoint_percent:03d}"
                    checkpoint_elapsed = time.perf_counter() - run_started
                    checkpoint_dir = self.save_checkpoint(
                        model=model,
                        tokenizer=tokenizer,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        step_in_epoch=batch_index,
                        global_step=global_step,
                        checkpoint_kind="scheduled",
                        checkpoint_percent=float(checkpoint_percent),
                        elapsed_seconds=checkpoint_elapsed,
                        validation_metrics=last_validation_metrics,
                        save_model_files=True,
                    )
                    validation_result = self._evaluate_checkpoint_on_main(
                        checkpoint_dir=checkpoint_dir,
                        examples=validation_examples,
                        split_name=split_name,
                    )
                    validation_metrics = validation_result.metrics
                    last_validation_metrics = validation_metrics
                    if (
                        self.is_main_process
                        and previous_validation_predictions is not None
                        and previous_validation_split_name is not None
                    ):
                        self.write_prediction_transition_artifacts(
                            from_split=previous_validation_split_name,
                            to_split=split_name,
                            previous_predictions=previous_validation_predictions,
                            current_predictions=validation_result.predictions,
                        )
                    previous_validation_predictions = validation_result.predictions
                    previous_validation_split_name = split_name
                    checkpoint_record = self.record_checkpoint(
                        model=model,
                        tokenizer=tokenizer,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        step_in_epoch=batch_index,
                        global_step=global_step,
                        checkpoint_kind="scheduled",
                        checkpoint_percent=float(checkpoint_percent),
                        elapsed_seconds=checkpoint_elapsed,
                        train_loss=(
                            cumulative_train_loss_total / cumulative_train_loss_count
                            if cumulative_train_loss_count > 0
                            else None
                        ),
                        gradient_norm=gradient_norm,
                        validation_metrics=validation_metrics,
                        references=base_references,
                        previous_snapshots=previous_snapshots,
                        previous_incremental_updates=previous_incremental_updates,
                        # Model files were saved before standalone validation.
                        save_model_files=False,
                    )
                    checkpoint_records.append(checkpoint_record)
                    saved_checkpoint_steps.add(global_step)
                    if best_checkpoint is None or self._is_better_checkpoint(checkpoint_record, best_checkpoint):
                        best_checkpoint = checkpoint_record
                    next_schedule_index += 1

                # Checkpoint evaluation and serialization are intentionally
                # excluded from the next optimizer-step throughput interval.
                optimizer_step_started = time.perf_counter()

            train_loss = epoch_loss_total / max(1, epoch_loss_count)
            validation_metrics: EvalMetrics | None = None
            epoch_checkpoint_saved = False
            if (
                self.config.eval_every_epoch
                and (not checkpoint_records or checkpoint_records[-1].global_step != global_step)
            ):
                split_name = f"validation_epoch_{epoch:03d}"
                checkpoint_percent = 100.0 * global_step / total_optimizer_steps
                checkpoint_elapsed = time.perf_counter() - run_started
                if self.config.save_every_epoch:
                    evaluation_checkpoint_dir = self.save_checkpoint(
                        model=model,
                        tokenizer=tokenizer,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        epoch=epoch,
                        step_in_epoch=len(train_loader),
                        global_step=global_step,
                        checkpoint_kind="epoch_end",
                        checkpoint_percent=checkpoint_percent,
                        elapsed_seconds=checkpoint_elapsed,
                        validation_metrics=last_validation_metrics,
                        save_model_files=True,
                    )
                    epoch_checkpoint_saved = True
                else:
                    evaluation_checkpoint_dir = self._create_evaluation_snapshot(
                        model=model,
                        tokenizer=tokenizer,
                        split_name=split_name,
                    )
                try:
                    validation_result = self._evaluate_checkpoint_on_main(
                        checkpoint_dir=evaluation_checkpoint_dir,
                        examples=validation_examples,
                        split_name=split_name,
                    )
                finally:
                    if not epoch_checkpoint_saved:
                        self._remove_evaluation_snapshot(evaluation_checkpoint_dir)
                validation_metrics = validation_result.metrics
                last_validation_metrics = validation_metrics
                if (
                    self.is_main_process
                    and previous_validation_predictions is not None
                    and previous_validation_split_name is not None
                ):
                    self.write_prediction_transition_artifacts(
                        from_split=previous_validation_split_name,
                        to_split=split_name,
                        previous_predictions=previous_validation_predictions,
                        current_predictions=validation_result.predictions,
                    )
                previous_validation_predictions = validation_result.predictions
                previous_validation_split_name = split_name

            if self.config.save_every_epoch and global_step not in saved_checkpoint_steps:
                if validation_metrics is None:
                    raise RuntimeError("Checkpoint saving requires validation metrics.")
                checkpoint_record = self.record_checkpoint(
                    model=model,
                    tokenizer=tokenizer,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    step_in_epoch=len(train_loader),
                    global_step=global_step,
                    checkpoint_kind="epoch_end",
                    checkpoint_percent=100.0 * global_step / total_optimizer_steps,
                    elapsed_seconds=time.perf_counter() - run_started,
                    train_loss=train_loss,
                    gradient_norm=step_logs[-1].gradient_norm if step_logs else None,
                    validation_metrics=validation_metrics,
                    references=base_references,
                    previous_snapshots=previous_snapshots,
                    previous_incremental_updates=previous_incremental_updates,
                    # A persisted snapshot was created before validation.
                    save_model_files=not epoch_checkpoint_saved,
                )
                checkpoint_records.append(checkpoint_record)
                if best_checkpoint is None or self._is_better_checkpoint(checkpoint_record, best_checkpoint):
                    best_checkpoint = checkpoint_record
                saved_checkpoint_steps.add(global_step)

            if self.is_main_process:
                write_jsonl(
                    self.logs_dir / "train_steps.jsonl",
                    (asdict(record) for record in step_logs),
                )
                if checkpoint_records:
                    write_jsonl(
                        self.logs_dir / "checkpoints.jsonl",
                        (asdict(record) for record in checkpoint_records),
                    )

        if last_validation_metrics is None:
            final_snapshot_dir = self._create_evaluation_snapshot(
                model=model,
                tokenizer=tokenizer,
                split_name="validation",
            )
            try:
                final_validation_result = self._evaluate_checkpoint_on_main(
                    checkpoint_dir=final_snapshot_dir,
                    examples=validation_examples,
                    split_name="validation",
                )
            finally:
                self._remove_evaluation_snapshot(final_snapshot_dir)
            last_validation_metrics = final_validation_result.metrics
            if (
                self.is_main_process
                and previous_validation_predictions is not None
                and previous_validation_split_name is not None
            ):
                self.write_prediction_transition_artifacts(
                    from_split=previous_validation_split_name,
                    to_split="validation",
                    previous_predictions=previous_validation_predictions,
                    current_predictions=final_validation_result.predictions,
                )

        if best_checkpoint is None:
            checkpoint_record = self.record_checkpoint(
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=self.config.num_epochs,
                step_in_epoch=len(train_loader),
                global_step=global_step,
                checkpoint_kind="final",
                checkpoint_percent=100.0,
                elapsed_seconds=time.perf_counter() - run_started,
                train_loss=train_loss,
                gradient_norm=step_logs[-1].gradient_norm if step_logs else None,
                validation_metrics=last_validation_metrics,
                references=base_references,
                previous_snapshots=previous_snapshots,
                previous_incremental_updates=previous_incremental_updates,
                save_model_files=True,
            )
            checkpoint_records.append(checkpoint_record)
            best_checkpoint = checkpoint_record
            if self.is_main_process:
                write_jsonl(
                    self.logs_dir / "checkpoints.jsonl",
                    (asdict(record) for record in checkpoint_records),
                )

        final_validation_metrics = last_validation_metrics
        del optimizer
        del scheduler
        del model
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

        test_metrics: EvalMetrics | None = None
        if test_examples:
            def evaluate_best_checkpoint() -> EvalResult:
                best_tokenizer, best_model = self.load_model_and_tokenizer(best_checkpoint.checkpoint_dir)
                try:
                    best_test_dataset = PubMedQASupervisedDataset(test_examples, best_tokenizer)
                    best_test_loader = self.build_eval_dataloader(best_test_dataset, best_tokenizer)
                    return self.evaluate_split(
                        model=best_model,
                        tokenizer=best_tokenizer,
                        supervised_loader=best_test_loader,
                        examples=test_examples,
                        split_name="test",
                    )
                finally:
                    del best_model
                    gc.collect()
                    if self.device.type == "cuda":
                        torch.cuda.empty_cache()

            test_result = self._run_on_main_process(
                evaluate_best_checkpoint,
                operation_name="best-checkpoint test evaluation",
            )
            test_metrics = test_result.metrics

        end_time = current_time_iso()
        total_elapsed = time.perf_counter() - run_started
        total_seen_samples = self._all_reduce_int(total_seen_samples)
        total_seen_tokens = self._all_reduce_int(total_seen_tokens)
        total_checkpoint_size = sum(record.checkpoint_size_bytes for record in checkpoint_records)
        final_checkpoint_size = checkpoint_records[-1].checkpoint_size_bytes
        best_checkpoint_size = best_checkpoint.checkpoint_size_bytes
        peak_train_memory = _memory_snapshot(self.device)
        distributed_runtime = self._collect_distributed_runtime(
            idle_memory=idle_memory,
            loaded_memory=loaded_memory,
            peak_train_memory=peak_train_memory,
        )

        summary = TrainingSummary(
            run_id=self.config.run_id,
            run_tag=self.config.run_tag,
            method_name=self.config.method_name,
            model_name=self.config.model_name,
            condition=self.config.condition,
            data_regime=self.config.data_regime,
            data_fraction=self.config.data_fraction,
            target_modules=self.config.target_modules,
            target_layers=self.config.target_layers,
            layer_scope=self.config.layer_scope,
            lora_rank=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            notes=self.config.notes,
            title=self.title,
            start_time=start_time,
            end_time=end_time,
            total_training_time_seconds=total_elapsed,
            epochs=self.config.num_epochs,
            optimizer_steps=global_step,
            steps_per_epoch=steps_per_epoch,
            train_examples=len(train_examples),
            validation_examples=len(validation_examples),
            test_examples=len(test_examples),
            total_params=total_params,
            trainable_params=trainable_params,
            trainable_ratio=trainable_ratio,
            idle_allocated_gb=idle_memory["allocated_gb"],
            idle_reserved_gb=idle_memory["reserved_gb"],
            model_loaded_allocated_gb=loaded_memory["allocated_gb"],
            model_loaded_reserved_gb=loaded_memory["reserved_gb"],
            peak_train_allocated_gb=peak_train_memory["max_allocated_gb"],
            peak_train_reserved_gb=peak_train_memory["max_reserved_gb"],
            final_train_loss=train_loss,
            best_checkpoint_dir=best_checkpoint.checkpoint_dir,
            best_checkpoint_kind=best_checkpoint.checkpoint_kind,
            best_checkpoint_percent=best_checkpoint.checkpoint_percent,
            best_epoch=best_checkpoint.epoch,
            best_validation_accuracy=best_checkpoint.validation_accuracy,
            best_validation_macro_f1=best_checkpoint.validation_macro_f1,
            best_validation_class_f1=best_checkpoint.validation_class_f1,
            best_validation_loss=best_checkpoint.validation_loss,
            final_validation_accuracy=final_validation_metrics.accuracy,
            final_validation_macro_f1=final_validation_metrics.macro_f1,
            final_validation_class_f1=final_validation_metrics.class_f1,
            final_validation_loss=final_validation_metrics.loss,
            test_accuracy=None if test_metrics is None else test_metrics.accuracy,
            test_macro_f1=None if test_metrics is None else test_metrics.macro_f1,
            test_class_f1=None if test_metrics is None else test_metrics.class_f1,
            test_loss=None if test_metrics is None else test_metrics.loss,
            test_invalid_rate=None if test_metrics is None else test_metrics.invalid_rate,
            train_val_gap_loss=train_loss - final_validation_metrics.loss,
            train_val_gap_accuracy=None,
            train_val_gap_macro_f1=None,
            total_checkpoint_size_bytes=total_checkpoint_size,
            best_checkpoint_size_bytes=best_checkpoint_size,
            final_checkpoint_size_bytes=final_checkpoint_size,
            training_samples_per_second=(total_seen_samples / total_elapsed) if total_elapsed > 0 else 0.0,
            training_tokens_per_second=(total_seen_tokens / total_elapsed) if total_elapsed > 0 else 0.0,
            training_seconds_per_step=(total_elapsed / global_step) if global_step > 0 else 0.0,
            inference_examples_per_second=final_validation_metrics.examples_per_second,
            inference_avg_latency_seconds=final_validation_metrics.avg_latency_seconds,
            inference_peak_allocated_gb=final_validation_metrics.peak_allocated_gb,
            inference_peak_reserved_gb=final_validation_metrics.peak_reserved_gb,
        )

        if self.is_main_process:
            write_json(self.output_root / "summary.json", asdict(summary))
            write_json(self.output_root / "distributed_runtime.json", distributed_runtime)
            write_json(
                self.output_root / "run_metadata.json",
                {
                    "run_id": self.config.run_id,
                    "run_tag": self.config.run_tag,
                    "method_name": self.config.method_name,
                    "condition": self.config.condition,
                    "model_name": self.config.model_name,
                    "data_regime": self.config.data_regime,
                    "data_fraction": self.config.data_fraction,
                    "target_modules": list(self.config.target_modules),
                    "target_layers": list(self.config.target_layers),
                    "layer_scope": self.config.layer_scope,
                    "lora_rank": self.config.lora_rank,
                    "lora_alpha": self.config.lora_alpha,
                    "lora_dropout": self.config.lora_dropout,
                    "notes": self.config.notes,
                    "track_layerwise_updates": self.config.track_layerwise_updates,
                    "created_at": current_time_iso(),
                    "distributed": self._distributed_metadata(),
                },
            )
            self.write_analysis_groups(summary, checkpoint_records, step_logs)
            write_json(
                self.output_root / "artifacts.json",
                {
                    "checkpoints": [asdict(record) for record in checkpoint_records],
                    "best_checkpoint": asdict(best_checkpoint),
                },
            )
            if test_metrics is not None:
                write_json(self.evaluations_dir / "test_summary.json", asdict(test_metrics))
        return summary

    def load_model_and_tokenizer(self, model_name_or_path: str) -> tuple[Any, torch.nn.Module]:
        common_kwargs = {
            "token": self.environment.hf_token,
            "trust_remote_code": self.config.trust_remote_code,
        }
        model_kwargs = {
            **common_kwargs,
            "torch_dtype": self.config.dtype,
        }
        if self.config.attn_implementation is not None:
            model_kwargs["attn_implementation"] = self.config.attn_implementation

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **common_kwargs)
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise RuntimeError(f"Tokenizer for {model_name_or_path} has neither pad_token nor eos_token.")
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        try:
            model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
        except ValueError:
            from transformers import AutoModelForImageTextToText

            model = AutoModelForImageTextToText.from_pretrained(model_name_or_path, **model_kwargs)

        model.to(self.device)
        return tokenizer, model

    def build_train_dataloader(self, dataset: PubMedQASupervisedDataset, tokenizer: Any) -> DataLoader[Any]:
        generator = torch.Generator()
        generator.manual_seed(self.config.seed)
        sampler = None
        shuffle = True
        if self.world_size > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True,
                seed=self.config.seed,
                drop_last=False,
            )
            shuffle = False
        self.train_sampler = sampler
        return DataLoader(
            dataset,
            batch_size=self.config.train_batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.config.num_workers,
            collate_fn=SupervisedDataCollator(tokenizer, self.config.max_input_tokens),
            generator=generator,
        )

    def build_eval_dataloader(self, dataset: PubMedQASupervisedDataset, tokenizer: Any) -> DataLoader[Any]:
        return DataLoader(
            dataset,
            batch_size=self.config.eval_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=SupervisedDataCollator(tokenizer, self.config.max_input_tokens),
        )

    def evaluate_split(
        self,
        *,
        model: torch.nn.Module,
        tokenizer: Any,
        supervised_loader: DataLoader[Any],
        examples: Sequence[PubMedQAExample],
        split_name: str,
    ) -> EvalResult:
        model.eval()
        evaluation_started = time.perf_counter()
        loss_total = 0.0
        loss_count = 0
        predictions: list[EvalPrediction] = []
        total_input_tokens = 0
        total_generation_time = 0.0
        original_padding_side = tokenizer.padding_side

        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(self.device)

        with torch.no_grad():
            for batch in supervised_loader:
                batch = self._move_batch_to_device(batch)
                with self._autocast_context():
                    outputs = model(**batch)
                loss_total += float(outputs.loss.detach().item())
                loss_count += 1

            tokenizer.padding_side = "left"
            for batch_start in range(0, len(examples), self.config.eval_batch_size):
                batch_examples = examples[batch_start : batch_start + self.config.eval_batch_size]
                prompts = [build_tokenizer_prompt(tokenizer, example, include_answer=False) for example in batch_examples]
                tokenize_kwargs: dict[str, Any] = {
                    "return_tensors": "pt",
                    "padding": True,
                    "truncation": self.config.max_input_tokens is not None,
                }
                if self.config.max_input_tokens is not None:
                    tokenize_kwargs["max_length"] = self.config.max_input_tokens
                encoded = tokenizer(prompts, **tokenize_kwargs)
                encoded = {
                    key: value.to(self.device)
                    for key, value in encoded.items()
                    if isinstance(value, torch.Tensor)
                }
                input_width = encoded["input_ids"].shape[1]
                total_input_tokens += int(encoded["attention_mask"].sum().item())
                started = time.perf_counter()
                generated = model.generate(
                    **encoded,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,
                )
                elapsed = time.perf_counter() - started
                total_generation_time += elapsed
                responses = tokenizer.batch_decode(generated[:, input_width:], skip_special_tokens=True)
                for example, response_text in zip(batch_examples, responses):
                    parsed = parse_pubmedqa_answer(response_text.strip(), strict=self.config.strict_parser)
                    predictions.append(
                        EvalPrediction(
                            pubid=example.pubid,
                            gold_label=example.final_decision or "",
                            predicted_label=parsed.label,
                            response_text=response_text.strip(),
                            parse_method=parsed.method,
                            parse_error=parsed.error,
                        )
                    )

        tokenizer.padding_side = original_padding_side
        peak_memory = _memory_snapshot(self.device)
        total_elapsed = time.perf_counter() - evaluation_started
        result = build_eval_result(
            split_name=split_name,
            loss=(loss_total / loss_count) if loss_count > 0 else 0.0,
            predictions=predictions,
            elapsed_seconds=total_elapsed,
            input_tokens=total_input_tokens,
            peak_allocated_gb=peak_memory["max_allocated_gb"],
            peak_reserved_gb=peak_memory["max_reserved_gb"],
        )
        write_eval_result(self.evaluations_dir, split_name, result)
        model.train()
        return result

    def _run_on_main_process(
        self,
        operation: Callable[[], T],
        *,
        operation_name: str,
    ) -> T:
        payload: list[dict[str, Any] | None] = [None]
        if self.is_main_process:
            try:
                payload[0] = {"ok": True, "result": operation()}
            except Exception as exc:
                payload[0] = {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
        if dist.is_initialized():
            dist.broadcast_object_list(payload, src=0)

        message = payload[0]
        if message is None:
            raise RuntimeError(f"{operation_name} produced no rank-0 result.")
        if not message["ok"]:
            raise RuntimeError(
                f"{operation_name} failed on rank 0: "
                f"{message['error_type']}: {message['error']}"
            )
        return cast(T, message["result"])

    def _evaluate_checkpoint_on_main(
        self,
        *,
        checkpoint_dir: Path,
        examples: Sequence[PubMedQAExample],
        split_name: str,
    ) -> EvalResult:
        def evaluate_checkpoint() -> EvalResult:
            eval_tokenizer, eval_model = self.load_model_and_tokenizer(str(checkpoint_dir))
            try:
                eval_dataset = PubMedQASupervisedDataset(examples, eval_tokenizer)
                eval_loader = self.build_eval_dataloader(eval_dataset, eval_tokenizer)
                return self.evaluate_split(
                    model=eval_model,
                    tokenizer=eval_tokenizer,
                    supervised_loader=eval_loader,
                    examples=examples,
                    split_name=split_name,
                )
            finally:
                del eval_model
                gc.collect()
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()

        return self._run_on_main_process(
            evaluate_checkpoint,
            operation_name=f"{split_name} checkpoint evaluation",
        )

    def _checkpoint_directory(
        self,
        *,
        checkpoint_kind: str,
        checkpoint_percent: float,
        epoch: int,
        global_step: int,
    ) -> Path:
        return checkpoint_directory(
            self.checkpoints_dir,
            checkpoint_kind=checkpoint_kind,
            checkpoint_percent=checkpoint_percent,
            epoch=epoch,
            global_step=global_step,
        )

    def _save_model_files_to_directory(
        self,
        *,
        model: torch.nn.Module,
        tokenizer: Any,
        checkpoint_dir: Path,
    ) -> None:
        if self.is_main_process:
            model_to_save = self._unwrap_model(model)
            if self.fsdp_enabled:
                if (
                    FullyShardedDataParallel is None
                    or FullStateDictConfig is None
                    or StateDictType is None
                ):
                    raise RuntimeError("FSDP checkpoint saving is not available in this torch build.")
                save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                with FullyShardedDataParallel.state_dict_type(
                    model_to_save,
                    StateDictType.FULL_STATE_DICT,
                    save_policy,
                ):
                    state_dict = model_to_save.state_dict()
                model_to_save.save_pretrained(checkpoint_dir, state_dict=state_dict)
            else:
                model_to_save.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)

    def _create_evaluation_snapshot(
        self,
        *,
        model: torch.nn.Module,
        tokenizer: Any,
        split_name: str,
    ) -> Path:
        snapshot_dir = self.output_root / ".evaluation_snapshots" / safe_name(split_name)
        if self.is_main_process:
            snapshot_dir.mkdir(parents=True, exist_ok=True)
        self._barrier()
        self._save_model_files_to_directory(
            model=model,
            tokenizer=tokenizer,
            checkpoint_dir=snapshot_dir,
        )
        self._barrier()
        return snapshot_dir

    def _remove_evaluation_snapshot(self, snapshot_dir: Path) -> None:
        if self.is_main_process and snapshot_dir.exists():
            shutil.rmtree(snapshot_dir)

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
        validation_metrics: EvalMetrics,
        save_model_files: bool,
    ) -> Path:
        checkpoint_dir = self._checkpoint_directory(
            checkpoint_kind=checkpoint_kind,
            checkpoint_percent=checkpoint_percent,
            epoch=epoch,
            global_step=global_step,
        )
        if self.is_main_process:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._barrier()
        if save_model_files:
            self._save_model_files_to_directory(
                model=model,
                tokenizer=tokenizer,
                checkpoint_dir=checkpoint_dir,
            )
        if save_model_files and self.is_main_process:
            if self.config.save_optimizer_state:
                torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
                torch.save(scheduler.state_dict(), checkpoint_dir / "scheduler.pt")
        self._barrier()
        if self.is_main_process:
            write_training_state(
                checkpoint_dir,
                checkpoint_kind=checkpoint_kind,
                checkpoint_percent=checkpoint_percent,
                epoch=epoch,
                step_in_epoch=step_in_epoch,
                global_step=global_step,
                elapsed_seconds=elapsed_seconds,
                validation_metrics=validation_metrics,
                title=self.title,
                run_id=self.config.run_id,
                run_tag=self.config.run_tag,
                model_name=self.config.model_name,
                condition=self.config.condition,
                distributed=self._distributed_metadata(),
            )
        self._barrier()
        return checkpoint_dir

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
        validation_metrics: EvalMetrics,
        references: Sequence[LayerwiseReference],
        previous_snapshots: dict[str, torch.Tensor],
        previous_incremental_updates: dict[str, torch.Tensor],
        save_model_files: bool,
    ) -> CheckpointRecord:
        checkpoint_dir = self.save_checkpoint(
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
            validation_metrics=validation_metrics,
            save_model_files=save_model_files,
        )
        checkpoint_size = _count_directory_size_bytes(checkpoint_dir)
        peak_memory = _memory_snapshot(self.device)
        self.write_layerwise_update_artifacts(
            model=model,
            checkpoint_kind=checkpoint_kind,
            checkpoint_percent=checkpoint_percent,
            epoch=epoch,
            global_step=global_step,
            checkpoint_dir=checkpoint_dir,
            references=references,
            previous_snapshots=previous_snapshots,
            previous_incremental_updates=previous_incremental_updates,
        )
        return CheckpointRecord(
            checkpoint_kind=checkpoint_kind,
            checkpoint_percent=checkpoint_percent,
            epoch=epoch,
            step_in_epoch=step_in_epoch,
            global_step=global_step,
            elapsed_seconds=elapsed_seconds,
            checkpoint_dir=str(checkpoint_dir),
            checkpoint_size_bytes=checkpoint_size,
            train_loss=0.0 if train_loss is None else train_loss,
            validation_loss=validation_metrics.loss,
            validation_accuracy=validation_metrics.accuracy,
            validation_macro_f1=validation_metrics.macro_f1,
            validation_class_f1=validation_metrics.class_f1,
            validation_invalid_rate=validation_metrics.invalid_rate,
            learning_rate=float(scheduler.get_last_lr()[0]),
            gradient_norm=0.0 if gradient_norm is None else gradient_norm,
            peak_allocated_gb=peak_memory["max_allocated_gb"],
            peak_reserved_gb=peak_memory["max_reserved_gb"],
        )

    def capture_layerwise_references(self, model: torch.nn.Module) -> list[LayerwiseReference]:
        model = self._unwrap_model(model)
        references: list[LayerwiseReference] = []
        for parameter_name, parameter in model.named_parameters():
            match = _match_tracked_module(parameter_name)
            if match is None:
                continue
            module_name, component_name = match
            base_tensor = parameter.detach().cpu().clone()
            if base_tensor.is_floating_point():
                base_tensor = base_tensor.to(torch.float16)
            references.append(
                LayerwiseReference(
                    parameter_name=parameter_name,
                    layer_index=_parse_layer_index(parameter_name),
                    module_name=module_name,
                    component_name=component_name,
                    shape=tuple(parameter.shape),
                    num_parameters=parameter.numel(),
                    base_weight_norm=float(parameter.detach().float().norm().item()),
                    base_tensor=base_tensor,
                )
            )
        write_json(
            self.layerwise_dir / "base_reference_summary.json",
            {
                "run_id": self.config.run_id,
                "run_tag": self.config.run_tag,
                "model_name": self.config.model_name,
                "tracked_parameters": [
                    {
                        "parameter_name": reference.parameter_name,
                        "layer_index": reference.layer_index,
                        "module_name": reference.module_name,
                        "component_name": reference.component_name,
                        "shape": list(reference.shape),
                        "num_parameters": reference.num_parameters,
                        "base_weight_norm": reference.base_weight_norm,
                    }
                    for reference in references
                ],
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
        if not self.config.track_layerwise_updates or not references:
            return

        named_parameters = dict(self._unwrap_model(model).named_parameters())
        records: list[LayerwiseUpdateRecord] = []
        current_snapshots: dict[str, torch.Tensor] = {}
        for reference in references:
            parameter = named_parameters.get(reference.parameter_name)
            if parameter is None:
                continue
            current_tensor = parameter.detach().cpu().float()
            base_tensor = reference.base_tensor.float()
            previous_tensor = previous_snapshots.get(reference.parameter_name, base_tensor).float()
            previous_incremental_tensor = previous_incremental_updates.get(reference.parameter_name)
            update_tensor = current_tensor - base_tensor
            incremental_update_tensor = current_tensor - previous_tensor
            current_weight_norm = float(current_tensor.norm().item())
            update_norm = float(update_tensor.norm().item())
            incremental_update_norm = float(incremental_update_tensor.norm().item())
            relative_update_norm = (
                0.0 if reference.base_weight_norm == 0.0 else update_norm / reference.base_weight_norm
            )
            relative_incremental_update_norm = (
                0.0
                if reference.base_weight_norm == 0.0
                else incremental_update_norm / reference.base_weight_norm
            )
            incremental_update_cosine_similarity = None
            if previous_incremental_tensor is not None:
                incremental_update_cosine_similarity = _cosine_similarity(
                    incremental_update_tensor,
                    previous_incremental_tensor.float(),
                )
            records.append(
                LayerwiseUpdateRecord(
                    checkpoint_kind=checkpoint_kind,
                    checkpoint_percent=checkpoint_percent,
                    epoch=epoch,
                    global_step=global_step,
                    layer_index=reference.layer_index,
                    module_name=reference.module_name,
                    component_name=reference.component_name,
                    parameter_name=reference.parameter_name,
                    shape=reference.shape,
                    num_parameters=reference.num_parameters,
                    base_weight_norm=reference.base_weight_norm,
                    current_weight_norm=current_weight_norm,
                    update_norm=update_norm,
                    relative_update_norm=relative_update_norm,
                    incremental_update_norm=incremental_update_norm,
                    relative_incremental_update_norm=relative_incremental_update_norm,
                    incremental_update_cosine_similarity=incremental_update_cosine_similarity,
                    cumulative_update_share=0.0,
                    incremental_update_share=0.0,
                )
            )
            current_snapshots[reference.parameter_name] = current_tensor.to(torch.float16)
            previous_incremental_updates[reference.parameter_name] = incremental_update_tensor.to(torch.float16)

        total_cumulative_update = sum(record.update_norm for record in records)
        total_incremental_update = sum(record.incremental_update_norm for record in records)
        normalized_records: list[LayerwiseUpdateRecord] = []
        for record in records:
            normalized_records.append(
                LayerwiseUpdateRecord(
                    checkpoint_kind=record.checkpoint_kind,
                    checkpoint_percent=record.checkpoint_percent,
                    epoch=record.epoch,
                    global_step=record.global_step,
                    layer_index=record.layer_index,
                    module_name=record.module_name,
                    component_name=record.component_name,
                    parameter_name=record.parameter_name,
                    shape=record.shape,
                    num_parameters=record.num_parameters,
                    base_weight_norm=record.base_weight_norm,
                    current_weight_norm=record.current_weight_norm,
                    update_norm=record.update_norm,
                    relative_update_norm=record.relative_update_norm,
                    incremental_update_norm=record.incremental_update_norm,
                    relative_incremental_update_norm=record.relative_incremental_update_norm,
                    incremental_update_cosine_similarity=record.incremental_update_cosine_similarity,
                    cumulative_update_share=(
                        0.0 if total_cumulative_update == 0.0 else record.update_norm / total_cumulative_update
                    ),
                    incremental_update_share=(
                        0.0
                        if total_incremental_update == 0.0
                        else record.incremental_update_norm / total_incremental_update
                    ),
                )
            )

        previous_snapshots.update(current_snapshots)

        rows = [asdict(record) for record in normalized_records]
        file_stem = (
            f"{checkpoint_kind}_pct_{int(round(checkpoint_percent)):03d}_"
            f"epoch_{epoch:03d}_step_{global_step:06d}"
        )
        write_jsonl(self.layerwise_dir / f"{file_stem}.jsonl", rows)
        write_jsonl(checkpoint_dir / "layerwise_updates.jsonl", rows)
        write_json(
            self.layerwise_dir / f"{file_stem}_summary.json",
            {
                "checkpoint_kind": checkpoint_kind,
                "checkpoint_percent": checkpoint_percent,
                "epoch": epoch,
                "global_step": global_step,
                "num_records": len(normalized_records),
                "total_cumulative_update_norm": total_cumulative_update,
                "total_incremental_update_norm": total_incremental_update,
                "by_component": self._summarize_layerwise_by_component(normalized_records),
                "by_layer": self._summarize_layerwise_by_layer(normalized_records),
                "avg_incremental_update_cosine_similarity": self._average_cosine_similarity(
                    normalized_records
                ),
            },
        )

    @staticmethod
    def _summarize_layerwise_by_component(
        records: Sequence[LayerwiseUpdateRecord],
    ) -> dict[str, dict[str, float | int]]:
        return summarize_by_component(records)

    @staticmethod
    def _summarize_layerwise_by_layer(
        records: Sequence[LayerwiseUpdateRecord],
    ) -> dict[str, dict[str, float | int]]:
        return summarize_by_layer(records)

    @staticmethod
    def _average_cosine_similarity(records: Sequence[LayerwiseUpdateRecord]) -> float | None:
        return average_incremental_cosine_similarity(records)

    def write_prediction_transition_artifacts(
        self,
        *,
        from_split: str,
        to_split: str,
        previous_predictions: Sequence[EvalPrediction],
        current_predictions: Sequence[EvalPrediction],
    ) -> None:
        previous_by_pubid = {prediction.pubid: prediction for prediction in previous_predictions}
        rows: list[dict[str, Any]] = []
        counts = {
            "wrong_to_correct": 0,
            "correct_to_wrong": 0,
            "wrong_to_wrong_changed": 0,
            "correct_to_correct": 0,
            "missing_previous": 0,
        }

        for prediction in current_predictions:
            previous = previous_by_pubid.get(prediction.pubid)
            if previous is None:
                counts["missing_previous"] += 1
                continue
            prev_correct = previous.predicted_label == previous.gold_label
            curr_correct = prediction.predicted_label == prediction.gold_label
            transition_type = "unchanged"
            if not prev_correct and curr_correct:
                transition_type = "wrong_to_correct"
                counts["wrong_to_correct"] += 1
            elif prev_correct and not curr_correct:
                transition_type = "correct_to_wrong"
                counts["correct_to_wrong"] += 1
            elif not prev_correct and not curr_correct:
                transition_type = "wrong_to_wrong_changed" if previous.predicted_label != prediction.predicted_label else "wrong_to_wrong_same"
                if transition_type == "wrong_to_wrong_changed":
                    counts["wrong_to_wrong_changed"] += 1
            else:
                transition_type = "correct_to_correct"
                counts["correct_to_correct"] += 1

            rows.append(
                {
                    "pubid": prediction.pubid,
                    "gold_label": prediction.gold_label,
                    "from_split": from_split,
                    "to_split": to_split,
                    "from_prediction": previous.predicted_label,
                    "to_prediction": prediction.predicted_label,
                    "from_correct": prev_correct,
                    "to_correct": curr_correct,
                    "transition_type": transition_type,
                }
            )

        stem = f"{from_split}__to__{to_split}"
        write_json(self.transitions_dir / f"{stem}_summary.json", counts)
        write_jsonl(self.transitions_dir / f"{stem}.jsonl", rows)

    def write_analysis_groups(
        self,
        summary: TrainingSummary,
        checkpoint_records: Sequence[CheckpointRecord],
        step_logs: Sequence[TrainStepLog],
    ) -> None:
        write_json(
            self.output_root / "analysis" / "parameter_model_size.json",
            {
                "run_tag": summary.run_tag,
                "method_name": summary.method_name,
                "total_params": summary.total_params,
                "trainable_params": summary.trainable_params,
                "trainable_ratio": summary.trainable_ratio,
                "best_checkpoint_size_bytes": summary.best_checkpoint_size_bytes,
                "final_checkpoint_size_bytes": summary.final_checkpoint_size_bytes,
                "total_checkpoint_size_bytes": summary.total_checkpoint_size_bytes,
            },
        )
        write_json(
            self.output_root / "analysis" / "memory.json",
            {
                "run_tag": summary.run_tag,
                "method_name": summary.method_name,
                "idle_allocated_gb": summary.idle_allocated_gb,
                "idle_reserved_gb": summary.idle_reserved_gb,
                "model_loaded_allocated_gb": summary.model_loaded_allocated_gb,
                "model_loaded_reserved_gb": summary.model_loaded_reserved_gb,
                "peak_train_allocated_gb": summary.peak_train_allocated_gb,
                "peak_train_reserved_gb": summary.peak_train_reserved_gb,
                "inference_peak_allocated_gb": summary.inference_peak_allocated_gb,
                "inference_peak_reserved_gb": summary.inference_peak_reserved_gb,
            },
        )
        write_json(
            self.output_root / "analysis" / "training_efficiency.json",
            {
                "run_tag": summary.run_tag,
                "method_name": summary.method_name,
                "total_training_time_seconds": summary.total_training_time_seconds,
                "training_seconds_per_step": summary.training_seconds_per_step,
                "training_samples_per_second": summary.training_samples_per_second,
                "training_tokens_per_second": summary.training_tokens_per_second,
                "optimizer_steps": summary.optimizer_steps,
                "steps_per_epoch": summary.steps_per_epoch,
            },
        )
        write_json(
            self.output_root / "analysis" / "optimization.json",
            {
                "run_tag": summary.run_tag,
                "method_name": summary.method_name,
                "final_train_loss": summary.final_train_loss,
                "final_validation_loss": summary.final_validation_loss,
                "best_validation_loss": summary.best_validation_loss,
                "train_val_gap_loss": summary.train_val_gap_loss,
                "train_val_gap_accuracy": summary.train_val_gap_accuracy,
                "train_val_gap_macro_f1": summary.train_val_gap_macro_f1,
            },
        )
        write_json(
            self.output_root / "analysis" / "learning_dynamics.json",
            {
                "run_tag": summary.run_tag,
                "method_name": summary.method_name,
                "checkpoint_percents": list(self.config.checkpoint_percents),
                "checkpoint_timeline": [asdict(record) for record in checkpoint_records],
            },
        )
        write_json(
            self.output_root / "analysis" / "performance_dynamics.json",
            {
                "run_tag": summary.run_tag,
                "method_name": summary.method_name,
                "checkpoint_timeline": [
                    {
                        "checkpoint_kind": record.checkpoint_kind,
                        "checkpoint_percent": record.checkpoint_percent,
                        "epoch": record.epoch,
                        "step_in_epoch": record.step_in_epoch,
                        "global_step": record.global_step,
                        "elapsed_seconds": record.elapsed_seconds,
                        "train_loss": record.train_loss,
                        "validation_loss": record.validation_loss,
                        "validation_accuracy": record.validation_accuracy,
                        "validation_macro_f1": record.validation_macro_f1,
                        "validation_class_f1": record.validation_class_f1,
                        "validation_invalid_rate": record.validation_invalid_rate,
                    }
                    for record in checkpoint_records
                ],
                "step_logs_file": str(self.logs_dir / "train_steps.jsonl"),
            },
        )
        write_json(
            self.output_root / "analysis" / "performance_generalization.json",
            {
                "run_tag": summary.run_tag,
                "method_name": summary.method_name,
                "best_checkpoint_dir": summary.best_checkpoint_dir,
                "best_checkpoint_kind": summary.best_checkpoint_kind,
                "best_checkpoint_percent": summary.best_checkpoint_percent,
                "best_epoch": summary.best_epoch,
                "best_validation_accuracy": summary.best_validation_accuracy,
                "best_validation_macro_f1": summary.best_validation_macro_f1,
                "best_validation_class_f1": summary.best_validation_class_f1,
                "final_validation_accuracy": summary.final_validation_accuracy,
                "final_validation_macro_f1": summary.final_validation_macro_f1,
                "final_validation_class_f1": summary.final_validation_class_f1,
                "test_accuracy": summary.test_accuracy,
                "test_macro_f1": summary.test_macro_f1,
                "test_class_f1": summary.test_class_f1,
                "test_invalid_rate": summary.test_invalid_rate,
            },
        )
        write_json(
            self.output_root / "analysis" / "inference_deployment.json",
            {
                "run_tag": summary.run_tag,
                "method_name": summary.method_name,
                "inference_examples_per_second": summary.inference_examples_per_second,
                "inference_avg_latency_seconds": summary.inference_avg_latency_seconds,
                "inference_peak_allocated_gb": summary.inference_peak_allocated_gb,
                "inference_peak_reserved_gb": summary.inference_peak_reserved_gb,
                "best_checkpoint_size_bytes": summary.best_checkpoint_size_bytes,
                "final_checkpoint_size_bytes": summary.final_checkpoint_size_bytes,
            },
        )
        write_json(
            self.output_root / "analysis" / "lora_configuration.json",
            {
                "run_tag": summary.run_tag,
                "method_name": summary.method_name,
                "target_modules": list(summary.target_modules),
                "target_layers": list(summary.target_layers),
                "layer_scope": summary.layer_scope,
                "lora_rank": summary.lora_rank,
                "lora_alpha": summary.lora_alpha,
                "lora_dropout": summary.lora_dropout,
                "data_regime": summary.data_regime,
                "data_fraction": summary.data_fraction,
            },
        )
        write_json(
            self.output_root / "analysis" / "update_distribution.json",
            {
                "run_tag": summary.run_tag,
                "method_name": summary.method_name,
                "layerwise_update_dir": str(self.layerwise_dir),
                "checkpoint_timeline": [
                    {
                        "checkpoint_kind": record.checkpoint_kind,
                        "checkpoint_percent": record.checkpoint_percent,
                        "epoch": record.epoch,
                        "global_step": record.global_step,
                        "layerwise_jsonl": str(
                            self.layerwise_dir
                            / (
                                f"{record.checkpoint_kind}_pct_{int(round(record.checkpoint_percent)):03d}_"
                                f"epoch_{record.epoch:03d}_step_{record.global_step:06d}.jsonl"
                            )
                        ),
                        "layerwise_summary_json": str(
                            self.layerwise_dir
                            / (
                                f"{record.checkpoint_kind}_pct_{int(round(record.checkpoint_percent)):03d}_"
                                f"epoch_{record.epoch:03d}_step_{record.global_step:06d}_summary.json"
                            )
                        ),
                    }
                    for record in checkpoint_records
                ],
            },
        )
        write_json(
            self.output_root / "analysis" / "prediction_transitions.json",
            {
                "run_tag": summary.run_tag,
                "method_name": summary.method_name,
                "transitions_dir": str(self.transitions_dir),
            },
        )
        write_jsonl(
            self.output_root / "analysis" / "checkpoint_timeline.jsonl",
            (asdict(record) for record in checkpoint_records),
        )
        write_jsonl(
            self.output_root / "analysis" / "optimization_timeline.jsonl",
            (asdict(record) for record in step_logs),
        )

    def _autocast_context(self):
        enabled = _supports_cuda_amp(self.config.dtype, self.device)
        if not enabled:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.config.dtype, enabled=True)

    def _prepare_output_dirs(self) -> None:
        if self.is_main_process:
            self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
            self.logs_dir.mkdir(parents=True, exist_ok=True)
            self.evaluations_dir.mkdir(parents=True, exist_ok=True)
            self.layerwise_dir.mkdir(parents=True, exist_ok=True)
            self.transitions_dir.mkdir(parents=True, exist_ok=True)
        self._barrier()

    def _set_runtime(self) -> None:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
        torch.manual_seed(self.config.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.config.seed)
        torch.set_num_threads(max(1, self.config.cpu_threads))
        try:
            torch.set_num_interop_threads(max(1, min(4, self.config.cpu_threads)))
        except RuntimeError:
            pass

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    @property
    def fsdp_enabled(self) -> bool:
        return self.config.distributed_mode == "fsdp"

    @property
    def ddp_enabled(self) -> bool:
        return self.config.distributed_mode == "ddp"

    def _initialize_runtime_topology(self) -> None:
        if self.config.distributed_mode == "single":
            self.rank = 0
            self.local_rank = 0
            self.world_size = 1
            return

        required = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
        missing = [name for name in required if name not in os.environ]
        if missing:
            raise RuntimeError(
                f"{self.config.distributed_mode.upper()} mode must be launched with torchrun; "
                f"missing environment variables: {', '.join(missing)}"
            )
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError(f"{self.config.distributed_mode.upper()} mode requires CUDA.")
        self.rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device(f"cuda:{self.local_rank}")
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
            self._owns_process_group = True

    def close(self) -> None:
        if self._owns_process_group and dist.is_initialized():
            dist.destroy_process_group()

    def _barrier(self) -> None:
        if dist.is_initialized():
            dist.barrier()

    def _all_reduce_int(self, value: int) -> int:
        if not dist.is_initialized():
            return value
        tensor = torch.tensor(value, device=self.device, dtype=torch.long)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return int(tensor.item())

    def _distributed_metadata(self) -> dict[str, Any]:
        return {
            "mode": self.config.distributed_mode,
            "world_size": self.world_size,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "sharding_strategy": (
                "FULL_SHARD" if self.fsdp_enabled else None
            ),
            "fsdp_cpu_offload": (
                self.config.fsdp_cpu_offload if self.fsdp_enabled else None
            ),
            "fsdp_auto_wrap_layer_classes": [],
        }

    def _collect_distributed_runtime(
        self,
        *,
        idle_memory: dict[str, float | None],
        loaded_memory: dict[str, float | None],
        peak_train_memory: dict[str, float | None],
    ) -> dict[str, Any]:
        local_record = {
            "rank": self.rank,
            "local_rank": self.local_rank,
            "device": str(self.device),
            "gpu_name": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
            "idle_memory": idle_memory,
            "model_loaded_memory": loaded_memory,
            "peak_training_memory": peak_train_memory,
        }
        records = [local_record]
        return {
            **self._distributed_metadata(),
            "per_rank": records,
            "max_peak_allocated_gb": max(
                (record["peak_training_memory"]["max_allocated_gb"] or 0.0 for record in records),
                default=0.0,
            ),
            "max_peak_reserved_gb": max(
                (record["peak_training_memory"]["max_reserved_gb"] or 0.0 for record in records),
                default=0.0,
            ),
        }

    def wrap_model_for_training(self, model: torch.nn.Module) -> torch.nn.Module:
        if self.ddp_enabled:
            return DistributedDataParallel(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                broadcast_buffers=False,
            )
        if self.fsdp_enabled:
            if (
                FullyShardedDataParallel is None
                or MixedPrecision is None
                or CPUOffload is None
                or ShardingStrategy is None
            ):
                raise RuntimeError("FSDP is not available in this torch build.")
            mixed_precision = None
            if self.config.dtype in {torch.float16, torch.bfloat16}:
                mixed_precision = MixedPrecision(
                    param_dtype=self.config.dtype,
                    reduce_dtype=self.config.dtype,
                    buffer_dtype=self.config.dtype,
                )
            return FullyShardedDataParallel(
                model,
                device_id=self.local_rank,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                cpu_offload=CPUOffload(offload_params=self.config.fsdp_cpu_offload),
                mixed_precision=mixed_precision,
                use_orig_params=True,
            )
        return model

    @staticmethod
    def _is_fsdp_model(model: torch.nn.Module) -> bool:
        return (
            FullyShardedDataParallel is not None
            and isinstance(model, FullyShardedDataParallel)
        )

    @staticmethod
    def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
        if isinstance(model, DistributedDataParallel):
            return model.module
        return model

    @staticmethod
    def _limit_examples(
        examples: list[PubMedQAExample],
        max_examples: int | None,
    ) -> list[PubMedQAExample]:
        if max_examples is None:
            return examples
        return examples[:max_examples]

    @staticmethod
    def _is_better_checkpoint(candidate: CheckpointRecord, current_best: CheckpointRecord) -> bool:
        if candidate.validation_macro_f1 != current_best.validation_macro_f1:
            return candidate.validation_macro_f1 > current_best.validation_macro_f1
        if candidate.validation_accuracy != current_best.validation_accuracy:
            return candidate.validation_accuracy > current_best.validation_accuracy
        return candidate.validation_loss < current_best.validation_loss

    def _move_batch_to_device(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: value.to(self.device) for key, value in batch.items()}


class PubMedQAFullFineTuner(PubMedQATrainingEngine):
    """Full-parameter training strategy using the shared engine unchanged."""



def main() -> None:
    cli_config = FullFineTuneCliConfig.from_env()
    trainer = PubMedQAFullFineTuner(cli_config.config, cli_config.environment)
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
