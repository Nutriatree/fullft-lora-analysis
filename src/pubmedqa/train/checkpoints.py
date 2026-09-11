"""Checkpoint cadence, save/reload and artifact records."""

from __future__ import annotations

import math
import shutil
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from pubmedqa.config.train import normalize_checkpoint_percents
from pubmedqa.data.records import (
    count_directory_size_bytes as _count_directory_size_bytes,
)
from pubmedqa.data.records import (
    current_time_iso,
    safe_name,
    write_json,
)
from pubmedqa.eval.validation import EvalMetrics
from pubmedqa.model.device import (
    memory_snapshot as _memory_snapshot,
)
from pubmedqa.train.distributed import (
    FullStateDictConfig,
    FullyShardedDataParallel,
    StateDictType,
    SynchronizedOperationError,
    _unwrap_model,
)

if TYPE_CHECKING:
    from pubmedqa.train.analysis import LayerwiseReference


@dataclass(frozen=True)
class CheckpointRecord:
    checkpoint_kind: str
    checkpoint_percent: float
    epoch: int
    step_in_epoch: int
    global_step: int
    elapsed_seconds: float
    checkpoint_dir: str
    checkpoint_size_bytes: int
    train_loss: float
    validation_loss: float
    validation_accuracy: float
    validation_macro_f1: float
    validation_class_f1: dict[str, float]
    validation_invalid_rate: float
    learning_rate: float
    gradient_norm: float
    peak_allocated_gb: float | None
    peak_reserved_gb: float | None


def checkpoint_directory(
    checkpoints_dir: Path,
    *,
    checkpoint_kind: str,
    checkpoint_percent: float,
    epoch: int,
    global_step: int,
) -> Path:
    """Return the stable on-disk directory name for a scheduled checkpoint."""

    return checkpoints_dir / (
        f"{checkpoint_kind}_pct_{int(round(checkpoint_percent)):03d}_"
        f"epoch_{epoch:03d}_step_{global_step:06d}"
    )


def write_training_state(
    checkpoint_dir: Path,
    *,
    checkpoint_kind: str,
    checkpoint_percent: float,
    epoch: int,
    step_in_epoch: int,
    global_step: int,
    elapsed_seconds: float,
    validation_metrics: EvalMetrics,
    title: str,
    run_id: str,
    run_tag: str,
    model_name: str,
    condition: str,
    distributed: dict[str, Any],
) -> None:
    """Persist the strategy-independent training-state artifact schema."""

    write_json(
        checkpoint_dir / "training_state.json",
        {
            "checkpoint_kind": checkpoint_kind,
            "checkpoint_percent": checkpoint_percent,
            "epoch": epoch,
            "step_in_epoch": step_in_epoch,
            "global_step": global_step,
            "elapsed_seconds": elapsed_seconds,
            "validation_metrics": asdict(validation_metrics),
            "title": title,
            "run_id": run_id,
            "run_tag": run_tag,
            "model_name": model_name,
            "condition": condition,
            "distributed": distributed,
            "saved_at": current_time_iso(),
        },
    )


def save_model_files_to_directory(
    *,
    session,
    model: torch.nn.Module,
    tokenizer: Any,
    checkpoint_dir: Path,
) -> None:
    model_to_save = _unwrap_model(model)
    state_dict = None
    if session.fsdp_enabled:
        if (
            FullyShardedDataParallel is None
            or FullStateDictConfig is None
            or StateDictType is None
        ):
            raise RuntimeError(
                "FSDP checkpoint saving is not available in this torch build."
            )
        # rank0_only controls the returned CPU dictionary, not participation.
        # Collective failures here are fatal, not recoverable filesystem errors.
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FullyShardedDataParallel.state_dict_type(
            model_to_save,
            StateDictType.FULL_STATE_DICT,
            save_policy,
        ):
            state_dict = model_to_save.state_dict()

    def write_files():
        if session.fsdp_enabled:
            model_to_save.save_pretrained(checkpoint_dir, state_dict=state_dict)
        else:
            model_to_save.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)

    session.run_on_main_process(
        write_files, operation_name="checkpoint model/tokenizer write"
    )


def create_evaluation_snapshot(
    *,
    files,
    session,
    model: torch.nn.Module,
    tokenizer: Any,
    split_name: str,
) -> Path:
    snapshot_dir = files.output_root / ".evaluation_snapshots" / safe_name(split_name)
    try:
        session.run_on_main_process(
            lambda: snapshot_dir.mkdir(parents=True, exist_ok=True),
            operation_name="snapshot mkdir",
        )
        save_model_files_to_directory(
            session=session,
            model=model,
            tokenizer=tokenizer,
            checkpoint_dir=snapshot_dir,
        )
    except Exception as exc:
        if session.control_group is None or isinstance(exc, SynchronizedOperationError):
            remove_evaluation_snapshot(snapshot_dir, session=session)
        elif session.is_main_process:
            # A failed tensor collective is not safe for another Gloo handshake.
            shutil.rmtree(snapshot_dir, ignore_errors=True)
        raise
    return snapshot_dir


def remove_evaluation_snapshot(snapshot_dir: Path, *, session) -> None:
    """Remove only the snapshot supplied by the owning run's evaluation path."""

    def remove():
        if session.is_main_process and snapshot_dir.exists():
            shutil.rmtree(snapshot_dir)

    if session.control_group is not None:
        session.run_on_main_process(remove, operation_name="snapshot cleanup")
    else:
        remove()


def save_checkpoint(
    *,
    files,
    save_optimizer_state,
    session,
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
    checkpoint_dir = checkpoint_directory(
        files.checkpoints_dir,
        checkpoint_kind=checkpoint_kind,
        checkpoint_percent=checkpoint_percent,
        epoch=epoch,
        global_step=global_step,
    )
    session.run_on_main_process(
        lambda: checkpoint_dir.mkdir(parents=True, exist_ok=True),
        operation_name="checkpoint mkdir",
    )
    if save_model_files:
        save_model_files_to_directory(
            session=session,
            model=model,
            tokenizer=tokenizer,
            checkpoint_dir=checkpoint_dir,
        )

    def write_metadata():
        if save_model_files and save_optimizer_state:
            torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
            torch.save(scheduler.state_dict(), checkpoint_dir / "scheduler.pt")
        write_training_state(
            checkpoint_dir,
            checkpoint_kind=checkpoint_kind,
            checkpoint_percent=checkpoint_percent,
            epoch=epoch,
            step_in_epoch=step_in_epoch,
            global_step=global_step,
            elapsed_seconds=elapsed_seconds,
            validation_metrics=validation_metrics,
            title=files.title,
            run_id=files.run_id,
            run_tag=files.run_tag,
            model_name=files.model_name,
            condition=files.condition,
            distributed=session.metadata(),
        )

    session.run_on_main_process(
        write_metadata, operation_name="checkpoint metadata write"
    )
    return checkpoint_dir


def record_checkpoint(
    *,
    save_checkpoint,
    write_updates,
    session,
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
    checkpoint_dir = save_checkpoint(
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
    checkpoint_size = session.run_on_main_process(
        lambda: _count_directory_size_bytes(checkpoint_dir),
        operation_name="checkpoint size",
    )
    peak_memory = _memory_snapshot(session.device)
    write_updates(
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


def _is_better_checkpoint(
    candidate: CheckpointRecord, current_best: CheckpointRecord
) -> bool:
    if candidate.validation_macro_f1 != current_best.validation_macro_f1:
        return candidate.validation_macro_f1 > current_best.validation_macro_f1
    if candidate.validation_accuracy != current_best.validation_accuracy:
        return candidate.validation_accuracy > current_best.validation_accuracy
    return candidate.validation_loss < current_best.validation_loss


def save_lora_checkpoint(
    *,
    files,
    save_optimizer_state,
    adapter,
    session,
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
    checkpoint_dir = files.checkpoints_dir / (
        f"{checkpoint_kind}_pct_{int(round(checkpoint_percent)):03d}_"
        f"epoch_{epoch:03d}_step_{global_step:06d}"
    )
    session.run_on_main_process(
        lambda: checkpoint_dir.mkdir(parents=True, exist_ok=True),
        operation_name="LoRA checkpoint mkdir",
    )
    if save_model_files:
        save_model_files_to_directory(
            session=session,
            model=model,
            tokenizer=tokenizer,
            checkpoint_dir=checkpoint_dir,
        )
    checkpoint_id = checkpoint_dir.name

    def write_metadata():
        if save_model_files and save_optimizer_state:
            torch.save(optimizer.state_dict(), checkpoint_dir / "optimizer.pt")
            torch.save(scheduler.state_dict(), checkpoint_dir / "scheduler.pt")
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
                "title": files.title,
                "run_id": files.run_id,
                "run_tag": files.run_tag,
                "model_name": files.model_name,
                "condition": files.condition,
                "distributed": session.metadata(),
                "lora": {
                    "target_modules": list(adapter.target_modules),
                    "target_layers": list(adapter.target_layers),
                    "layer_scope": adapter.layer_scope,
                    "rank": adapter.lora_rank,
                    "alpha": adapter.lora_alpha,
                    "dropout": adapter.lora_dropout,
                    "bias": adapter.lora_bias,
                    "task_type": adapter.lora_task_type,
                    "modules_to_save": list(adapter.modules_to_save),
                    "merge_for_eval": adapter.merge_for_eval,
                },
                "saved_at": current_time_iso(),
            },
        )

    session.run_on_main_process(
        write_metadata, operation_name="LoRA checkpoint metadata write"
    )
    return checkpoint_dir


def build_checkpoint_schedule(
    total_steps: int,
    percents: Sequence[int],
) -> list[tuple[int, int]]:
    schedule: list[tuple[int, int]] = []
    seen_steps: set[int] = set()
    for percent in normalize_checkpoint_percents(percents):
        target_step = max(1, math.ceil(total_steps * (percent / 100.0)))
        if target_step in seen_steps:
            continue
        seen_steps.add(target_step)
        schedule.append((percent, target_step))
    return schedule
