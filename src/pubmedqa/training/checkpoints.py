"""Checkpoint naming and metadata persistence shared by training strategies."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from pubmedqa.runtime.io import current_time_iso, write_json
from pubmedqa.training.contracts import EvalMetrics


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
