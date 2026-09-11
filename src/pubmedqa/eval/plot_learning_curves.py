"""Draw RQ-specific train/validation loss learning curves for PubMedQA."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pubmedqa.eval.plotting import apply_report_style, moving_average, save_figure
from pubmedqa.eval.reports import RunArtifacts, StudyArtifactReader

CHECKPOINT_STEPS = (313, 625, 938, 1250)

# Each run has a saturated train-loss colour and a visually paired pastel
# validation-loss colour.  RQ1/RQ2/RQ3 reuse a run's colour consistently.
RUN_STYLES = {
    "full-ft": ("F1: Full FT", "#D62728", "#F4A6A6"),
    "lora": ("L1: LoRA Q,V (r=8)", "#1F77B4", "#9ECAE1"),
    "lora-qkvo": ("L2: LoRA Q,K,V,O (r=8)", "#2CA02C", "#A1D99B"),
    "lora-r4": ("L3: LoRA Q,V (r=4)", "#9467BD", "#D4B9DA"),
    "lora-r16": ("L4: LoRA Q,V (r=16)", "#FF7F0E", "#FDCB9E"),
    "selective-lora-high-update": ("LL1: High-update layers", "#8C564B", "#D7B5AD"),
    "selective-lora-low-update": ("LL2: Low-update layers", "#7F7F7F", "#C7C7C7"),
}

RQ_RUNS = {
    "rq1": ("RQ1: Full Fine-Tuning vs. LoRA", ("full-ft", "lora")),
    "rq2": (
        "RQ2: LoRA target-module and rank configurations",
        ("lora", "lora-qkvo", "lora-r4", "lora-r16"),
    ),
    "rq3": (
        "RQ3: All-layer and selective-layer LoRA",
        ("lora", "selective-lora-high-update", "selective-lora-low-update"),
    ),
}


def load_run(run_dir: Path) -> tuple[list[dict], list[dict]]:
    artifacts = RunArtifacts(run_dir)
    return artifacts.optimization_timeline(), artifacts.performance_timeline()


def draw_rq_figure(
    *,
    model_dir: Path,
    output_dir: Path,
    rq_key: str,
    smooth_window: int,
    dpi: int,
) -> tuple[Path, Path]:
    title, runs = RQ_RUNS[rq_key]
    figure, train_axis = plt.subplots(figsize=(8.2, 5.1), constrained_layout=True)
    validation_axis = train_axis.twinx()

    for run in runs:
        label, train_color, validation_color = RUN_STYLES[run]
        train_timeline, checkpoints = load_run(model_dir / run)
        steps = np.asarray([entry["global_step"] for entry in train_timeline])
        train_losses = np.asarray(
            [entry["train_loss"] for entry in train_timeline], dtype=float
        )
        train_axis.plot(
            steps, train_losses, color=train_color, alpha=0.10, linewidth=0.5
        )
        train_axis.plot(
            steps,
            moving_average(train_losses, smooth_window),
            color=train_color,
            linewidth=2.0,
            label=f"{label} — train",
        )

        validation_steps = np.asarray([entry["global_step"] for entry in checkpoints])
        validation_losses = np.asarray(
            [entry["validation_loss"] for entry in checkpoints], dtype=float
        )
        validation_axis.plot(
            validation_steps,
            validation_losses,
            color=validation_color,
            marker="o",
            markersize=5.5,
            linewidth=2.0,
            linestyle="--",
            label=f"{label} — validation",
        )

    for step in CHECKPOINT_STEPS:
        train_axis.axvline(
            step, color="#bdbdbd", linestyle="--", linewidth=0.7, zorder=0
        )
    train_axis.set_xlim(0, 1250)
    train_axis.set_xlabel("Optimizer step")
    train_axis.set_ylabel(f"Train loss (left; rolling mean, window={smooth_window})")
    validation_axis.set_ylabel("Validation loss (right; checkpoint evaluation)")
    train_axis.set_yscale("log")
    validation_axis.set_yscale("log")
    train_axis.grid(axis="y", alpha=0.22, linewidth=0.7)
    train_axis.set_title(title)

    train_handles, train_labels = train_axis.get_legend_handles_labels()
    validation_handles, validation_labels = validation_axis.get_legend_handles_labels()
    train_axis.legend(
        train_handles + validation_handles,
        train_labels + validation_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=False,
        fontsize=8,
    )

    return save_figure(
        figure,
        output_dir,
        f"figure_{rq_key}_train_validation_loss",
        dpi=dpi,
    )


def generate_rq_learning_curves(
    study_dir: Path,
    output_dir: Path,
    *,
    smooth_window: int = 25,
    dpi: int = 300,
) -> list[Path]:
    reader = StudyArtifactReader(study_dir)
    reader.require_run_directories("Qwen_Qwen3-1.7B", tuple(RUN_STYLES))
    apply_report_style(dpi=dpi)
    model_dir = study_dir / "Qwen_Qwen3-1.7B"
    generated: list[Path] = []
    for rq_key in RQ_RUNS:
        generated.extend(
            draw_rq_figure(
                model_dir=model_dir,
                output_dir=output_dir,
                rq_key=rq_key,
                smooth_window=smooth_window,
                dpi=dpi,
            )
        )
    return generated
