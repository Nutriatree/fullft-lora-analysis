"""Create the main-text RQ3 selective-LoRA performance-efficiency figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pubmedqa.eval.plotting import annotate_bars, apply_report_style, save_figure
from pubmedqa.eval.reports import RunArtifacts, StudyArtifactReader

RUNS = (
    ("L1", "lora", "All layers", "#1F77B4"),
    ("LL1", "selective-lora-high-update", "High-update\nlayers", "#2CA02C"),
    ("LL2", "selective-lora-low-update", "Low-update\nlayers", "#F58518"),
)


def load_summaries(study_dir: Path) -> list[dict]:
    model_dir = study_dir / "Qwen_Qwen3-1.7B"
    summaries = []
    for tag, directory, display_name, color in RUNS:
        summary = RunArtifacts(model_dir / directory).summary()
        summaries.append(
            {
                "tag": tag,
                "name": display_name,
                "color": color,
                **summary,
            }
        )
    return summaries


def generate_rq3_selective_figure(
    study_dir: Path,
    output_dir: Path,
    *,
    dpi: int = 300,
) -> tuple[Path, Path]:
    StudyArtifactReader(study_dir).require_run_directories(
        "Qwen_Qwen3-1.7B", tuple(directory for _, directory, _, _ in RUNS)
    )
    summaries = load_summaries(study_dir)
    apply_report_style(dpi=dpi)
    figure, (performance_axis, parameter_axis) = plt.subplots(
        1, 2, figsize=(11.5, 4.5), constrained_layout=True
    )
    positions = np.arange(len(summaries))
    colors = [summary["color"] for summary in summaries]
    width = 0.36

    accuracy_bars = performance_axis.bar(
        positions - width / 2,
        [summary["test_accuracy"] for summary in summaries],
        width,
        label="Accuracy",
        color=colors,
    )
    macro_bars = performance_axis.bar(
        positions + width / 2,
        [summary["test_macro_f1"] for summary in summaries],
        width,
        label="Macro-F1",
        color=colors,
        alpha=0.42,
        hatch="//",
    )
    annotate_bars(performance_axis, accuracy_bars, lambda value: f"{value:.3f}")
    annotate_bars(performance_axis, macro_bars, lambda value: f"{value:.3f}")
    performance_axis.set_xticks(positions, [summary["tag"] for summary in summaries])
    performance_axis.set_ylim(0, 0.82)
    performance_axis.set_ylabel("PQA-L test score")
    performance_axis.set_title("(a) Final test performance")
    performance_axis.legend(frameon=False)
    performance_axis.grid(axis="y", alpha=0.22, linewidth=0.7)

    parameter_bars = parameter_axis.bar(
        positions,
        [summary["trainable_params"] for summary in summaries],
        color=colors,
    )
    parameter_axis.set_yscale("log")
    parameter_axis.set_xticks(positions, [summary["tag"] for summary in summaries])
    parameter_axis.set_ylabel("Trainable parameters (log scale)")
    parameter_axis.set_title("(b) Parameter efficiency")
    parameter_axis.grid(axis="y", alpha=0.22, linewidth=0.7)
    for bar, summary in zip(parameter_bars, summaries):
        parameter_axis.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.14,
            f"{summary['trainable_params'] / 1e6:.2f}M\n({100 * summary['trainable_ratio']:.4f}%)",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    figure.suptitle(
        "Selective-Layer LoRA: Test Performance and Parameter Efficiency", y=1.03
    )
    return save_figure(
        figure,
        output_dir,
        "figure_rq3_selective_performance_parameter",
        dpi=dpi,
    )
