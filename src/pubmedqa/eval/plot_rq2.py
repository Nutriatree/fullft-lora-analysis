"""Create the RQ2 LoRA test-performance versus parameter-efficiency figure."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from pubmedqa.eval.plotting import apply_report_style, save_figure
from pubmedqa.eval.reports import RunArtifacts, StudyArtifactReader

RUNS = {
    "L3": ("lora-r4", "Q,V", 4),
    "L1": ("lora", "Q,V", 8),
    "L4": ("lora-r16", "Q,V", 16),
    "L2": ("lora-qkvo", "Q,K,V,O", 8),
}


def load_run(study_dir: Path, directory: str) -> dict:
    return RunArtifacts(study_dir / "Qwen_Qwen3-1.7B" / directory).summary()


def generate_rq2_tradeoff_figure(
    study_dir: Path,
    output_dir: Path,
    *,
    dpi: int = 300,
) -> tuple[Path, Path]:
    StudyArtifactReader(study_dir).require_run_directories(
        "Qwen_Qwen3-1.7B", tuple(directory for directory, _, _ in RUNS.values())
    )
    values = {
        tag: load_run(study_dir, directory) for tag, (directory, _, _) in RUNS.items()
    }
    apply_report_style(dpi=dpi)
    figure, axis = plt.subplots(figsize=(7.5, 5.2), constrained_layout=True)

    # Rank sweep: L3 -> L1 -> L4.  A single line makes the capacity trend
    # readable without implying that L2 belongs to the same rank-only sweep.
    rank_tags = ("L3", "L1", "L4")
    rank_x = [100 * values[tag]["trainable_ratio"] for tag in rank_tags]
    rank_y = [values[tag]["test_macro_f1"] for tag in rank_tags]
    axis.plot(
        rank_x, rank_y, color="#1F77B4", linewidth=1.8, zorder=2, label="Q,V rank sweep"
    )
    for tag in rank_tags:
        _, modules, rank = RUNS[tag]
        summary = values[tag]
        x = 100 * summary["trainable_ratio"]
        y = summary["test_macro_f1"]
        size = 105 + 22 * rank
        axis.scatter(
            x, y, s=size, color="#1F77B4", edgecolor="white", linewidth=1.2, zorder=3
        )
        axis.annotate(
            f"{tag}: {modules}, r={rank}\n{summary['trainable_params'] / 1e6:.2f}M",
            (x, y),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            fontsize=8.5,
        )

    # Target-module comparison: same rank as L1, but a different module set.
    l2 = values["L2"]
    l2_x = 100 * l2["trainable_ratio"]
    l2_y = l2["test_macro_f1"]
    axis.scatter(
        l2_x,
        l2_y,
        marker="s",
        s=225,
        color="#F58518",
        edgecolor="white",
        linewidth=1.2,
        zorder=4,
        label="Q,K,V,O expansion (r=8)",
    )
    axis.annotate(
        f"L2: Q,K,V,O, r=8\n{l2['trainable_params'] / 1e6:.2f}M",
        (l2_x, l2_y),
        xytext=(10, -33),
        textcoords="offset points",
        ha="left",
        fontsize=8.5,
    )

    axis.set_xscale("log")
    axis.set_xlabel("Trainable parameters (% of 1.72B total; log scale)")
    axis.set_ylabel("PQA-L test Macro-F1")
    axis.set_title("LoRA Configuration: Test Performance–Parameter Trade-off")
    axis.set_xlim(0.035, 0.24)
    axis.set_ylim(0.498, 0.515)
    axis.grid(alpha=0.25, linewidth=0.7)
    axis.legend(frameon=False, loc="lower right")

    return save_figure(
        figure,
        output_dir,
        "figure_rq2_performance_parameter_tradeoff",
        dpi=dpi,
    )
