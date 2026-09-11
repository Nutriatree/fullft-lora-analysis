"""Create appendix figures for layer-wise and temporal adaptation dynamics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pubmedqa.eval.plotting import apply_report_style, save_figure
from pubmedqa.eval.reports import (
    StudyArtifactReader,
    StudyLayout,
    write_report_manifest,
)

CHECKPOINTS = (25, 50, 75, 100)
RANK_RUNS = (
    ("L3", "lora-r4", 4, "#9467BD"),
    ("L1", "lora", 8, "#1F77B4"),
    ("L4", "lora-r16", 16, "#FF7F0E"),
)


def read_jsonl(file_path: Path) -> list[dict]:
    return StudyArtifactReader(file_path.parent).read_jsonl(file_path.name)


def read_json(file_path: Path) -> dict:
    return StudyArtifactReader(file_path.parent).read_json(file_path.name)


def final_update_file(run_dir: Path) -> Path:
    return (
        run_dir / "layerwise_updates" / "scheduled_pct_100_epoch_001_step_001250.jsonl"
    )


def checkpoint_update_file(run_dir: Path, percent: int) -> Path:
    steps = {25: 313, 50: 625, 75: 938, 100: 1250}
    return (
        run_dir
        / "layerwise_updates"
        / f"scheduled_pct_{percent:03d}_epoch_001_step_{steps[percent]:06d}.jsonl"
    )


def save(
    figure: plt.Figure, output_dir: Path, name: str, dpi: int
) -> tuple[Path, Path]:
    return save_figure(figure, output_dir, name, dpi=dpi)


def draw_a1_selection(lora_dir: Path, output_dir: Path, dpi: int) -> None:
    high = read_json(lora_dir / "analysis" / "selective_layers_high.json")
    low = read_json(lora_dir / "analysis" / "selective_layers_low.json")
    scores = {int(layer): value for layer, value in high["layer_scores"].items()}
    layers = np.asarray(sorted(scores))
    colors = [
        "#2CA02C"
        if layer in high["target_layers"]
        else "#F58518"
        if layer in low["target_layers"]
        else "#BDBDBD"
        for layer in layers
    ]
    figure, axis = plt.subplots(figsize=(9.2, 4.5), constrained_layout=True)
    axis.bar(layers, [scores[layer] for layer in layers], color=colors, width=0.8)
    axis.set_xlabel("Transformer layer index")
    axis.set_ylabel("Sum of relative LoRA update norm (Q + V)")
    axis.set_title("Layer-wise LoRA Update Scores for Selective-Layer Selection")
    axis.set_xticks(layers)
    axis.grid(axis="y", alpha=0.22)
    axis.legend(
        handles=[
            plt.Rectangle(
                (0, 0), 1, 1, color="#2CA02C", label="LL1 high-update layers"
            ),
            plt.Rectangle((0, 0), 1, 1, color="#F58518", label="LL2 low-update layers"),
            plt.Rectangle((0, 0), 1, 1, color="#BDBDBD", label="Other layers"),
        ],
        frameon=False,
    )
    save(figure, output_dir, "appendix_a1_layer_selection_profile", dpi)


def matrix(records: list[dict], modules: list[str]) -> np.ndarray:
    result = np.zeros((28, len(modules)))
    positions = {module: index for index, module in enumerate(modules)}
    for record in records:
        module = record["module_name"]
        if module in positions:
            result[record["layer_index"], positions[module]] = record[
                "relative_update_norm"
            ]
    return result


def draw_a2_final_heatmaps(
    full_dir: Path, lora_dir: Path, output_dir: Path, dpi: int
) -> None:
    full_modules = ["Q", "K", "V", "O", "gate", "up", "down"]
    lora_modules = ["Q", "V"]
    full_values = matrix(read_jsonl(final_update_file(full_dir)), full_modules)
    lora_values = matrix(read_jsonl(final_update_file(lora_dir)), lora_modules)
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 7.0), constrained_layout=True)
    for axis, values, modules, title in (
        (axes[0], full_values, full_modules, "Full Fine-Tuning"),
        (axes[1], lora_values, lora_modules, "LoRA (L1)"),
    ):
        image = axis.imshow(values, aspect="auto", origin="lower", cmap="magma")
        axis.set_title(title)
        axis.set_xlabel("Component")
        axis.set_ylabel("Transformer layer index")
        axis.set_xticks(range(len(modules)), modules)
        axis.set_yticks(range(0, 28, 3))
        figure.colorbar(image, ax=axis, shrink=0.82, label="Relative update norm")
    figure.suptitle("Final Layer-wise and Component-wise Relative Updates", y=1.02)
    save(figure, output_dir, "appendix_a2_final_update_heatmaps", dpi)


def draw_a3_temporal_heatmaps(lora_dir: Path, output_dir: Path, dpi: int) -> None:
    values = {module: [] for module in ("Q", "V")}
    for percent in CHECKPOINTS:
        records = read_jsonl(checkpoint_update_file(lora_dir, percent))
        for module in values:
            layer_values = np.zeros(28)
            for record in records:
                if record["module_name"] == module:
                    layer_values[record["layer_index"]] = record["relative_update_norm"]
            values[module].append(layer_values)
    figure, axes = plt.subplots(
        1, 2, figsize=(11, 4.2), sharey=True, constrained_layout=True
    )
    for axis, module in zip(axes, ("Q", "V")):
        image = axis.imshow(np.asarray(values[module]), aspect="auto", cmap="viridis")
        axis.set_title(f"{module} projection")
        axis.set_xlabel("Transformer layer index")
        axis.set_xticks(range(0, 28, 3))
        axis.set_yticks(
            range(len(CHECKPOINTS)), [f"{percent}%" for percent in CHECKPOINTS]
        )
        figure.colorbar(image, ax=axis, shrink=0.86, label="Relative ΔW norm")
    axes[0].set_ylabel("Training progress")
    figure.suptitle("Temporal Evolution of Layer-wise LoRA Updates", y=1.02)
    save(figure, output_dir, "appendix_a3_temporal_layer_updates", dpi)


def draw_a4_update_dynamics(lora_dir: Path, output_dir: Path, dpi: int) -> None:
    shares = read_json(lora_dir / "analysis" / "module_update_share.json")[
        "checkpoint_timeline"
    ]
    performance = read_json(lora_dir / "analysis" / "performance_dynamics.json")[
        "checkpoint_timeline"
    ]
    progress = [entry["checkpoint_percent"] for entry in shares]
    q_cumulative = [
        entry["by_module"]["Q"]["sum_delta_w_relative_norm"] for entry in shares
    ]
    v_cumulative = [
        entry["by_module"]["V"]["sum_delta_w_relative_norm"] for entry in shares
    ]
    q_share = [
        entry["by_module"]["Q"]["sum_cumulative_update_share"] for entry in shares
    ]
    v_share = [
        entry["by_module"]["V"]["sum_cumulative_update_share"] for entry in shares
    ]
    f1 = [entry["validation_macro_f1"] for entry in performance]
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), constrained_layout=True)
    axes[0].plot(progress, q_cumulative, marker="o", color="#1F77B4", label="Q")
    axes[0].plot(progress, v_cumulative, marker="o", color="#FF7F0E", label="V")
    axes[0].set_xlabel("Training progress (%)")
    axes[0].set_ylabel("Sum of relative ΔW norms")
    axes[0].set_title("Cumulative LoRA update magnitude")
    axes[0].grid(alpha=0.22)
    axes[0].legend(frameon=False)
    axes[1].plot(progress, q_share, marker="o", color="#1F77B4", label="Q update share")
    axes[1].plot(progress, v_share, marker="o", color="#FF7F0E", label="V update share")
    f1_axis = axes[1].twinx()
    f1_axis.plot(
        progress,
        f1,
        marker="s",
        linestyle="--",
        color="#333333",
        label="Validation Macro-F1",
    )
    axes[1].set_xlabel("Training progress (%)")
    axes[1].set_ylabel("Cumulative update share")
    f1_axis.set_ylabel("Validation Macro-F1 (yes/no)")
    axes[1].set_title("Module update share and validation performance")
    axes[1].grid(alpha=0.22)
    h1, l1 = axes[1].get_legend_handles_labels()
    h2, l2 = f1_axis.get_legend_handles_labels()
    axes[1].legend(h1 + h2, l1 + l2, frameon=False, loc="center right", fontsize=8)
    figure.suptitle(
        "LoRA Update Growth, Module Allocation, and Validation Performance", y=1.02
    )
    save(figure, output_dir, "appendix_a4_lora_update_dynamics", dpi)


def draw_a5_effective_rank(study_dir: Path, output_dir: Path, dpi: int) -> None:
    figure, axis = plt.subplots(figsize=(8.2, 4.7), constrained_layout=True)
    for tag, directory, rank, color in RANK_RUNS:
        timeline = read_json(
            study_dir
            / "Qwen_Qwen3-1.7B"
            / directory
            / "analysis"
            / "lora_effective_rank.json"
        )["checkpoint_timeline"]
        progress = [entry["checkpoint_percent"] for entry in timeline]
        means = [
            np.mean([record["effective_rank"] for record in entry["records"]])
            for entry in timeline
        ]
        stds = [
            np.std([record["effective_rank"] for record in entry["records"]])
            for entry in timeline
        ]
        axis.plot(
            progress,
            means,
            marker="o",
            color=color,
            label=f"{tag}: configured r={rank}",
        )
        axis.fill_between(
            progress,
            np.asarray(means) - np.asarray(stds),
            np.asarray(means) + np.asarray(stds),
            color=color,
            alpha=0.13,
        )
    axis.set_xlabel("Training progress (%)")
    axis.set_ylabel("Mean effective rank across Q/V adapters")
    axis.set_title("Effective Rank Utilization over Training")
    axis.set_xticks((0, *CHECKPOINTS))
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    save(figure, output_dir, "appendix_a5_effective_rank_trajectory", dpi)


def draw_a6_singular_spectrum(study_dir: Path, output_dir: Path, dpi: int) -> None:
    figure, axis = plt.subplots(figsize=(8.2, 4.7), constrained_layout=True)
    for tag, directory, rank, color in RANK_RUNS:
        timeline = read_json(
            study_dir
            / "Qwen_Qwen3-1.7B"
            / directory
            / "analysis"
            / "lora_singular_values.json"
        )["checkpoint_timeline"]
        records = timeline[-1]["records"]
        singular_values = np.asarray(
            [record["singular_values"][:rank] for record in records], dtype=float
        )
        mean_values = singular_values.mean(axis=0)
        normalized = (
            mean_values / mean_values.sum() if mean_values.sum() > 0 else mean_values
        )
        axis.plot(
            range(1, rank + 1),
            normalized,
            marker="o",
            color=color,
            label=f"{tag}: r={rank}",
        )
    axis.set_xlabel("Singular component index")
    axis.set_ylabel("Mean normalized singular value")
    axis.set_title("Final LoRA Singular-Value Spectra")
    axis.set_xticks(range(1, 17))
    axis.grid(alpha=0.22)
    axis.legend(frameon=False)
    save(figure, output_dir, "appendix_a6_singular_value_spectra", dpi)


def generate_appendix_figures(
    study_dir: Path,
    output_dir: Path,
    *,
    dpi: int = 300,
) -> None:
    StudyArtifactReader(study_dir).require_run_directories(
        "Qwen_Qwen3-1.7B", ("full-ft", "lora", "lora-r4", "lora-r16")
    )
    apply_report_style(dpi=dpi, title_size=11)
    model_dir = study_dir / "Qwen_Qwen3-1.7B"
    draw_a1_selection(model_dir / "lora", output_dir, dpi)
    draw_a2_final_heatmaps(model_dir / "full-ft", model_dir / "lora", output_dir, dpi)
    draw_a3_temporal_heatmaps(model_dir / "lora", output_dir, dpi)
    draw_a4_update_dynamics(model_dir / "lora", output_dir, dpi)
    draw_a5_effective_rank(study_dir, output_dir, dpi)
    draw_a6_singular_spectrum(study_dir, output_dir, dpi)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--study-dir",
        type=Path,
        default=Path("outputs/pubmedqa_train/study_single_epoch1_used"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layout = StudyLayout.from_study_dir(args.study_dir)
    generate_appendix_figures(
        args.study_dir,
        args.output_dir or layout.appendix_figures_dir,
        dpi=args.dpi,
    )
    print(f"Updated {write_report_manifest(layout)}")


if __name__ == "__main__":
    main()
