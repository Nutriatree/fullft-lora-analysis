"""Create RQ1 Figure 2 from final PQA-L test outputs (B0, F1, and L1)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pubmedqa.eval.metrics import classwise_f1
from pubmedqa.eval.plotting import annotate_bars, apply_report_style, save_figure
from pubmedqa.eval.reports import RunArtifacts, StudyArtifactReader

LABELS = ("yes", "no", "maybe")
RUNS = (
    ("B0", "Baseline", "#7F7F7F"),
    ("F1", "Full FT", "#D62728"),
    ("L1", "LoRA", "#1F77B4"),
)


def load_metrics(
    study_dir: Path,
    baseline_dir: Path,
) -> dict[str, dict[str, float | dict[str, float]]]:
    baseline_reader = StudyArtifactReader(baseline_dir)
    baseline = baseline_reader.read_json("summary.json")
    baseline_predictions = baseline_reader.read_jsonl("outputs.jsonl")
    baseline_class_f1 = classwise_f1(
        [item["gold_label"] for item in baseline_predictions],
        [item["predicted_label"] for item in baseline_predictions],
        labels=LABELS,
    )

    model_dir = study_dir / "Qwen_Qwen3-1.7B"
    metrics: dict[str, dict[str, float | dict[str, float]]] = {
        "B0": {
            "accuracy": baseline["accuracy"],
            "macro_f1": baseline["macro_f1"],
            "class_f1": baseline_class_f1,
        }
    }
    for tag, directory in (("F1", "full-ft"), ("L1", "lora")):
        summary = RunArtifacts(model_dir / directory).evaluation_summary("test")
        metrics[tag] = {
            "accuracy": summary["accuracy"],
            "macro_f1": summary["macro_f1"],
            "class_f1": summary["class_f1"],
        }
    return metrics


def generate_rq1_performance_figure(
    study_dir: Path,
    baseline_dir: Path,
    output_dir: Path,
    *,
    dpi: int = 300,
) -> tuple[Path, Path]:
    StudyArtifactReader(study_dir).require_run_directories(
        "Qwen_Qwen3-1.7B", ("full-ft", "lora")
    )
    metrics = load_metrics(study_dir, baseline_dir)
    apply_report_style(dpi=dpi)
    figure, (overall_axis, class_axis) = plt.subplots(
        1, 2, figsize=(12.5, 4.5), constrained_layout=True
    )

    run_positions = np.arange(len(RUNS))
    width = 0.34
    accuracy_bars = overall_axis.bar(
        run_positions - width / 2,
        [float(metrics[tag]["accuracy"]) for tag, _, _ in RUNS],
        width,
        label="Accuracy",
        color="#4C78A8",
    )
    macro_f1_bars = overall_axis.bar(
        run_positions + width / 2,
        [float(metrics[tag]["macro_f1"]) for tag, _, _ in RUNS],
        width,
        label="Macro-F1",
        color="#F58518",
    )
    annotate_bars(overall_axis, accuracy_bars, lambda value: f"{value:.3f}")
    annotate_bars(overall_axis, macro_f1_bars, lambda value: f"{value:.3f}")
    overall_axis.set_xticks(run_positions, [label for _, label, _ in RUNS])
    overall_axis.set_ylim(0, 0.85)
    overall_axis.set_ylabel("Score")
    overall_axis.set_title("(a) Overall PQA-L test performance")
    overall_axis.legend(frameon=False)
    overall_axis.grid(axis="y", alpha=0.22, linewidth=0.7)

    label_positions = np.arange(len(LABELS))
    width = 0.23
    for index, (tag, label, color) in enumerate(RUNS):
        values = [
            float(metrics[tag]["class_f1"][class_label]) for class_label in LABELS
        ]
        bars = class_axis.bar(
            label_positions + (index - 1) * width,
            values,
            width,
            label=label,
            color=color,
        )
        annotate_bars(class_axis, bars, lambda value: f"{value:.3f}")
    class_axis.set_xticks(label_positions, [item.capitalize() for item in LABELS])
    class_axis.set_ylim(0, 1.0)
    class_axis.set_ylabel("F1-score")
    class_axis.set_title("(b) Class-wise PQA-L test F1")
    class_axis.legend(frameon=False)
    class_axis.grid(axis="y", alpha=0.22, linewidth=0.7)

    return save_figure(figure, output_dir, "figure_rq1_test_performance", dpi=dpi)
