#!/usr/bin/env python3
"""Validate PubMedQA experiment outputs and aggregate key metrics across runs."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from pubmedqa.experiment_runs import (
    DEFAULT_BASELINE_OUTPUT_DIR,
    DEFAULT_SHARED_DEFAULTS,
    DEFAULT_TRAIN_OUTPUT_DIR,
    list_run_tags,
    resolve_run_spec,
)
from pubmedqa.data.records import safe_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs", default="B0,F1,L1,L2,L3,L4,LL1")
    parser.add_argument("--baseline-output-dir", type=Path, default=DEFAULT_BASELINE_OUTPUT_DIR)
    parser.add_argument("--train-output-dir", type=Path, default=DEFAULT_TRAIN_OUTPUT_DIR)
    parser.add_argument("--model-name", default=DEFAULT_SHARED_DEFAULTS.model_name)
    parser.add_argument("--output-file", type=Path, default=None)
    return parser.parse_args()


def _parse_run_tags(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(list_run_tags())
    return [part.strip() for part in raw.split(",") if part.strip()]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _expected_run_dir(
    *,
    run_id: str,
    run_tag: str,
    model_name: str,
    baseline_output_dir: Path,
    train_output_dir: Path,
) -> Path:
    spec = resolve_run_spec(run_tag)
    root = baseline_output_dir if spec.method == "baseline" else train_output_dir
    return root / run_id / safe_name(model_name) / safe_name(spec.condition)


def _expected_files(run_tag: str) -> list[str]:
    spec = resolve_run_spec(run_tag)
    if spec.method == "baseline":
        return [
            "summary.json",
            "outputs.jsonl",
            "run.json",
            "metrics/ACC.json",
            "metrics/Macro_F1.json",
            "metrics/Invalid_rate.json",
        ]

    files = [
        "summary.json",
        "config.json",
        "run_metadata.json",
        "artifacts.json",
        "logs/train_steps.jsonl",
        "logs/checkpoints.jsonl",
        "analysis/parameter_model_size.json",
        "analysis/memory.json",
        "analysis/training_efficiency.json",
        "analysis/optimization.json",
        "analysis/learning_dynamics.json",
        "analysis/performance_dynamics.json",
        "analysis/performance_generalization.json",
        "analysis/inference_deployment.json",
        "analysis/update_distribution.json",
        "analysis/prediction_transitions.json",
        "analysis/checkpoint_timeline.jsonl",
        "analysis/optimization_timeline.jsonl",
    ]
    if spec.method == "lora":
        files.extend(
            [
                "analysis/lora_configuration.json",
                "analysis/lora_dynamics.json",
                "analysis/lora_singular_values.json",
                "analysis/lora_effective_rank.json",
                "analysis/lora_adapter_norms.json",
                "analysis/lora_update_direction.json",
                "analysis/module_update_share.json",
                "analysis/val_update_alignment.json",
                "analysis/lora_adapter_metrics.jsonl",
            ]
        )
    return files


def _compare_shared_config(config: dict[str, Any], reference: dict[str, Any]) -> dict[str, dict[str, Any]]:
    keys = [
        "model_name",
        "num_epochs",
        "train_batch_size",
        "eval_batch_size",
        "gradient_accumulation_steps",
        "learning_rate",
        "weight_decay",
        "warmup_ratio",
        "max_grad_norm",
        "max_input_tokens",
        "max_new_tokens",
        "device",
        "cpu_threads",
        "strict_parser",
        "seed",
    ]
    diff: dict[str, dict[str, Any]] = {}
    for key in keys:
        left = config.get(key)
        right = reference.get(key)
        if left != right:
            diff[key] = {"current": left, "reference": right}
    return diff


def main() -> None:
    args = parse_args()
    run_tags = _parse_run_tags(args.runs)
    model_name = args.model_name
    safe_model = safe_name(model_name)

    results: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    reference_config: dict[str, Any] | None = None

    for run_tag in run_tags:
        spec = resolve_run_spec(run_tag)
        run_dir = _expected_run_dir(
            run_id=args.run_id,
            run_tag=run_tag,
            model_name=model_name,
            baseline_output_dir=args.baseline_output_dir,
            train_output_dir=args.train_output_dir,
        )
        missing = [path for path in _expected_files(run_tag) if not (run_dir / path).exists()]
        entry: dict[str, Any] = {
            "run_tag": run_tag,
            "method": spec.method,
            "condition": spec.condition,
            "run_dir": str(run_dir),
            "exists": run_dir.exists(),
            "missing_files": missing,
            "valid": run_dir.exists() and not missing,
        }

        if (run_dir / "summary.json").is_file():
            summary = _load_json(run_dir / "summary.json")
            entry["summary"] = summary
            comparison_rows.append(
                {
                    "run_tag": run_tag,
                    "method": spec.method,
                    "condition": spec.condition,
                    "accuracy": summary.get("accuracy", summary.get("best_validation_accuracy")),
                    "macro_f1": summary.get("macro_f1", summary.get("best_validation_macro_f1")),
                    "trainable_params": summary.get("trainable_params"),
                    "trainable_ratio": summary.get("trainable_ratio"),
                    "peak_train_allocated_gb": summary.get("peak_train_allocated_gb"),
                    "total_training_time_seconds": summary.get("total_training_time_seconds", summary.get("elapsed_seconds")),
                    "best_checkpoint_size_bytes": summary.get("best_checkpoint_size_bytes"),
                    "inference_examples_per_second": summary.get("inference_examples_per_second", summary.get("examples_per_second")),
                    "inference_avg_latency_seconds": summary.get("inference_avg_latency_seconds"),
                    "invalid_rate": summary.get("invalid_rate", summary.get("test_invalid_rate")),
                }
            )

        if spec.method != "baseline" and (run_dir / "config.json").is_file():
            config = _load_json(run_dir / "config.json")
            entry["config"] = config
            if reference_config is None and run_tag == "F1":
                reference_config = config
            elif reference_config is not None:
                entry["shared_config_diff_vs_F1"] = _compare_shared_config(config, reference_config)

        results.append(entry)

    report = {
        "run_id": args.run_id,
        "model_name": model_name,
        "model_name_safe": safe_model,
        "runs": results,
        "comparison_rows": comparison_rows,
    }

    output_file = args.output_file or (args.train_output_dir / args.run_id / "run_validation_report.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    csv_path = output_file.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "run_tag",
                "method",
                "condition",
                "accuracy",
                "macro_f1",
                "trainable_params",
                "trainable_ratio",
                "peak_train_allocated_gb",
                "total_training_time_seconds",
                "best_checkpoint_size_bytes",
                "inference_examples_per_second",
                "inference_avg_latency_seconds",
                "invalid_rate",
            ],
        )
        writer.writeheader()
        for row in comparison_rows:
            writer.writerow(row)

    has_error = any(not row["valid"] for row in results)
    for row in results:
        status = "OK" if row["valid"] else "MISSING"
        print(f"[{status}] {row['run_tag']} -> {row['run_dir']}")
        if row["missing_files"]:
            for missing in row["missing_files"]:
                print(f"  - missing: {missing}")
        diff = row.get("shared_config_diff_vs_F1")
        if diff:
            print(f"  - config_diff_vs_F1: {sorted(diff.keys())}")

    print(f"[report] {output_file}")
    print(f"[table] {csv_path}")
    if has_error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
