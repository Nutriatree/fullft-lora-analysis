"""Shared parameter-update analysis helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from pubmedqa.config.train import TRAIN_LAYER_CONFIG
from pubmedqa.data.records import (
    write_json,
    write_jsonl,
)
from pubmedqa.eval.validation import EvalPrediction
from pubmedqa.train.artifacts import TrainingSummary
from pubmedqa.train.checkpoints import CheckpointRecord
from pubmedqa.train.distributed import (
    _unwrap_model,
    canonical_parameter_name,
    rank_zero_analysis,
    tensor_bytes,
)
from pubmedqa.train.loop import TrainStepLog


@dataclass(frozen=True)
class LayerwiseReference:
    parameter_name: str
    layer_index: int | None
    module_name: str
    component_name: str
    shape: tuple[int, ...]
    num_parameters: int
    base_weight_norm: float
    base_tensor: torch.Tensor


@dataclass(frozen=True)
class LayerwiseUpdateRecord:
    checkpoint_kind: str
    checkpoint_percent: float
    epoch: int
    global_step: int
    layer_index: int | None
    module_name: str
    component_name: str
    parameter_name: str
    shape: tuple[int, ...]
    num_parameters: int
    base_weight_norm: float
    current_weight_norm: float
    update_norm: float
    relative_update_norm: float
    incremental_update_norm: float
    relative_incremental_update_norm: float
    incremental_update_cosine_similarity: float | None
    cumulative_update_share: float
    incremental_update_share: float


TRACKED_MODULE_SUFFIXES = dict(TRAIN_LAYER_CONFIG.default_tracked_module_suffixes)


def cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left_norm = float(left.norm().item())
    right_norm = float(right.norm().item())
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return float(
        torch.nn.functional.cosine_similarity(
            left.flatten(),
            right.flatten(),
            dim=0,
        ).item()
    )


def parse_layer_index(parameter_name: str) -> int | None:
    parts = parameter_name.split(".")
    for index, part in enumerate(parts[:-1]):
        if part != "layers":
            continue
        try:
            return int(parts[index + 1])
        except ValueError:
            return None
    return None


def match_tracked_module(parameter_name: str) -> tuple[str, str] | None:
    for suffix, match in TRACKED_MODULE_SUFFIXES.items():
        if parameter_name.endswith(suffix):
            return match
    return None


def summarize_by_component(
    records: Sequence[LayerwiseUpdateRecord],
) -> dict[str, dict[str, float | int]]:
    """Aggregate layer-update measurements by Transformer component and module."""

    summary: dict[str, dict[str, float | int]] = {}
    for record in records:
        key = f"{record.component_name}:{record.module_name}"
        bucket = summary.setdefault(
            key,
            {
                "count": 0,
                "sum_base_weight_norm": 0.0,
                "sum_cumulative_update_norm": 0.0,
                "sum_incremental_update_norm": 0.0,
                "sum_relative_update_norm": 0.0,
                "sum_relative_incremental_update_norm": 0.0,
                "sum_cumulative_update_share": 0.0,
                "sum_incremental_update_share": 0.0,
            },
        )
        bucket["count"] += 1
        bucket["sum_base_weight_norm"] += record.base_weight_norm
        bucket["sum_cumulative_update_norm"] += record.update_norm
        bucket["sum_incremental_update_norm"] += record.incremental_update_norm
        bucket["sum_relative_update_norm"] += record.relative_update_norm
        bucket["sum_relative_incremental_update_norm"] += (
            record.relative_incremental_update_norm
        )
        bucket["sum_cumulative_update_share"] += record.cumulative_update_share
        bucket["sum_incremental_update_share"] += record.incremental_update_share
    for bucket in summary.values():
        count = int(bucket["count"])
        bucket["avg_relative_update_norm"] = (
            bucket["sum_relative_update_norm"] / count if count > 0 else 0.0
        )
        bucket["avg_relative_incremental_update_norm"] = (
            bucket["sum_relative_incremental_update_norm"] / count if count > 0 else 0.0
        )
    return summary


def summarize_by_layer(
    records: Sequence[LayerwiseUpdateRecord],
) -> dict[str, dict[str, float | int]]:
    """Aggregate layer-update measurements by Transformer layer index."""

    summary: dict[str, dict[str, float | int]] = {}
    for record in records:
        key = "none" if record.layer_index is None else str(record.layer_index)
        bucket = summary.setdefault(
            key,
            {
                "count": 0,
                "sum_cumulative_update_norm": 0.0,
                "sum_incremental_update_norm": 0.0,
                "sum_cumulative_update_share": 0.0,
                "sum_incremental_update_share": 0.0,
            },
        )
        bucket["count"] += 1
        bucket["sum_cumulative_update_norm"] += record.update_norm
        bucket["sum_incremental_update_norm"] += record.incremental_update_norm
        bucket["sum_cumulative_update_share"] += record.cumulative_update_share
        bucket["sum_incremental_update_share"] += record.incremental_update_share
    return summary


def average_incremental_cosine_similarity(
    records: Sequence[LayerwiseUpdateRecord],
) -> float | None:
    values = [
        record.incremental_update_cosine_similarity
        for record in records
        if record.incremental_update_cosine_similarity is not None
    ]
    return None if not values else sum(values) / len(values)


@rank_zero_analysis
def capture_layerwise_references(
    model: torch.nn.Module, *, files
) -> list[LayerwiseReference]:
    model = _unwrap_model(model)
    references: list[LayerwiseReference] = []
    for parameter_name, parameter in model.named_parameters():
        parameter_name = canonical_parameter_name(parameter_name)
        match = _match_tracked_module(parameter_name)
        if match is None:
            continue
        module_name, component_name = match
        base_tensor = parameter.detach().cpu().clone()
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
    if not references:
        raise ValueError("Tracking enabled but no supported parameters were found")
    write_json(
        files.layerwise_dir / "base_reference_summary.json",
        {
            "run_id": files.run_id,
            "run_tag": files.run_tag,
            "model_name": files.model_name,
            "tracked_parameters": [
                {
                    "parameter_name": reference.parameter_name,
                    "layer_index": reference.layer_index,
                    "module_name": reference.module_name,
                    "component_name": reference.component_name,
                    "shape": list(reference.shape),
                    "num_parameters": reference.num_parameters,
                    "base_weight_norm": reference.base_weight_norm,
                    "reference_dtype": str(reference.base_tensor.dtype),
                }
                for reference in references
            ],
            "schema_version": 2,
            "accumulation_dtype": "torch.float32",
            "reference_bytes": tensor_bytes(r.base_tensor for r in references),
        },
    )
    return references


@rank_zero_analysis
def write_layerwise_update_artifacts(
    *,
    files,
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
    if not references:
        raise ValueError("Tracking enabled but no parameter references were supplied")

    named_parameters = {
        canonical_parameter_name(name): parameter
        for name, parameter in _unwrap_model(model).named_parameters()
    }
    records: list[LayerwiseUpdateRecord] = []
    current_snapshots: dict[str, torch.Tensor] = {}
    for reference in references:
        parameter = named_parameters.get(reference.parameter_name)
        if parameter is None:
            raise ValueError(
                f"Missing full parameter for analysis: {reference.parameter_name}"
            )
        current_tensor = parameter.detach().cpu().float().clone()
        base_tensor = reference.base_tensor.float()
        previous_tensor = previous_snapshots.get(
            reference.parameter_name, base_tensor
        ).float()
        previous_incremental_tensor = previous_incremental_updates.get(
            reference.parameter_name
        )
        update_tensor = current_tensor - base_tensor
        incremental_update_tensor = current_tensor - previous_tensor
        current_weight_norm = float(current_tensor.norm().item())
        update_norm = float(update_tensor.norm().item())
        incremental_update_norm = float(incremental_update_tensor.norm().item())
        relative_update_norm = (
            0.0
            if reference.base_weight_norm == 0.0
            else update_norm / reference.base_weight_norm
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
        current_snapshots[reference.parameter_name] = parameter.detach().cpu().clone()
        previous_incremental_updates[reference.parameter_name] = (
            incremental_update_tensor.clone()
        )

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
                    0.0
                    if total_cumulative_update == 0.0
                    else record.update_norm / total_cumulative_update
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
    write_jsonl(files.layerwise_dir / f"{file_stem}.jsonl", rows)
    write_jsonl(checkpoint_dir / "layerwise_updates.jsonl", rows)
    write_json(
        files.layerwise_dir / f"{file_stem}_summary.json",
        {
            "checkpoint_kind": checkpoint_kind,
            "checkpoint_percent": checkpoint_percent,
            "epoch": epoch,
            "global_step": global_step,
            "num_records": len(normalized_records),
            "schema_version": 2,
            "reference_dtypes": sorted({str(r.base_tensor.dtype) for r in references}),
            "accumulation_dtype": "torch.float32",
            "reference_bytes": tensor_bytes(r.base_tensor for r in references),
            "snapshot_bytes": tensor_bytes(previous_snapshots.values()),
            "incremental_bytes": tensor_bytes(previous_incremental_updates.values()),
            "total_cumulative_update_norm": total_cumulative_update,
            "total_incremental_update_norm": total_incremental_update,
            "by_component": _summarize_layerwise_by_component(normalized_records),
            "by_layer": _summarize_layerwise_by_layer(normalized_records),
            "avg_incremental_update_cosine_similarity": _average_cosine_similarity(
                normalized_records
            ),
        },
    )


def _summarize_layerwise_by_component(
    records: Sequence[LayerwiseUpdateRecord],
) -> dict[str, dict[str, float | int]]:
    return summarize_by_component(records)


def _summarize_layerwise_by_layer(
    records: Sequence[LayerwiseUpdateRecord],
) -> dict[str, dict[str, float | int]]:
    return summarize_by_layer(records)


def _average_cosine_similarity(
    records: Sequence[LayerwiseUpdateRecord],
) -> float | None:
    return average_incremental_cosine_similarity(records)


def write_prediction_transition_artifacts(
    *,
    files,
    from_split: str,
    to_split: str,
    previous_predictions: Sequence[EvalPrediction],
    current_predictions: Sequence[EvalPrediction],
) -> None:
    previous_by_pubid = {
        prediction.pubid: prediction for prediction in previous_predictions
    }
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
            transition_type = (
                "wrong_to_wrong_changed"
                if previous.predicted_label != prediction.predicted_label
                else "wrong_to_wrong_same"
            )
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
    write_json(files.transitions_dir / f"{stem}_summary.json", counts)
    write_jsonl(files.transitions_dir / f"{stem}.jsonl", rows)


def write_analysis_groups(
    files,
    summary: TrainingSummary,
    checkpoint_records: Sequence[CheckpointRecord],
    step_logs: Sequence[TrainStepLog],
    *,
    checkpoint_percents,
) -> None:
    write_json(
        files.output_root / "analysis" / "parameter_model_size.json",
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
        files.output_root / "analysis" / "memory.json",
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
        files.output_root / "analysis" / "training_efficiency.json",
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
        files.output_root / "analysis" / "optimization.json",
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
        files.output_root / "analysis" / "learning_dynamics.json",
        {
            "run_tag": summary.run_tag,
            "method_name": summary.method_name,
            "checkpoint_percents": list(checkpoint_percents),
            "checkpoint_timeline": [asdict(record) for record in checkpoint_records],
        },
    )
    write_json(
        files.output_root / "analysis" / "performance_dynamics.json",
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
            "step_logs_file": str(files.logs_dir / "train_steps.jsonl"),
        },
    )
    write_json(
        files.output_root / "analysis" / "performance_generalization.json",
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
        files.output_root / "analysis" / "inference_deployment.json",
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
        files.output_root / "analysis" / "lora_configuration.json",
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
        files.output_root / "analysis" / "update_distribution.json",
        {
            "run_tag": summary.run_tag,
            "method_name": summary.method_name,
            "layerwise_update_dir": str(files.layerwise_dir),
            "checkpoint_timeline": [
                {
                    "checkpoint_kind": record.checkpoint_kind,
                    "checkpoint_percent": record.checkpoint_percent,
                    "epoch": record.epoch,
                    "global_step": record.global_step,
                    "layerwise_jsonl": str(
                        files.layerwise_dir
                        / (
                            f"{record.checkpoint_kind}_pct_{int(round(record.checkpoint_percent)):03d}_"
                            f"epoch_{record.epoch:03d}_step_{record.global_step:06d}.jsonl"
                        )
                    ),
                    "layerwise_summary_json": str(
                        files.layerwise_dir
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
        files.output_root / "analysis" / "prediction_transitions.json",
        {
            "run_tag": summary.run_tag,
            "method_name": summary.method_name,
            "transitions_dir": str(files.transitions_dir),
        },
    )
    write_jsonl(
        files.output_root / "analysis" / "checkpoint_timeline.jsonl",
        (asdict(record) for record in checkpoint_records),
    )
    write_jsonl(
        files.output_root / "analysis" / "optimization_timeline.jsonl",
        (asdict(record) for record in step_logs),
    )


_cosine_similarity = cosine_similarity
_match_tracked_module = match_tracked_module
_parse_layer_index = parse_layer_index
