"""LoRA update measurements and their own history, independent of the trainer."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import torch

from pubmedqa.data.records import (
    write_json,
    write_jsonl,
)
from pubmedqa.train.artifacts import TrainingSummary
from pubmedqa.train.checkpoints import CheckpointRecord
from pubmedqa.train.distributed import (
    _unwrap_model,
    canonical_parameter_name,
    rank_zero_analysis,
    tensor_bytes,
)
from pubmedqa.train.full_ft import (
    LayerwiseReference,
    LayerwiseUpdateRecord,
    _average_cosine_similarity,
    _summarize_layerwise_by_component,
    _summarize_layerwise_by_layer,
    match_tracked_module,
    parse_layer_index,
)
from pubmedqa.train.full_ft import (
    cosine_similarity as tensor_cosine_similarity,
)
from pubmedqa.train.full_ft import (
    write_analysis_groups as write_full_analysis_groups,
)


@dataclass
class AdapterHistory:
    """Only adapter measurements; never retain a model or optimizer."""

    adapter_metrics_history: list[dict[str, Any]] = field(default_factory=list)
    module_share_history: list[dict[str, Any]] = field(default_factory=list)


def _named_lora_modules(model: torch.nn.Module) -> list[tuple[str, Any]]:
    model = _unwrap_model(model)
    modules: list[tuple[str, Any]] = []
    for module_name, module in model.named_modules():
        if (
            hasattr(module, "lora_A")
            and hasattr(module, "lora_B")
            and hasattr(module, "base_layer")
        ):
            modules.append((canonical_parameter_name(module_name), module))
    return modules


def _compute_lora_delta_tensor(
    module: Any,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    adapter_names = list(module.lora_A.keys())
    if not adapter_names:
        raise RuntimeError("LoRA module has no registered adapter.")
    adapter_name = adapter_names[0]
    a_weight = module.lora_A[adapter_name].weight.detach().cpu().float()
    b_weight = module.lora_B[adapter_name].weight.detach().cpu().float()
    scaling = float(module.scaling[adapter_name])
    delta = scaling * (b_weight @ a_weight)
    return a_weight, b_weight, delta, scaling


@rank_zero_analysis
def capture_layerwise_references(
    model: torch.nn.Module, *, files
) -> list[LayerwiseReference]:
    references: list[LayerwiseReference] = []
    tracked_parameters: list[dict[str, Any]] = []
    for module_name, module in _named_lora_modules(model):
        parameter_name = f"{module_name}.weight"
        match = match_tracked_module(parameter_name)
        if match is None:
            continue
        module_alias, component_name = match
        base_weight = module.base_layer.weight.detach().cpu().float()
        references.append(
            LayerwiseReference(
                parameter_name=parameter_name,
                layer_index=parse_layer_index(parameter_name),
                module_name=module_alias,
                component_name=component_name,
                shape=tuple(base_weight.shape),
                num_parameters=base_weight.numel(),
                base_weight_norm=float(base_weight.norm().item()),
                base_tensor=torch.zeros_like(base_weight, dtype=torch.float32),
            )
        )
        tracked_parameters.append(
            {
                "parameter_name": parameter_name,
                "layer_index": parse_layer_index(parameter_name),
                "module_name": module_alias,
                "component_name": component_name,
                "shape": list(base_weight.shape),
                "num_parameters": base_weight.numel(),
                "base_weight_norm": float(base_weight.norm().item()),
                "tracked_as": "lora_delta",
            }
        )
    if not references:
        raise ValueError("Tracking enabled but no supported LoRA parameters were found")
    write_json(
        files.layerwise_dir / "base_reference_summary.json",
        {
            "run_id": files.run_id,
            "run_tag": files.run_tag,
            "model_name": files.model_name,
            "tracked_parameters": tracked_parameters,
            "schema_version": 2,
            "reference_dtype": "torch.float32",
            "accumulation_dtype": "torch.float32",
            "reference_bytes": tensor_bytes(r.base_tensor for r in references),
        },
    )
    return references


@rank_zero_analysis
def write_layerwise_update_artifacts(
    *,
    files,
    history,
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
        raise ValueError("Tracking enabled but no LoRA references were supplied")

    reference_map = {reference.parameter_name: reference for reference in references}
    module_map = {name: module for name, module in _named_lora_modules(model)}
    layerwise_records: list[LayerwiseUpdateRecord] = []
    adapter_records: list[LoRAAdapterMetric] = []
    current_snapshots: dict[str, torch.Tensor] = {}

    for module_name, module in module_map.items():
        parameter_name = f"{module_name}.weight"
        reference = reference_map.get(parameter_name)
        if reference is None:
            continue

        a_weight, b_weight, delta_tensor, scaling = _compute_lora_delta_tensor(module)
        previous_tensor = previous_snapshots.get(parameter_name)
        if previous_tensor is None:
            previous_tensor = torch.zeros_like(delta_tensor)
        previous_tensor = previous_tensor.float()
        previous_incremental_tensor = previous_incremental_updates.get(parameter_name)

        incremental_delta = delta_tensor - previous_tensor
        delta_norm = float(delta_tensor.norm().item())
        incremental_delta_norm = float(incremental_delta.norm().item())
        delta_relative_norm = (
            0.0
            if reference.base_weight_norm == 0.0
            else delta_norm / reference.base_weight_norm
        )
        incremental_relative_norm = (
            0.0
            if reference.base_weight_norm == 0.0
            else incremental_delta_norm / reference.base_weight_norm
        )
        singular_values = torch.linalg.svdvals(delta_tensor)
        effective_rank = _effective_rank_from_singular_values(singular_values)
        cosine_similarity = None
        if previous_incremental_tensor is not None:
            cosine_similarity = tensor_cosine_similarity(
                incremental_delta,
                previous_incremental_tensor.float(),
            )

        layerwise_records.append(
            LayerwiseUpdateRecord(
                checkpoint_kind=checkpoint_kind,
                checkpoint_percent=checkpoint_percent,
                epoch=epoch,
                global_step=global_step,
                layer_index=reference.layer_index,
                module_name=reference.module_name,
                component_name=reference.component_name,
                parameter_name=parameter_name,
                shape=reference.shape,
                num_parameters=reference.num_parameters,
                base_weight_norm=reference.base_weight_norm,
                current_weight_norm=delta_norm,
                update_norm=delta_norm,
                relative_update_norm=delta_relative_norm,
                incremental_update_norm=incremental_delta_norm,
                relative_incremental_update_norm=incremental_relative_norm,
                incremental_update_cosine_similarity=cosine_similarity,
                cumulative_update_share=0.0,
                incremental_update_share=0.0,
            )
        )
        adapter_records.append(
            LoRAAdapterMetric(
                checkpoint_id=checkpoint_dir.name,
                checkpoint_kind=checkpoint_kind,
                checkpoint_percent=checkpoint_percent,
                epoch=epoch,
                global_step=global_step,
                layer_index=reference.layer_index,
                module_name=reference.module_name,
                component_name=reference.component_name,
                parameter_name=parameter_name,
                base_weight_norm=reference.base_weight_norm,
                a_norm=float(a_weight.norm().item()),
                b_norm=float(b_weight.norm().item()),
                scaling=scaling,
                delta_w_norm=delta_norm,
                delta_w_relative_norm=delta_relative_norm,
                incremental_delta_w_norm=incremental_delta_norm,
                incremental_delta_w_relative_norm=incremental_relative_norm,
                incremental_delta_w_cosine_similarity=cosine_similarity,
                effective_rank=effective_rank,
                singular_values=_tensor_list(singular_values),
                cumulative_update_share=0.0,
                incremental_update_share=0.0,
            )
        )
        current_snapshots[parameter_name] = delta_tensor.clone()
        previous_incremental_updates[parameter_name] = incremental_delta.clone()

    if len(layerwise_records) != len(references):
        raise ValueError("Missing full LoRA parameters for analysis")

    total_cumulative = sum(record.update_norm for record in layerwise_records)
    total_incremental = sum(
        record.incremental_update_norm for record in layerwise_records
    )
    normalized_layerwise: list[LayerwiseUpdateRecord] = []
    normalized_adapter: list[LoRAAdapterMetric] = []
    for layer_record, adapter_record in zip(layerwise_records, adapter_records):
        cumulative_share = (
            0.0
            if total_cumulative == 0.0
            else layer_record.update_norm / total_cumulative
        )
        incremental_share = (
            0.0
            if total_incremental == 0.0
            else layer_record.incremental_update_norm / total_incremental
        )
        normalized_layerwise.append(
            LayerwiseUpdateRecord(
                checkpoint_kind=layer_record.checkpoint_kind,
                checkpoint_percent=layer_record.checkpoint_percent,
                epoch=layer_record.epoch,
                global_step=layer_record.global_step,
                layer_index=layer_record.layer_index,
                module_name=layer_record.module_name,
                component_name=layer_record.component_name,
                parameter_name=layer_record.parameter_name,
                shape=layer_record.shape,
                num_parameters=layer_record.num_parameters,
                base_weight_norm=layer_record.base_weight_norm,
                current_weight_norm=layer_record.current_weight_norm,
                update_norm=layer_record.update_norm,
                relative_update_norm=layer_record.relative_update_norm,
                incremental_update_norm=layer_record.incremental_update_norm,
                relative_incremental_update_norm=layer_record.relative_incremental_update_norm,
                incremental_update_cosine_similarity=layer_record.incremental_update_cosine_similarity,
                cumulative_update_share=cumulative_share,
                incremental_update_share=incremental_share,
            )
        )
        normalized_adapter.append(
            LoRAAdapterMetric(
                checkpoint_id=adapter_record.checkpoint_id,
                checkpoint_kind=adapter_record.checkpoint_kind,
                checkpoint_percent=adapter_record.checkpoint_percent,
                epoch=adapter_record.epoch,
                global_step=adapter_record.global_step,
                layer_index=adapter_record.layer_index,
                module_name=adapter_record.module_name,
                component_name=adapter_record.component_name,
                parameter_name=adapter_record.parameter_name,
                base_weight_norm=adapter_record.base_weight_norm,
                a_norm=adapter_record.a_norm,
                b_norm=adapter_record.b_norm,
                scaling=adapter_record.scaling,
                delta_w_norm=adapter_record.delta_w_norm,
                delta_w_relative_norm=adapter_record.delta_w_relative_norm,
                incremental_delta_w_norm=adapter_record.incremental_delta_w_norm,
                incremental_delta_w_relative_norm=adapter_record.incremental_delta_w_relative_norm,
                incremental_delta_w_cosine_similarity=adapter_record.incremental_delta_w_cosine_similarity,
                effective_rank=adapter_record.effective_rank,
                singular_values=adapter_record.singular_values,
                cumulative_update_share=cumulative_share,
                incremental_update_share=incremental_share,
            )
        )

    previous_snapshots.update(current_snapshots)
    file_stem = checkpoint_dir.name
    write_jsonl(
        files.layerwise_dir / f"{file_stem}.jsonl",
        (asdict(row) for row in normalized_layerwise),
    )
    write_jsonl(
        checkpoint_dir / "layerwise_updates.jsonl",
        (asdict(row) for row in normalized_layerwise),
    )
    write_jsonl(
        checkpoint_dir / "lora_adapter_metrics.jsonl",
        (asdict(row) for row in normalized_adapter),
    )

    component_summary = _summarize_layerwise_by_component(normalized_layerwise)
    layer_summary = _summarize_layerwise_by_layer(normalized_layerwise)
    module_share_summary = _summarize_module_share(normalized_adapter)
    avg_cosine = _average_cosine_similarity(normalized_layerwise)
    summary_payload = {
        "checkpoint_id": checkpoint_dir.name,
        "checkpoint_kind": checkpoint_kind,
        "checkpoint_percent": checkpoint_percent,
        "epoch": epoch,
        "global_step": global_step,
        "num_records": len(normalized_adapter),
        "schema_version": 2,
        "reference_dtype": "torch.float32",
        "accumulation_dtype": "torch.float32",
        "reference_bytes": tensor_bytes(r.base_tensor for r in references),
        "snapshot_bytes": tensor_bytes(previous_snapshots.values()),
        "incremental_bytes": tensor_bytes(previous_incremental_updates.values()),
        "total_cumulative_update_norm": total_cumulative,
        "total_incremental_update_norm": total_incremental,
        "avg_incremental_update_cosine_similarity": avg_cosine,
        "by_component": component_summary,
        "by_layer": layer_summary,
        "by_module": module_share_summary,
    }
    write_json(files.layerwise_dir / f"{file_stem}_summary.json", summary_payload)

    adapter_summary = {
        "checkpoint_id": checkpoint_dir.name,
        "checkpoint_kind": checkpoint_kind,
        "checkpoint_percent": checkpoint_percent,
        "epoch": epoch,
        "global_step": global_step,
        "adapter_metrics_jsonl": str(checkpoint_dir / "lora_adapter_metrics.jsonl"),
        "module_update_share": module_share_summary,
        "avg_effective_rank": (
            sum(record.effective_rank for record in normalized_adapter)
            / len(normalized_adapter)
            if normalized_adapter
            else 0.0
        ),
    }
    write_json(checkpoint_dir / "lora_adapter_summary.json", adapter_summary)

    history.adapter_metrics_history.append(
        {
            "checkpoint_id": checkpoint_dir.name,
            "checkpoint_kind": checkpoint_kind,
            "checkpoint_percent": checkpoint_percent,
            "epoch": epoch,
            "global_step": global_step,
            "records": [asdict(row) for row in normalized_adapter],
        }
    )
    history.module_share_history.append(
        {
            "checkpoint_id": checkpoint_dir.name,
            "checkpoint_kind": checkpoint_kind,
            "checkpoint_percent": checkpoint_percent,
            "epoch": epoch,
            "global_step": global_step,
            "by_module": module_share_summary,
        }
    )


def _summarize_module_share(
    records: Sequence[LoRAAdapterMetric],
) -> dict[str, dict[str, float | int]]:
    summary: dict[str, dict[str, float | int]] = {}
    for record in records:
        key = record.module_name
        bucket = summary.setdefault(
            key,
            {
                "count": 0,
                "sum_delta_w_norm": 0.0,
                "sum_delta_w_relative_norm": 0.0,
                "sum_incremental_delta_w_norm": 0.0,
                "sum_incremental_delta_w_relative_norm": 0.0,
                "sum_cumulative_update_share": 0.0,
                "sum_incremental_update_share": 0.0,
            },
        )
        bucket["count"] += 1
        bucket["sum_delta_w_norm"] += record.delta_w_norm
        bucket["sum_delta_w_relative_norm"] += record.delta_w_relative_norm
        bucket["sum_incremental_delta_w_norm"] += record.incremental_delta_w_norm
        bucket["sum_incremental_delta_w_relative_norm"] += (
            record.incremental_delta_w_relative_norm
        )
        bucket["sum_cumulative_update_share"] += record.cumulative_update_share
        bucket["sum_incremental_update_share"] += record.incremental_update_share
    for bucket in summary.values():
        count = int(bucket["count"])
        bucket["avg_delta_w_relative_norm"] = (
            bucket["sum_delta_w_relative_norm"] / count if count > 0 else 0.0
        )
        bucket["avg_incremental_delta_w_relative_norm"] = (
            bucket["sum_incremental_delta_w_relative_norm"] / count
            if count > 0
            else 0.0
        )
    return summary


def write_analysis_groups(
    files,
    summary: TrainingSummary,
    checkpoint_records: Sequence[CheckpointRecord],
    step_logs: Sequence[Any],
    *,
    checkpoint_percents,
    adapter,
    history,
    enabled=True,
) -> None:
    write_full_analysis_groups(
        files,
        summary,
        checkpoint_records,
        step_logs,
        checkpoint_percents=checkpoint_percents,
    )
    if not enabled:
        return

    adapter_records: list[dict[str, Any]] = []
    singular_timeline: list[dict[str, Any]] = []
    effective_rank_timeline: list[dict[str, Any]] = []
    val_update_alignment: list[dict[str, Any]] = []
    checkpoint_to_val = {
        record.global_step: {
            "checkpoint_id": Path(record.checkpoint_dir).name,
            "checkpoint_kind": record.checkpoint_kind,
            "checkpoint_percent": record.checkpoint_percent,
            "validation_accuracy": record.validation_accuracy,
            "validation_macro_f1": record.validation_macro_f1,
            "validation_loss": record.validation_loss,
        }
        for record in checkpoint_records
    }

    for entry in history.adapter_metrics_history:
        adapter_records.extend(entry["records"])
        delta_total = sum(record["delta_w_norm"] for record in entry["records"])
        val_row = checkpoint_to_val.get(entry["global_step"], {})
        val_update_alignment.append(
            {
                "checkpoint_id": entry["checkpoint_id"],
                "checkpoint_kind": entry["checkpoint_kind"],
                "checkpoint_percent": entry["checkpoint_percent"],
                "epoch": entry["epoch"],
                "global_step": entry["global_step"],
                "delta_w_norm_total": delta_total,
                "validation_accuracy": val_row.get("validation_accuracy"),
                "validation_macro_f1": val_row.get("validation_macro_f1"),
                "validation_loss": val_row.get("validation_loss"),
            }
        )
        singular_timeline.append(
            {
                "checkpoint_id": entry["checkpoint_id"],
                "checkpoint_kind": entry["checkpoint_kind"],
                "checkpoint_percent": entry["checkpoint_percent"],
                "epoch": entry["epoch"],
                "global_step": entry["global_step"],
                "records": [
                    {
                        "parameter_name": record["parameter_name"],
                        "layer_index": record["layer_index"],
                        "module_name": record["module_name"],
                        "component_name": record["component_name"],
                        "singular_values": record["singular_values"],
                    }
                    for record in entry["records"]
                ],
            }
        )
        effective_rank_timeline.append(
            {
                "checkpoint_id": entry["checkpoint_id"],
                "checkpoint_kind": entry["checkpoint_kind"],
                "checkpoint_percent": entry["checkpoint_percent"],
                "epoch": entry["epoch"],
                "global_step": entry["global_step"],
                "records": [
                    {
                        "parameter_name": record["parameter_name"],
                        "layer_index": record["layer_index"],
                        "module_name": record["module_name"],
                        "component_name": record["component_name"],
                        "effective_rank": record["effective_rank"],
                    }
                    for record in entry["records"]
                ],
            }
        )

    write_json(
        files.output_root / "analysis" / "lora_configuration.json",
        {
            "run_tag": summary.run_tag,
            "method_name": summary.method_name,
            "target_modules": list(adapter.target_modules),
            "target_layers": list(adapter.target_layers),
            "layer_scope": adapter.layer_scope,
            "lora_rank": adapter.lora_rank,
            "lora_alpha": adapter.lora_alpha,
            "lora_dropout": adapter.lora_dropout,
            "lora_bias": adapter.lora_bias,
            "lora_task_type": adapter.lora_task_type,
            "modules_to_save": list(adapter.modules_to_save),
            "merge_for_eval": adapter.merge_for_eval,
            "data_regime": summary.data_regime,
            "data_fraction": summary.data_fraction,
        },
    )
    write_json(
        files.output_root / "analysis" / "lora_dynamics.json",
        {
            "run_tag": summary.run_tag,
            "method_name": summary.method_name,
            "checkpoint_timeline": history.adapter_metrics_history,
        },
    )
    write_json(
        files.output_root / "analysis" / "lora_singular_values.json",
        {
            "run_tag": summary.run_tag,
            "method_name": summary.method_name,
            "checkpoint_timeline": singular_timeline,
        },
    )
    write_json(
        files.output_root / "analysis" / "lora_effective_rank.json",
        {
            "run_tag": summary.run_tag,
            "method_name": summary.method_name,
            "checkpoint_timeline": effective_rank_timeline,
        },
    )
    write_json(
        files.output_root / "analysis" / "lora_adapter_norms.json",
        {
            "run_tag": summary.run_tag,
            "method_name": summary.method_name,
            "checkpoint_timeline": [
                {
                    "checkpoint_id": entry["checkpoint_id"],
                    "checkpoint_kind": entry["checkpoint_kind"],
                    "checkpoint_percent": entry["checkpoint_percent"],
                    "epoch": entry["epoch"],
                    "global_step": entry["global_step"],
                    "records": [
                        {
                            "parameter_name": record["parameter_name"],
                            "a_norm": record["a_norm"],
                            "b_norm": record["b_norm"],
                            "scaling": record["scaling"],
                            "delta_w_norm": record["delta_w_norm"],
                            "delta_w_relative_norm": record["delta_w_relative_norm"],
                        }
                        for record in entry["records"]
                    ],
                }
                for entry in history.adapter_metrics_history
            ],
        },
    )
    write_json(
        files.output_root / "analysis" / "lora_update_direction.json",
        {
            "run_tag": summary.run_tag,
            "method_name": summary.method_name,
            "checkpoint_timeline": [
                {
                    "checkpoint_id": entry["checkpoint_id"],
                    "checkpoint_kind": entry["checkpoint_kind"],
                    "checkpoint_percent": entry["checkpoint_percent"],
                    "epoch": entry["epoch"],
                    "global_step": entry["global_step"],
                    "records": [
                        {
                            "parameter_name": record["parameter_name"],
                            "module_name": record["module_name"],
                            "layer_index": record["layer_index"],
                            "incremental_delta_w_cosine_similarity": record[
                                "incremental_delta_w_cosine_similarity"
                            ],
                        }
                        for record in entry["records"]
                    ],
                }
                for entry in history.adapter_metrics_history
            ],
        },
    )
    write_json(
        files.output_root / "analysis" / "module_update_share.json",
        {
            "run_tag": summary.run_tag,
            "method_name": summary.method_name,
            "checkpoint_timeline": history.module_share_history,
        },
    )
    write_json(
        files.output_root / "analysis" / "val_update_alignment.json",
        {
            "run_tag": summary.run_tag,
            "method_name": summary.method_name,
            "checkpoint_timeline": val_update_alignment,
        },
    )
    write_jsonl(
        files.output_root / "analysis" / "lora_adapter_metrics.jsonl",
        (row for row in adapter_records),
    )


def _effective_rank_from_singular_values(singular_values: torch.Tensor) -> float:
    if singular_values.numel() == 0:
        return 0.0
    total = float(singular_values.sum().item())
    if total == 0.0:
        return 0.0
    probabilities = singular_values / total
    entropy = float(
        (-(probabilities * probabilities.clamp_min(1e-12).log())).sum().item()
    )
    return float(math.exp(entropy))


def _tensor_list(tensor: torch.Tensor) -> list[float]:
    return [float(value) for value in tensor.detach().cpu().tolist()]


@dataclass(frozen=True)
class LoRAAdapterMetric:
    checkpoint_id: str
    checkpoint_kind: str
    checkpoint_percent: float
    epoch: int
    global_step: int
    layer_index: int | None
    module_name: str
    component_name: str
    parameter_name: str
    base_weight_norm: float
    a_norm: float
    b_norm: float
    scaling: float
    delta_w_norm: float
    delta_w_relative_norm: float
    incremental_delta_w_norm: float
    incremental_delta_w_relative_norm: float
    incremental_delta_w_cosine_similarity: float | None
    effective_rank: float
    singular_values: list[float]
    cumulative_update_share: float
    incremental_update_share: float
