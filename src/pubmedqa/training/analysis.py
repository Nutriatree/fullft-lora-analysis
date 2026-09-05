"""Shared parameter-update analysis helpers."""

from __future__ import annotations

from typing import Sequence

import torch

from pubmedqa.config import TRAIN_LAYER_CONFIG
from pubmedqa.training.contracts import LayerwiseUpdateRecord

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
        bucket["sum_relative_incremental_update_norm"] += record.relative_incremental_update_norm
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
