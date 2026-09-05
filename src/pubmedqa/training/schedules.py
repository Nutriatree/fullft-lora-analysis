"""Optimizer-step checkpoint schedules."""

from __future__ import annotations

import math
from collections.abc import Sequence

from pubmedqa.config import TRAIN_LAYER_CONFIG


def normalize_checkpoint_percents(values: Sequence[int]) -> tuple[int, ...]:
    normalized = sorted({value for value in values if 0 < value <= 100})
    if not normalized:
        return TRAIN_LAYER_CONFIG.default_checkpoint_percents
    if normalized[-1] != 100:
        normalized.append(100)
    return tuple(normalized)


def build_checkpoint_schedule(
    total_steps: int,
    percents: Sequence[int],
) -> list[tuple[int, int]]:
    schedule: list[tuple[int, int]] = []
    seen_steps: set[int] = set()
    for percent in normalize_checkpoint_percents(percents):
        target_step = max(1, math.ceil(total_steps * (percent / 100.0)))
        if target_step in seen_steps:
            continue
        seen_steps.add(target_step)
        schedule.append((percent, target_step))
    return schedule
