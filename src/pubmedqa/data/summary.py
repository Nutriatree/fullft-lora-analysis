"""Dataset summary and split-integrity validation."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

EXPECTED_DATASET_ID_BY_PATH = {
    "pqa_artificial": "pqa_artificial",
    "pqa_labeled": "pqa_labeled",
}


def read_jsonl_summary(path: Path) -> dict[str, Any]:
    dataset_ids: Counter[str] = Counter()
    collections: Counter[str] = Counter()
    roles: Counter[str] = Counter()
    splits: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    examples = 0

    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSONL at {path}:{line_number}: {exc.msg}"
                ) from exc
            examples += 1
            dataset_ids[str(row.get("dataset_id"))] += 1
            collections[str(row.get("dataset_collection"))] += 1
            roles[str(row.get("split_role"))] += 1
            splits[str(row.get("split"))] += 1
            labels[str(row.get("final_decision"))] += 1
            if row.get("pubid") is None:
                raise ValueError(f"{path}:{line_number} is missing pubid")

    return {
        "path": str(path),
        "examples": examples,
        "dataset_ids": dict(dataset_ids),
        "dataset_collections": dict(collections),
        "split_roles": dict(roles),
        "splits": dict(splits),
        "label_counts": dict(labels),
    }


def expected_dataset_id(path: Path) -> str | None:
    return next(
        (
            EXPECTED_DATASET_ID_BY_PATH[part]
            for part in path.parts
            if part in EXPECTED_DATASET_ID_BY_PATH
        ),
        None,
    )


def validate_summary(summary: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    path = Path(summary["path"])
    dataset_ids = summary["dataset_ids"]
    if len(dataset_ids) != 1:
        errors.append(f"{path} has mixed dataset_id values: {dataset_ids}")
    expected = expected_dataset_id(path)
    if expected is not None and dataset_ids != {expected: summary["examples"]}:
        errors.append(f"{path} expected dataset_id={expected}, got {dataset_ids}")
    if "pqa_artificial" in dataset_ids and "maybe" in summary["label_counts"]:
        errors.append(f"{path} is PQA-A but contains maybe labels")
    return errors


def describe_directory(data_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    jsonl_files = sorted((data_dir / "pqa_artificial").rglob("*.jsonl"))
    jsonl_files.extend(sorted((data_dir / "pqa_labeled").rglob("*.jsonl")))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found under {data_dir}")
    summaries = [read_jsonl_summary(path) for path in jsonl_files]
    errors = [error for summary in summaries for error in validate_summary(summary)]
    return summaries, errors
