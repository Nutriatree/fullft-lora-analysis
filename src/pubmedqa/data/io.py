"""JSON/JSONL adapters for PubMedQA dataset records."""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Mapping

from pubmedqa.runtime.io import write_json, write_jsonl

DatasetRow = dict[str, Any]
OrderedRows = OrderedDict[str, DatasetRow]


def read_jsonl(path: Path) -> list[DatasetRow]:
    rows: list[DatasetRow] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL at {path}:{line_number}: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def write_dataset(path: Path, rows: Mapping[str, DatasetRow]) -> None:
    write_json(path, rows)


def write_dataset_jsonl(
    path: Path,
    rows: Mapping[str, DatasetRow] | Iterable[DatasetRow],
) -> None:
    write_jsonl(path, rows.values() if isinstance(rows, Mapping) else rows)
