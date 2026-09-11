"""Normalized PubMedQA records, labels and local JSON artifact I/O."""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

VALID_LABELS = ("yes", "no", "maybe")


LABEL_PATTERN = re.compile(r"\b(yes|no|maybe)\b", re.IGNORECASE)


def normalize_label(value: Any, *, allow_embedded: bool = False) -> str | None:
    """Normalize a value to a PubMedQA label, or return ``None``."""

    if value is None:
        return None
    text = str(value).strip().lower()
    if text in VALID_LABELS:
        return text
    if allow_embedded:
        match = LABEL_PATTERN.search(text)
        if match:
            return match.group(1).lower()
    return None


def require_label(value: Any, *, allow_embedded: bool = False) -> str:
    """Normalize a label or raise ``ValueError``."""

    label = normalize_label(value, allow_embedded=allow_embedded)
    if label is None:
        raise ValueError(f"Invalid PubMedQA label: {value!r}")
    return label


@dataclass(frozen=True)
class PubMedQAExample:
    pubid: str
    question: str
    contexts: tuple[str, ...]
    labels: tuple[str, ...] = ()
    meshes: tuple[str, ...] = ()
    final_decision: str | None = None
    long_answer: str | None = None
    dataset_id: str | None = None
    split: str | None = None


def example_from_record(record: dict[str, Any]) -> PubMedQAExample:
    """Normalize a generated PubMedQA JSON/JSONL row."""

    context = record.get("context") or {}
    contexts = context.get("contexts")
    if contexts is None:
        contexts = record.get("contexts", [])
    labels = context.get("labels") or record.get("labels") or ()
    meshes = context.get("meshes") or record.get("meshes") or ()

    return PubMedQAExample(
        pubid=str(record.get("pubid", "")),
        question=str(record["question"]),
        contexts=tuple(str(item) for item in contexts),
        labels=tuple(str(item) for item in labels),
        meshes=tuple(str(item) for item in meshes),
        final_decision=normalize_label(record.get("final_decision")),
        long_answer=record.get("long_answer"),
        dataset_id=record.get("dataset_id"),
        split=record.get("split"),
    )


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False, default=str)
        file.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def current_time_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def count_directory_size_bytes(path: Path) -> int:
    return sum(
        file_path.stat().st_size for file_path in path.rglob("*") if file_path.is_file()
    )


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
                raise ValueError(
                    f"Malformed JSONL at {path}:{line_number}: {exc.msg}"
                ) from exc
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


def load_local_jsonl(
    path: Path, expected_size: int | None = None
) -> list[PubMedQAExample]:
    # Import lazily so basic data/reporting tools remain framework-independent.
    from datasets import load_dataset

    if not path.is_file():
        raise FileNotFoundError(f"JSONL not found: {path}")

    dataset = load_dataset("json", data_files={"eval": str(path)}, split="eval")
    examples = [example_from_record(row) for row in dataset]

    if not examples:
        raise RuntimeError(f"Dataset is empty: {path}")
    if expected_size is not None and len(examples) != expected_size:
        raise RuntimeError(
            f"Expected {expected_size} examples but loaded {len(examples)} from {path}."
        )

    invalid_gold = [
        example.pubid
        for example in examples
        if example.final_decision not in VALID_LABELS
    ]
    if invalid_gold:
        raise RuntimeError(
            f"Invalid gold labels found, e.g. {', '.join(invalid_gold[:5])}"
        )

    return examples
