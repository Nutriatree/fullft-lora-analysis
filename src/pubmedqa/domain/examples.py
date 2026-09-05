"""Normalized PubMedQA records consumed by prompts and runners."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pubmedqa.domain.labels import normalize_label


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
