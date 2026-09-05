"""Pure, seed-controlled PubMedQA split construction rules."""

from __future__ import annotations

import math
import random
from collections import Counter, OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any

LABELS = ("yes", "no", "maybe")


def split_label(pmids: list[str], fold: int, rng: random.Random) -> list[list[str]]:
    rng.shuffle(pmids)
    num_split = math.ceil(len(pmids) / fold)
    return [
        pmids[index * num_split :] if index == fold - 1
        else pmids[index * num_split : (index + 1) * num_split]
        for index in range(fold)
    ]


def official_label_balanced_split(
    dataset: OrderedDict[str, dict[str, Any]],
    fold: int,
    rng: random.Random,
) -> list[OrderedDict[str, dict[str, Any]]]:
    label_to_pmids: dict[str, list[str]] = {label: [] for label in LABELS}
    for pmid, row in dataset.items():
        label = row["final_decision"]
        if label not in label_to_pmids:
            raise ValueError(f"Unexpected final_decision={label!r} for PMID {pmid}")
        label_to_pmids[label].append(pmid)

    split_by_label = {
        label: split_label(pmids, fold, rng)
        for label, pmids in label_to_pmids.items()
    }
    output: list[OrderedDict[str, dict[str, Any]]] = []
    for index in range(fold):
        fold_pmids = [
            pmid
            for label in LABELS
            for pmid in split_by_label[label][index]
        ]
        output.append(OrderedDict((pmid, dataset[pmid]) for pmid in fold_pmids))

    if output and len(output[-1]) != len(output[0]):
        for index in range(fold - 1):
            picked = rng.choice(list(output[index]))
            output[-1][picked] = output[index].pop(picked)
    return output


def annotate_rows(
    rows: Mapping[str, dict[str, Any]] | Sequence[dict[str, Any]],
    *,
    dataset_id: str,
    collection: str,
    source: str,
    split: str,
    role: str,
    fold: int | None = None,
) -> OrderedDict[str, dict[str, Any]]:
    values = rows.values() if isinstance(rows, Mapping) else rows
    annotated: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for row in values:
        item = dict(row)
        pubid = str(item["pubid"])
        item.update(
            dataset_id=dataset_id,
            dataset_collection=collection,
            source=source,
            split=split,
            split_role=role,
        )
        if fold is not None:
            item["fold"] = fold
        annotated[pubid] = item
    return annotated


def combine_folds(
    cv_folds: Sequence[OrderedDict[str, dict[str, Any]]],
    validation_fold: int,
) -> OrderedDict[str, dict[str, Any]]:
    combined: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for index, fold in enumerate(cv_folds):
        if index != validation_fold:
            combined.update(fold)
    return combined


def label_counts(
    rows: Mapping[str, dict[str, Any]] | Sequence[dict[str, Any]],
) -> dict[str, int]:
    values = rows.values() if isinstance(rows, Mapping) else rows
    return dict(Counter(str(row["final_decision"]).lower() for row in values))


def rows_by_label(
    rows: Sequence[dict[str, Any]],
    allowed_labels: tuple[str, ...],
) -> dict[str, list[dict[str, Any]]]:
    grouped = {label: [] for label in allowed_labels}
    for row in rows:
        label = str(row.get("final_decision", "")).lower()
        if label in grouped:
            grouped[label].append(row)
    return grouped


def split_balanced_artificial(
    rows: Sequence[dict[str, Any]],
    *,
    train_size: int,
    validation_size: int,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    labels = ("yes", "no")
    if train_size % 2 != 0 or validation_size % 2 != 0:
        raise ValueError("Balanced train/validation sizes must be even.")
    grouped = rows_by_label(rows, labels)
    for label in labels:
        rng.shuffle(grouped[label])

    train_per_label = train_size // 2
    validation_per_label = validation_size // 2
    train_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    for label in labels:
        required = train_per_label + validation_per_label
        if len(grouped[label]) < required:
            raise ValueError(
                f"Not enough {label!r} examples to create disjoint train/validation splits: "
                f"required {required}, found {len(grouped[label])}."
            )
        train_rows.extend(grouped[label][:train_per_label])
        validation_rows.extend(grouped[label][train_per_label:required])
    rng.shuffle(train_rows)
    rng.shuffle(validation_rows)
    return train_rows, validation_rows
