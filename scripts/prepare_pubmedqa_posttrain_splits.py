#!/usr/bin/env python3
"""Prepare balanced post-training splits for PubMedQA."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/posttrain_v1"))
    parser.add_argument(
        "--pqa-artificial-train-source",
        type=Path,
        default=Path("data/processed/pqa_artificial/train.jsonl"),
    )
    parser.add_argument(
        "--pqa-artificial-validation-source",
        type=Path,
        default=Path("data/processed/pqa_artificial/validation.jsonl"),
    )
    parser.add_argument(
        "--pqa-labeled-test-source",
        type=Path,
        default=Path("data/processed/pqa_labeled/test.jsonl"),
    )
    parser.add_argument("--artificial-train-size", type=int, default=10_000)
    parser.add_argument("--artificial-validation-size", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.write("\n")


def write_jsonl(path: Path, rows: OrderedDict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows.values():
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def label_counts(rows: OrderedDict[str, dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(row["final_decision"] for row in rows.values()))


def rows_by_label(rows: list[dict[str, Any]], allowed_labels: tuple[str, ...]) -> dict[str, list[dict[str, Any]]]:
    grouped = {label: [] for label in allowed_labels}
    for row in rows:
        label = str(row.get("final_decision", "")).lower()
        if label in grouped:
            grouped[label].append(row)
    return grouped


def annotate_rows(
    rows: list[dict[str, Any]],
    *,
    dataset_id: str,
    collection: str,
    source: str,
    split: str,
    role: str,
) -> OrderedDict[str, dict[str, Any]]:
    annotated: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for row in rows:
        item = dict(row)
        pubid = str(item["pubid"])
        item["dataset_id"] = dataset_id
        item["dataset_collection"] = collection
        item["source"] = source
        item["split"] = split
        item["split_role"] = role
        annotated[pubid] = item
    return annotated


def split_balanced_artificial(
    rows: list[dict[str, Any]],
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
def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    pqa_artificial_rows = load_jsonl(args.pqa_artificial_train_source) + load_jsonl(
        args.pqa_artificial_validation_source
    )
    pqa_labeled_test_rows = load_jsonl(args.pqa_labeled_test_source)

    artificial_train_rows, artificial_validation_rows = split_balanced_artificial(
        pqa_artificial_rows,
        train_size=args.artificial_train_size,
        validation_size=args.artificial_validation_size,
        rng=rng,
    )
    artificial_train = annotate_rows(
        artificial_train_rows,
        dataset_id="pqa_artificial",
        collection="PQA-A",
        source=str(args.pqa_artificial_train_source.parent),
        split="posttrain/train",
        role="train",
    )
    artificial_validation = annotate_rows(
        artificial_validation_rows,
        dataset_id="pqa_artificial",
        collection="PQA-A",
        source=str(args.pqa_artificial_train_source.parent),
        split="posttrain/validation",
        role="validation",
    )
    labeled_test = annotate_rows(
        pqa_labeled_test_rows,
        dataset_id="pqa_labeled",
        collection="PQA-L",
        source=str(args.pqa_labeled_test_source),
        split="test",
        role="test",
    )

    outputs = {
        "pqa_artificial/train": artificial_train,
        "pqa_artificial/validation": artificial_validation,
        "pqa_labeled/test": labeled_test,
    }
    for name, rows in outputs.items():
        write_json(args.output_dir / f"{name}.json", rows)
        write_jsonl(args.output_dir / f"{name}.jsonl", rows)

    write_json(
        args.output_dir / "metadata.json",
        {
            "seed": args.seed,
            "policy": {
                "pqa_artificial_train_size": args.artificial_train_size,
                "pqa_artificial_validation_size": args.artificial_validation_size,
                "pqa_artificial_train_distribution": {"yes": args.artificial_train_size // 2, "no": args.artificial_train_size // 2},
                "pqa_artificial_validation_distribution": {
                    "yes": args.artificial_validation_size // 2,
                    "no": args.artificial_validation_size // 2,
                },
                "pqa_labeled_test_size": len(labeled_test),
                "uses_folds": False,
            },
            "splits": {
                name: {
                    "path_json": f"{name}.json",
                    "path_jsonl": f"{name}.jsonl",
                    "examples": len(rows),
                    "label_counts": label_counts(rows),
                }
                for name, rows in outputs.items()
            },
        },
    )


if __name__ == "__main__":
    main()
