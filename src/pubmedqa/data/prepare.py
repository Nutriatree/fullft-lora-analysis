"""Prepare canonical and post-training datasets with explicit split rules."""

from __future__ import annotations

import math
import random
import shutil
from collections import Counter, OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pubmedqa.data.records import (
    read_jsonl,
    write_dataset,
    write_dataset_jsonl,
    write_json,
)

HF_DATASET = "qiaojin/PubMedQA"


HF_DATASET_URL = "https://huggingface.co/datasets/qiaojin/PubMedQA"


OFFICIAL_PQAL_URL = (
    "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json"
)


OFFICIAL_PQAL_TEST_URL = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/test_ground_truth.json"


class CanonicalSourcePort(Protocol):
    @property
    def labeled_source_url(self) -> str: ...

    def load_artificial(self) -> OrderedDict[str, dict[str, Any]]: ...

    def load_labeled(self) -> OrderedDict[str, dict[str, Any]]: ...

    def load_official_test_labels(self) -> dict[str, str] | None: ...


LABELS = ("yes", "no", "maybe")


def split_label(pmids: list[str], fold: int, rng: random.Random) -> list[list[str]]:
    rng.shuffle(pmids)
    num_split = math.ceil(len(pmids) / fold)
    return [
        pmids[index * num_split :]
        if index == fold - 1
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
        label: split_label(pmids, fold, rng) for label, pmids in label_to_pmids.items()
    }
    output: list[OrderedDict[str, dict[str, Any]]] = []
    for index in range(fold):
        fold_pmids = [pmid for label in LABELS for pmid in split_by_label[label][index]]
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


@dataclass(frozen=True)
class CanonicalSplitConfig:
    output_dir: Path
    seed: int = 0
    clean_output_dir: bool = False
    pqal_source: str = "github"


def prepare_canonical_splits(
    config: CanonicalSplitConfig,
    source: CanonicalSourcePort,
) -> Path:
    if config.clean_output_dir and config.output_dir.exists():
        shutil.rmtree(config.output_dir)

    artificial = source.load_artificial()
    labeled = source.load_labeled()
    rng = random.Random(config.seed)
    artificial_pmids = list(artificial)
    rng.shuffle(artificial_pmids)
    artificial_train = OrderedDict(
        (pmid, artificial[pmid]) for pmid in artificial_pmids[:200_000]
    )
    artificial_validation = OrderedDict(
        (pmid, artificial[pmid]) for pmid in artificial_pmids[200_000:]
    )

    official_labels = source.load_official_test_labels()
    labeled_rng = random.Random(config.seed)
    simulated_cv, simulated_test = official_label_balanced_split(
        labeled, 2, labeled_rng
    )
    exact_official_split = False
    if official_labels is None:
        labeled_cv, labeled_test = simulated_cv, simulated_test
        exact_official_split = True
    else:
        official_test_pmids = [pmid for pmid in official_labels if pmid in labeled]
        missing_pmids = sorted(set(official_labels) - set(labeled))
        if missing_pmids:
            raise ValueError(
                f"{len(missing_pmids)} official PQA-L test PMIDs are missing from source"
            )
        if set(simulated_test) == set(official_test_pmids):
            labeled_cv, labeled_test = simulated_cv, simulated_test
            exact_official_split = True
        else:
            labeled_test = OrderedDict(
                (pmid, labeled[pmid]) for pmid in official_test_pmids
            )
            labeled_cv = OrderedDict(
                (pmid, row) for pmid, row in labeled.items() if pmid not in labeled_test
            )
            labeled_rng = random.Random(config.seed)

    cv_folds = official_label_balanced_split(labeled_cv, 10, labeled_rng)
    canonical_splits = {
        "pqa_artificial/train": annotate_rows(
            artificial_train,
            dataset_id="pqa_artificial",
            collection="PQA-A",
            source=HF_DATASET_URL,
            split="train",
            role="train",
        ),
        "pqa_artificial/validation": annotate_rows(
            artificial_validation,
            dataset_id="pqa_artificial",
            collection="PQA-A",
            source=HF_DATASET_URL,
            split="validation",
            role="validation",
        ),
        "pqa_labeled/cv": annotate_rows(
            labeled_cv,
            dataset_id="pqa_labeled",
            collection="PQA-L",
            source=source.labeled_source_url,
            split="cv",
            role="cv",
        ),
        "pqa_labeled/test": annotate_rows(
            labeled_test,
            dataset_id="pqa_labeled",
            collection="PQA-L",
            source=source.labeled_source_url,
            split="test",
            role="test",
        ),
    }
    for name, rows in canonical_splits.items():
        write_dataset(config.output_dir / f"{name}.json", rows)
        write_dataset_jsonl(config.output_dir / f"{name}.jsonl", rows)

    for index, validation in enumerate(cv_folds):
        fold_dir = config.output_dir / "pqa_labeled" / "folds" / f"fold_{index}"
        train = annotate_rows(
            combine_folds(cv_folds, index),
            dataset_id="pqa_labeled",
            collection="PQA-L",
            source=source.labeled_source_url,
            split=f"fold_{index}/train",
            role="train",
            fold=index,
        )
        validation_rows = annotate_rows(
            validation,
            dataset_id="pqa_labeled",
            collection="PQA-L",
            source=source.labeled_source_url,
            split=f"fold_{index}/validation",
            role="validation",
            fold=index,
        )
        write_dataset(fold_dir / "train.json", train)
        write_dataset(fold_dir / "validation.json", validation_rows)
        write_dataset_jsonl(fold_dir / "train.jsonl", train)
        write_dataset_jsonl(fold_dir / "validation.jsonl", validation_rows)

    metadata_path = config.output_dir / "metadata.json"
    write_json(
        metadata_path,
        {
            "source": {
                "huggingface_dataset": HF_DATASET,
                "pqal_source": config.pqal_source,
                "official_pqal": OFFICIAL_PQAL_URL,
                "pqa_artificial_examples": len(artificial),
                "pqa_labeled_examples": len(labeled),
                "official_pqal_test_ground_truth": OFFICIAL_PQAL_TEST_URL,
            },
            "seed": config.seed,
            "exact_official_pqal_split_reproduced": exact_official_split,
            "splits": {
                name: {
                    "path_json": f"{name}.json",
                    "path_jsonl": f"{name}.jsonl",
                    "dataset_id": next(iter(rows.values()))["dataset_id"],
                    "dataset_collection": next(iter(rows.values()))[
                        "dataset_collection"
                    ],
                    "examples": len(rows),
                    "label_counts": label_counts(rows),
                }
                for name, rows in canonical_splits.items()
            },
            "pqal_cv_folds": [
                {
                    "fold": index,
                    "path_jsonl_train": f"pqa_labeled/folds/fold_{index}/train.jsonl",
                    "path_jsonl_validation": f"pqa_labeled/folds/fold_{index}/validation.jsonl",
                    "train_examples": len(combine_folds(cv_folds, index)),
                    "validation_examples": len(validation),
                    "validation_label_counts": label_counts(validation),
                }
                for index, validation in enumerate(cv_folds)
            ],
        },
    )
    return metadata_path


@dataclass(frozen=True)
class PosttrainSplitConfig:
    output_dir: Path
    pqa_artificial_train_source: Path
    pqa_artificial_validation_source: Path
    pqa_labeled_test_source: Path
    artificial_train_size: int = 10_000
    artificial_validation_size: int = 2_000
    seed: int = 42


def prepare_posttrain_splits(config: PosttrainSplitConfig) -> Path:
    rng = random.Random(config.seed)
    artificial_rows = read_jsonl(config.pqa_artificial_train_source) + read_jsonl(
        config.pqa_artificial_validation_source
    )
    labeled_test_rows = read_jsonl(config.pqa_labeled_test_source)
    train_rows, validation_rows = split_balanced_artificial(
        artificial_rows,
        train_size=config.artificial_train_size,
        validation_size=config.artificial_validation_size,
        rng=rng,
    )

    artificial_train = annotate_rows(
        train_rows,
        dataset_id="pqa_artificial",
        collection="PQA-A",
        source=str(config.pqa_artificial_train_source.parent),
        split="posttrain/train",
        role="train",
    )
    artificial_validation = annotate_rows(
        validation_rows,
        dataset_id="pqa_artificial",
        collection="PQA-A",
        source=str(config.pqa_artificial_train_source.parent),
        split="posttrain/validation",
        role="validation",
    )
    labeled_test = annotate_rows(
        labeled_test_rows,
        dataset_id="pqa_labeled",
        collection="PQA-L",
        source=str(config.pqa_labeled_test_source),
        split="test",
        role="test",
    )
    outputs = {
        "pqa_artificial/train": artificial_train,
        "pqa_artificial/validation": artificial_validation,
        "pqa_labeled/test": labeled_test,
    }
    for name, rows in outputs.items():
        write_dataset(config.output_dir / f"{name}.json", rows)
        write_dataset_jsonl(config.output_dir / f"{name}.jsonl", rows)

    metadata_path = config.output_dir / "metadata.json"
    write_json(
        metadata_path,
        {
            "seed": config.seed,
            "policy": {
                "pqa_artificial_train_size": config.artificial_train_size,
                "pqa_artificial_validation_size": config.artificial_validation_size,
                "pqa_artificial_train_distribution": {
                    "yes": config.artificial_train_size // 2,
                    "no": config.artificial_train_size // 2,
                },
                "pqa_artificial_validation_distribution": {
                    "yes": config.artificial_validation_size // 2,
                    "no": config.artificial_validation_size // 2,
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
    return metadata_path
