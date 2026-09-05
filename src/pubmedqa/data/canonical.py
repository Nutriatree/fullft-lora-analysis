"""Application service for canonical PubMedQA source split preparation."""

from __future__ import annotations

import random
import shutil
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from pubmedqa.data.catalog import HF_DATASET, HF_DATASET_URL, OFFICIAL_PQAL_TEST_URL, OFFICIAL_PQAL_URL
from pubmedqa.data.io import write_dataset, write_dataset_jsonl
from pubmedqa.data.ports import CanonicalSourcePort
from pubmedqa.data.splits import (
    annotate_rows,
    combine_folds,
    label_counts,
    official_label_balanced_split,
)
from pubmedqa.runtime.io import write_json


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
    simulated_cv, simulated_test = official_label_balanced_split(labeled, 2, labeled_rng)
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
            labeled_test = OrderedDict((pmid, labeled[pmid]) for pmid in official_test_pmids)
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
                    "dataset_collection": next(iter(rows.values()))["dataset_collection"],
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
