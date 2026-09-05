"""Application service for the balanced PQA-A/PQA-L post-training dataset."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from pubmedqa.data.io import read_jsonl, write_dataset, write_dataset_jsonl
from pubmedqa.data.splits import annotate_rows, label_counts, split_balanced_artificial
from pubmedqa.runtime.io import write_json


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
