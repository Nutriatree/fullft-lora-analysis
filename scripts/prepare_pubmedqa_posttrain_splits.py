#!/usr/bin/env python3
"""Prepare balanced post-training splits for PubMedQA."""

from __future__ import annotations

import argparse
from pathlib import Path

from pubmedqa.data.posttrain import PosttrainSplitConfig, prepare_posttrain_splits


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


def main() -> None:
    args = parse_args()
    metadata_path = prepare_posttrain_splits(
        PosttrainSplitConfig(
            output_dir=args.output_dir,
            pqa_artificial_train_source=args.pqa_artificial_train_source,
            pqa_artificial_validation_source=args.pqa_artificial_validation_source,
            pqa_labeled_test_source=args.pqa_labeled_test_source,
            artificial_train_size=args.artificial_train_size,
            artificial_validation_size=args.artificial_validation_size,
            seed=args.seed,
        )
    )
    print(f"Prepared post-training splits: {metadata_path}")


if __name__ == "__main__":
    main()
