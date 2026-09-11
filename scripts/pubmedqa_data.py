#!/usr/bin/env python3
"""PubMedQA data utility CLI.

Subcommands:
- prepare: create reproducible PQA-A/PQA-L split files.
- verify-sources: compare Hugging Face PQA-L with official GitHub PQA-L files.
- describe: summarize generated split files and validate PQA-A/PQA-L separation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pubmedqa.data.prepare import CanonicalSplitConfig, prepare_canonical_splits
from pubmedqa.data.sources import RemotePubMedQASource
from pubmedqa.data.summary import describe_directory
from pubmedqa.data.verification import verify_sources


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Create PubMedQA split files.")
    prepare.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    prepare.add_argument("--seed", type=int, default=0)
    prepare.add_argument(
        "--clean-output-dir",
        action="store_true",
        help="Remove the output directory before writing generated split files.",
    )
    prepare.add_argument(
        "--pqal-test-ground-truth",
        type=Path,
        default=None,
        help=(
            "Optional local copy of PubMedQA data/test_ground_truth.json. "
            "When omitted, the official GitHub file is downloaded."
        ),
    )
    prepare.add_argument(
        "--pqal-source",
        choices=("github", "hf"),
        default="github",
        help="Use the official GitHub PQA-L file by default; HF is available for comparison.",
    )
    prepare.add_argument(
        "--skip-github-test-download",
        action="store_true",
        help="Recreate the PQA-L 500/500 split with the official seed if GitHub access is unavailable.",
    )

    describe = subparsers.add_parser("describe", help="Summarize generated PubMedQA split files.")
    describe.add_argument("--data-dir", type=Path, default=Path("data/processed"))
    describe.add_argument(
        "--strict",
        action="store_true",
        help="Exit with an error if any JSONL file has mixed or path-inconsistent dataset_id values.",
    )

    subparsers.add_parser(
        "verify-sources",
        help="Compare HF pqa_labeled with official GitHub ori_pqal/test_ground_truth files.",
    )

    return parser.parse_args()


def command_prepare(args: argparse.Namespace) -> None:
    source = RemotePubMedQASource(
        pqal_source=args.pqal_source,
        test_ground_truth_path=args.pqal_test_ground_truth,
        skip_test_download=args.skip_github_test_download,
    )
    metadata_path = prepare_canonical_splits(
        CanonicalSplitConfig(
            output_dir=args.output_dir,
            seed=args.seed,
            clean_output_dir=args.clean_output_dir,
            pqal_source=args.pqal_source,
        ),
        source,
    )
    print(f"Prepared canonical splits: {metadata_path}")


def command_verify_sources(_: argparse.Namespace) -> None:
    report = verify_sources()
    print("HF pqa_labeled examples:", report["hf_labeled_examples"])
    print("GitHub ori_pqal examples:", report["github_labeled_examples"])
    print("PQA-L PMID sets equal:", report["pmid_sets_equal"])
    print("Only in HF pqa_labeled:", report["only_in_hf"])
    print("Only in GitHub ori_pqal:", report["only_in_github"])
    print("HF pqa_labeled label counts:", report["hf_labeled_label_counts"])
    print("GitHub ori_pqal label counts:", report["github_labeled_label_counts"])
    print("Question/context/answer/label mismatches:", len(report["field_mismatches"]))
    if report["field_mismatches"]:
        print("First mismatches:", report["field_mismatches"][:10])
    print("Official GitHub PQA-L test PMIDs:", report["official_test_pmids"])
    print("Official test PMIDs all present in HF:", report["official_test_all_in_hf"])
    print("Official test label mismatches vs HF:", len(report["official_test_label_mismatches"]))
    if report["official_test_label_mismatches"]:
        print("First label mismatches:", report["official_test_label_mismatches"][:10])
    print("HF pqa_artificial examples:", report["hf_artificial_examples"])
    print("HF pqa_artificial label counts:", report["hf_artificial_label_counts"])


def command_describe(args: argparse.Namespace) -> None:
    summaries, all_errors = describe_directory(args.data_dir)
    for summary in summaries:
        print(
            f"{summary['path']}: examples={summary['examples']} "
            f"dataset_id={summary['dataset_ids']} role={summary['split_roles']} "
            f"labels={summary['label_counts']}"
        )

    if all_errors:
        print("\nValidation errors:")
        for error in all_errors:
            print(f"- {error}")
        if args.strict:
            raise SystemExit(1)


def main() -> None:
    args = parse_args()
    if args.command == "prepare":
        command_prepare(args)
    elif args.command == "verify-sources":
        command_verify_sources(args)
    elif args.command == "describe":
        command_describe(args)
    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
