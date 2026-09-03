#!/usr/bin/env python3
"""PubMedQA data utility CLI.

Subcommands:
- prepare: create reproducible PQA-A/PQA-L split files.
- verify-sources: compare Hugging Face PQA-L with official GitHub PQA-L files.
- describe: summarize generated split files and validate PQA-A/PQA-L separation.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import urllib.request
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset


LABELS = ("yes", "no", "maybe")
HF_DATASET = "qiaojin/PubMedQA"
HF_DATASET_URL = "https://huggingface.co/datasets/qiaojin/PubMedQA"
OFFICIAL_PQAL_URL = "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/ori_pqal.json"
OFFICIAL_PQAL_TEST_URL = (
    "https://raw.githubusercontent.com/pubmedqa/pubmedqa/master/data/test_ground_truth.json"
)
EXPECTED_DATASET_ID_BY_PATH = {
    "pqa_artificial": "pqa_artificial",
    "pqa_labeled": "pqa_labeled",
}


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


def fetch_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read().decode("utf-8"))


def load_hf_config(config_name: str) -> "OrderedDict[str, dict[str, Any]]":
    ds = load_dataset(HF_DATASET, config_name, split="train")
    return rows_to_ordered_dict(ds, config_name)


def rows_to_ordered_dict(ds: Dataset, dataset_id: str) -> "OrderedDict[str, dict[str, Any]]":
    output: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
    for row in ds:
        pubid = str(row["pubid"])
        output[pubid] = {
            "pubid": pubid,
            "dataset_id": dataset_id,
            "question": row["question"],
            "context": row["context"],
            "long_answer": row.get("long_answer"),
            "final_decision": row.get("final_decision"),
        }
    return output


def load_github_pqal() -> "OrderedDict[str, dict[str, Any]]":
    rows: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
    for pmid, row in fetch_json(OFFICIAL_PQAL_URL).items():
        rows[str(pmid)] = {
            "pubid": str(pmid),
            "dataset_id": "pqa_labeled",
            "question": row["QUESTION"],
            "context": {
                "contexts": row["CONTEXTS"],
                "labels": row["LABELS"],
                "meshes": row["MESHES"],
                "reasoning_required_pred": row["reasoning_required_pred"],
                "reasoning_free_pred": row["reasoning_free_pred"],
            },
            "year": row.get("YEAR"),
            "long_answer": row["LONG_ANSWER"],
            "final_decision": row["final_decision"],
        }
    return rows


def load_official_pqal_test_labels(path: Path | None, skip_download: bool) -> dict[str, str] | None:
    if path is not None:
        with path.open() as f:
            return json.load(f)
    if skip_download:
        return None
    return fetch_json(OFFICIAL_PQAL_TEST_URL)


def split_label(pmids: list[str], fold: int, rng: random.Random) -> list[list[str]]:
    rng.shuffle(pmids)
    num_split = math.ceil(len(pmids) / fold)
    output: list[list[str]] = []
    for i in range(fold):
        if i == fold - 1:
            output.append(pmids[i * num_split :])
        else:
            output.append(pmids[i * num_split : (i + 1) * num_split])
    return output


def official_label_balanced_split(
    dataset: "OrderedDict[str, dict[str, Any]]", fold: int, rng: random.Random
) -> list["OrderedDict[str, dict[str, Any]]"]:
    label2pmid: dict[str, list[str]] = {label: [] for label in LABELS}
    for pmid, row in dataset.items():
        label = row["final_decision"]
        if label not in label2pmid:
            raise ValueError(f"Unexpected final_decision={label!r} for PMID {pmid}")
        label2pmid[label].append(pmid)

    split_by_label = {
        label: split_label(pmids, fold, rng) for label, pmids in label2pmid.items()
    }

    output: list["OrderedDict[str, dict[str, Any]]"] = []
    for i in range(fold):
        fold_pmids: list[str] = []
        for label in LABELS:
            fold_pmids.extend(split_by_label[label][i])
        output.append(OrderedDict((pmid, dataset[pmid]) for pmid in fold_pmids))

    if len(output[-1]) != len(output[0]):
        for i in range(fold - 1):
            picked = rng.choice(list(output[i]))
            output[-1][picked] = output[i][picked]
            output[i].pop(picked)

    return output


def with_split_metadata(
    rows: "OrderedDict[str, dict[str, Any]]",
    *,
    dataset_id: str,
    collection: str,
    source: str,
    split: str,
    role: str,
    fold: int | None = None,
) -> "OrderedDict[str, dict[str, Any]]":
    annotated: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
    for pmid, row in rows.items():
        item = dict(row)
        item["dataset_id"] = dataset_id
        item["dataset_collection"] = collection
        item["source"] = source
        item["split"] = split
        item["split_role"] = role
        if fold is not None:
            item["fold"] = fold
        annotated[pmid] = item
    return annotated


def combine_folds(
    cv_folds: list["OrderedDict[str, dict[str, Any]]"], validation_fold: int
) -> "OrderedDict[str, dict[str, Any]]":
    combined: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
    for i, fold in enumerate(cv_folds):
        if i != validation_fold:
            combined.update(fold)
    return combined


def label_counts(rows: "OrderedDict[str, dict[str, Any]]") -> dict[str, int]:
    return dict(Counter(row["final_decision"] for row in rows.values()))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def write_jsonl(path: Path, rows: "OrderedDict[str, dict[str, Any]]") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows.values():
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def command_prepare(args: argparse.Namespace) -> None:
    output_dir: Path = args.output_dir
    if args.clean_output_dir and output_dir.exists():
        shutil.rmtree(output_dir)

    pqaa = load_hf_config("pqa_artificial")
    pqal = load_github_pqal() if args.pqal_source == "github" else load_hf_config("pqa_labeled")

    rng = random.Random(args.seed)
    pqaa_pmids = list(pqaa)
    rng.shuffle(pqaa_pmids)
    pqaa_train = OrderedDict((pmid, pqaa[pmid]) for pmid in pqaa_pmids[:200_000])
    pqaa_validation = OrderedDict((pmid, pqaa[pmid]) for pmid in pqaa_pmids[200_000:])

    official_test_labels = load_official_pqal_test_labels(
        args.pqal_test_ground_truth, args.skip_github_test_download
    )
    pqal_rng = random.Random(args.seed)
    simulated_pqal_cv, simulated_pqal_test = official_label_balanced_split(pqal, 2, pqal_rng)
    exact_official_pqal_split = False
    if official_test_labels is None:
        pqal_cv, pqal_test = simulated_pqal_cv, simulated_pqal_test
        exact_official_pqal_split = True
    else:
        official_test_pmids = [pmid for pmid in official_test_labels if pmid in pqal]
        missing_pmids = sorted(set(official_test_labels) - set(pqal))
        if missing_pmids:
            raise ValueError(f"{len(missing_pmids)} official PQA-L test PMIDs are missing from source")
        if set(simulated_pqal_test) == set(official_test_pmids):
            pqal_cv, pqal_test = simulated_pqal_cv, simulated_pqal_test
            exact_official_pqal_split = True
        else:
            pqal_test = OrderedDict((pmid, pqal[pmid]) for pmid in official_test_pmids)
            pqal_cv = OrderedDict((pmid, row) for pmid, row in pqal.items() if pmid not in pqal_test)
            pqal_rng = random.Random(args.seed)

    cv_folds = official_label_balanced_split(pqal_cv, 10, pqal_rng)
    pqal_source_url = OFFICIAL_PQAL_URL if args.pqal_source == "github" else HF_DATASET_URL

    canonical_splits = {
        "pqa_artificial/train": with_split_metadata(
            pqaa_train,
            dataset_id="pqa_artificial",
            collection="PQA-A",
            source=HF_DATASET_URL,
            split="train",
            role="train",
        ),
        "pqa_artificial/validation": with_split_metadata(
            pqaa_validation,
            dataset_id="pqa_artificial",
            collection="PQA-A",
            source=HF_DATASET_URL,
            split="validation",
            role="validation",
        ),
        "pqa_labeled/cv": with_split_metadata(
            pqal_cv,
            dataset_id="pqa_labeled",
            collection="PQA-L",
            source=pqal_source_url,
            split="cv",
            role="cv",
        ),
        "pqa_labeled/test": with_split_metadata(
            pqal_test,
            dataset_id="pqa_labeled",
            collection="PQA-L",
            source=pqal_source_url,
            split="test",
            role="test",
        ),
    }
    for name, rows in canonical_splits.items():
        write_json(output_dir / f"{name}.json", rows)
        write_jsonl(output_dir / f"{name}.jsonl", rows)

    for i, validation in enumerate(cv_folds):
        fold_dir = output_dir / "pqa_labeled" / "folds" / f"fold_{i}"
        train = with_split_metadata(
            combine_folds(cv_folds, i),
            dataset_id="pqa_labeled",
            collection="PQA-L",
            source=pqal_source_url,
            split=f"fold_{i}/train",
            role="train",
            fold=i,
        )
        validation = with_split_metadata(
            validation,
            dataset_id="pqa_labeled",
            collection="PQA-L",
            source=pqal_source_url,
            split=f"fold_{i}/validation",
            role="validation",
            fold=i,
        )
        write_json(fold_dir / "train.json", train)
        write_json(fold_dir / "validation.json", validation)
        write_jsonl(fold_dir / "train.jsonl", train)
        write_jsonl(fold_dir / "validation.jsonl", validation)

    metadata = {
        "source": {
            "huggingface_dataset": HF_DATASET,
            "pqal_source": args.pqal_source,
            "official_pqal": OFFICIAL_PQAL_URL,
            "pqa_artificial_examples": len(pqaa),
            "pqa_labeled_examples": len(pqal),
            "official_pqal_test_ground_truth": OFFICIAL_PQAL_TEST_URL,
        },
        "seed": args.seed,
        "exact_official_pqal_split_reproduced": exact_official_pqal_split,
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
                "fold": i,
                "path_jsonl_train": f"pqa_labeled/folds/fold_{i}/train.jsonl",
                "path_jsonl_validation": f"pqa_labeled/folds/fold_{i}/validation.jsonl",
                "train_examples": len(combine_folds(cv_folds, i)),
                "validation_examples": len(validation),
                "validation_label_counts": label_counts(validation),
            }
            for i, validation in enumerate(cv_folds)
        ],
    }
    write_json(output_dir / "metadata.json", metadata)


def normalize_prediction(value: Any) -> str:
    if isinstance(value, list):
        return "".join(value)
    return str(value)


def command_verify_sources(_: argparse.Namespace) -> None:
    hf_pqal = load_dataset(HF_DATASET, "pqa_labeled", split="train")
    hf_pqaa = load_dataset(HF_DATASET, "pqa_artificial", split="train")
    hf_pqal_by_pmid = OrderedDict((str(row["pubid"]), dict(row)) for row in hf_pqal)
    hf_pqaa_by_pmid = OrderedDict((str(row["pubid"]), dict(row)) for row in hf_pqaa)
    gh_pqal = fetch_json(OFFICIAL_PQAL_URL)
    gh_test_gt = fetch_json(OFFICIAL_PQAL_TEST_URL)

    hf_pmids = set(hf_pqal_by_pmid)
    gh_pmids = set(gh_pqal)
    test_pmids = set(gh_test_gt)

    field_mismatches: list[tuple[str, str]] = []
    for pmid in sorted(hf_pmids & gh_pmids):
        hf_row = hf_pqal_by_pmid[pmid]
        gh_row = gh_pqal[pmid]
        checks = {
            "question": hf_row.get("question") == gh_row.get("QUESTION"),
            "long_answer": hf_row.get("long_answer") == gh_row.get("LONG_ANSWER"),
            "final_decision": hf_row.get("final_decision") == gh_row.get("final_decision"),
            "contexts": hf_row.get("context", {}).get("contexts") == gh_row.get("CONTEXTS"),
            "labels": hf_row.get("context", {}).get("labels") == gh_row.get("LABELS"),
            "meshes": hf_row.get("context", {}).get("meshes") == gh_row.get("MESHES"),
            "reasoning_required_pred": normalize_prediction(
                hf_row.get("context", {}).get("reasoning_required_pred")
            )
            == gh_row.get("reasoning_required_pred"),
            "reasoning_free_pred": normalize_prediction(
                hf_row.get("context", {}).get("reasoning_free_pred")
            )
            == gh_row.get("reasoning_free_pred"),
        }
        for field, ok in checks.items():
            if not ok:
                field_mismatches.append((pmid, field))

    label_mismatches = [
        pmid
        for pmid in sorted(test_pmids & hf_pmids)
        if hf_pqal_by_pmid[pmid].get("final_decision") != gh_test_gt[pmid]
    ]

    print("HF pqa_labeled examples:", len(hf_pqal_by_pmid))
    print("GitHub ori_pqal examples:", len(gh_pqal))
    print("PQA-L PMID sets equal:", hf_pmids == gh_pmids)
    print("Only in HF pqa_labeled:", len(hf_pmids - gh_pmids))
    print("Only in GitHub ori_pqal:", len(gh_pmids - hf_pmids))
    print("HF pqa_labeled label counts:", label_counts(hf_pqal_by_pmid))
    print("GitHub ori_pqal label counts:", label_counts(gh_pqal))
    print("Question/context/answer/label mismatches:", len(field_mismatches))
    if field_mismatches:
        print("First mismatches:", field_mismatches[:10])
    print("Official GitHub PQA-L test PMIDs:", len(test_pmids))
    print("Official test PMIDs all present in HF:", test_pmids <= hf_pmids)
    print("Official test label mismatches vs HF:", len(label_mismatches))
    if label_mismatches:
        print("First label mismatches:", label_mismatches[:10])
    print("HF pqa_artificial examples:", len(hf_pqaa_by_pmid))
    print("HF pqa_artificial label counts:", label_counts(hf_pqaa_by_pmid))


def read_jsonl_summary(path: Path) -> dict[str, Any]:
    dataset_ids: Counter[str] = Counter()
    collections: Counter[str] = Counter()
    roles: Counter[str] = Counter()
    splits: Counter[str] = Counter()
    labels: Counter[str] = Counter()
    examples = 0

    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            examples += 1
            dataset_ids[str(row.get("dataset_id"))] += 1
            collections[str(row.get("dataset_collection"))] += 1
            roles[str(row.get("split_role"))] += 1
            splits[str(row.get("split"))] += 1
            labels[str(row.get("final_decision"))] += 1
            if row.get("pubid") is None:
                raise ValueError(f"{path}:{line_no} is missing pubid")

    return {
        "path": str(path),
        "examples": examples,
        "dataset_ids": dict(dataset_ids),
        "dataset_collections": dict(collections),
        "split_roles": dict(roles),
        "splits": dict(splits),
        "label_counts": dict(labels),
    }


def expected_dataset_id(path: Path) -> str | None:
    for part in path.parts:
        if part in EXPECTED_DATASET_ID_BY_PATH:
            return EXPECTED_DATASET_ID_BY_PATH[part]
    return None


def validate_summary(summary: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    path = Path(summary["path"])
    dataset_ids = summary["dataset_ids"]

    if len(dataset_ids) != 1:
        errors.append(f"{path} has mixed dataset_id values: {dataset_ids}")

    expected = expected_dataset_id(path)
    if expected is not None and dataset_ids != {expected: summary["examples"]}:
        errors.append(f"{path} expected dataset_id={expected}, got {dataset_ids}")

    if "pqa_artificial" in dataset_ids and "maybe" in summary["label_counts"]:
        errors.append(f"{path} is PQA-A but contains maybe labels")

    return errors


def command_describe(args: argparse.Namespace) -> None:
    jsonl_files = sorted((args.data_dir / "pqa_artificial").rglob("*.jsonl"))
    jsonl_files.extend(sorted((args.data_dir / "pqa_labeled").rglob("*.jsonl")))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found under {args.data_dir}")

    all_errors: list[str] = []
    summaries = [read_jsonl_summary(path) for path in jsonl_files]
    for summary in summaries:
        all_errors.extend(validate_summary(summary))
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
