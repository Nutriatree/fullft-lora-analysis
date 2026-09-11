"""External Hugging Face and official-GitHub PubMedQA source adapters."""

from __future__ import annotations

import json
import urllib.request
from collections import OrderedDict
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset

from pubmedqa.data.prepare import (
    HF_DATASET,
    HF_DATASET_URL,
    OFFICIAL_PQAL_TEST_URL,
    OFFICIAL_PQAL_URL,
    CanonicalSourcePort,
    label_counts,
)


def fetch_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read().decode("utf-8"))


def rows_to_ordered_dict(
    dataset: Dataset,
    dataset_id: str,
) -> OrderedDict[str, dict[str, Any]]:
    output: OrderedDict[str, dict[str, Any]] = OrderedDict()
    for row in dataset:
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


def load_hf_config(config_name: str) -> OrderedDict[str, dict[str, Any]]:
    dataset = load_dataset(HF_DATASET, config_name, split="train")
    return rows_to_ordered_dict(dataset, config_name)


def load_github_pqal() -> OrderedDict[str, dict[str, Any]]:
    rows: OrderedDict[str, dict[str, Any]] = OrderedDict()
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


def load_official_pqal_test_labels(
    path: Path | None,
    skip_download: bool,
) -> dict[str, str] | None:
    if path is not None:
        return json.loads(path.read_text(encoding="utf-8"))
    if skip_download:
        return None
    return fetch_json(OFFICIAL_PQAL_TEST_URL)


class RemotePubMedQASource(CanonicalSourcePort):
    """Hugging Face/GitHub implementation of the canonical source port."""

    def __init__(
        self,
        *,
        pqal_source: str,
        test_ground_truth_path: Path | None,
        skip_test_download: bool,
    ) -> None:
        if pqal_source not in {"github", "hf"}:
            raise ValueError("pqal_source must be 'github' or 'hf'")
        self.pqal_source = pqal_source
        self.test_ground_truth_path = test_ground_truth_path
        self.skip_test_download = skip_test_download

    @property
    def labeled_source_url(self) -> str:
        return OFFICIAL_PQAL_URL if self.pqal_source == "github" else HF_DATASET_URL

    def load_artificial(self) -> OrderedDict[str, dict[str, Any]]:
        return load_hf_config("pqa_artificial")

    def load_labeled(self) -> OrderedDict[str, dict[str, Any]]:
        return (
            load_github_pqal()
            if self.pqal_source == "github"
            else load_hf_config("pqa_labeled")
        )

    def load_official_test_labels(self) -> dict[str, str] | None:
        return load_official_pqal_test_labels(
            self.test_ground_truth_path,
            self.skip_test_download,
        )


def normalize_prediction(value: Any) -> str:
    return "".join(value) if isinstance(value, list) else str(value)


def verify_sources() -> dict[str, Any]:
    """Compare Hugging Face rows with the official GitHub release."""
    hf_labeled = load_dataset(HF_DATASET, "pqa_labeled", split="train")
    hf_artificial = load_dataset(HF_DATASET, "pqa_artificial", split="train")
    hf_labeled_by_pmid = OrderedDict(
        (str(row["pubid"]), dict(row)) for row in hf_labeled
    )
    hf_artificial_by_pmid = OrderedDict(
        (str(row["pubid"]), dict(row)) for row in hf_artificial
    )
    github_labeled = fetch_json(OFFICIAL_PQAL_URL)
    github_test = fetch_json(OFFICIAL_PQAL_TEST_URL)

    hf_pmids = set(hf_labeled_by_pmid)
    github_pmids = set(github_labeled)
    test_pmids = set(github_test)
    field_mismatches: list[tuple[str, str]] = []
    for pmid in sorted(hf_pmids & github_pmids):
        hf_row = hf_labeled_by_pmid[pmid]
        github_row = github_labeled[pmid]
        checks = {
            "question": hf_row.get("question") == github_row.get("QUESTION"),
            "long_answer": hf_row.get("long_answer") == github_row.get("LONG_ANSWER"),
            "final_decision": hf_row.get("final_decision")
            == github_row.get("final_decision"),
            "contexts": hf_row.get("context", {}).get("contexts")
            == github_row.get("CONTEXTS"),
            "labels": hf_row.get("context", {}).get("labels")
            == github_row.get("LABELS"),
            "meshes": hf_row.get("context", {}).get("meshes")
            == github_row.get("MESHES"),
            "reasoning_required_pred": normalize_prediction(
                hf_row.get("context", {}).get("reasoning_required_pred")
            )
            == github_row.get("reasoning_required_pred"),
            "reasoning_free_pred": normalize_prediction(
                hf_row.get("context", {}).get("reasoning_free_pred")
            )
            == github_row.get("reasoning_free_pred"),
        }
        field_mismatches.extend(
            (pmid, field) for field, matches in checks.items() if not matches
        )

    label_mismatches = [
        pmid
        for pmid in sorted(test_pmids & hf_pmids)
        if hf_labeled_by_pmid[pmid].get("final_decision") != github_test[pmid]
    ]
    return {
        "hf_labeled_examples": len(hf_labeled_by_pmid),
        "github_labeled_examples": len(github_labeled),
        "pmid_sets_equal": hf_pmids == github_pmids,
        "only_in_hf": len(hf_pmids - github_pmids),
        "only_in_github": len(github_pmids - hf_pmids),
        "hf_labeled_label_counts": label_counts(hf_labeled_by_pmid),
        "github_labeled_label_counts": label_counts(github_labeled),
        "field_mismatches": field_mismatches,
        "official_test_pmids": len(test_pmids),
        "official_test_all_in_hf": test_pmids <= hf_pmids,
        "official_test_label_mismatches": label_mismatches,
        "hf_artificial_examples": len(hf_artificial_by_pmid),
        "hf_artificial_label_counts": label_counts(hf_artificial_by_pmid),
    }
