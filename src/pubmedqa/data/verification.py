"""Cross-source consistency checks for PubMedQA releases."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from datasets import load_dataset

from pubmedqa.data.prepare import (
    HF_DATASET,
    OFFICIAL_PQAL_TEST_URL,
    OFFICIAL_PQAL_URL,
    label_counts,
)
from pubmedqa.data.sources import fetch_json


def normalize_prediction(value: Any) -> str:
    return "".join(value) if isinstance(value, list) else str(value)


def verify_sources() -> dict[str, Any]:
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
