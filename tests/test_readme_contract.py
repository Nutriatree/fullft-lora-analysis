from __future__ import annotations

import re
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
README_PATH = REPOSITORY_ROOT / "README.md"
THIRD_PARTY_NOTICES_PATH = REPOSITORY_ROOT / "THIRD_PARTY_NOTICES.md"


class ReadmeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.readme = README_PATH.read_text(encoding="utf-8")

    def test_documents_current_dataset_policy(self) -> None:
        self.assertIn("PQA-A", self.readme)
        self.assertIn("5,000 / 5,000", self.readme)
        self.assertIn("1,000 / 1,000", self.readme)
        self.assertIn("PQA-L", self.readme)
        self.assertIn("500", self.readme)
        self.assertIn("test-only", self.readme)
        self.assertNotIn("200,000 train examples", self.readme)
        self.assertNotIn("PQA-L CV", self.readme)

    def test_documents_data_provenance_and_preprocessing(self) -> None:
        for expected in (
            "qiaojin/PubMedQA",
            "ori_pqal.json",
            "test_ground_truth.json",
            "class-wise",
            "Seed 42",
            "disjoint",
            "metadata.json",
            "describe --strict",
            "verify-sources",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, self.readme)

    def test_documents_supported_execution_modes_and_entrypoints(self) -> None:
        for mode in ("single", "ddp", "fsdp"):
            self.assertIn(f"PUBMEDQA_DISTRIBUTED_MODE={mode}", self.readme)
        self.assertIn("scripts/run_pubmedqa_full_study.sh", self.readme)
        self.assertIn("scripts/prepare_pubmedqa_posttrain_splits.py", self.readme)
        self.assertIn("scripts/plot_pubmedqa_rq_learning_curves.py", self.readme)

    def test_links_final_report_and_existing_readme_images(self) -> None:
        self.assertIn(".github/reports/PubMedQA%20Fine-Tune%20Report.pdf", self.readme)
        self.assertIn(".github/ARCHITECTURE.md", self.readme)
        image_paths = re.findall(r"!\[[^\]]*\]\((\.github/assets/readme/[^)]+)\)", self.readme)
        self.assertGreaterEqual(len(image_paths), 3)
        for relative_path in image_paths:
            with self.subTest(path=relative_path):
                self.assertTrue((REPOSITORY_ROOT / relative_path).is_file())

    def test_private_docs_are_ignored_and_not_linked(self) -> None:
        gitignore = (REPOSITORY_ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("/docs/", gitignore.splitlines())
        self.assertNotIn("](docs/", self.readme)

    def test_superseded_combined_plot_cli_is_removed(self) -> None:
        self.assertFalse((REPOSITORY_ROOT / "scripts/plot_pubmedqa_learning_curves.py").exists())
        self.assertFalse(
            (REPOSITORY_ROOT / "src/pubmedqa/reporting/figures/all_learning_curves.py").exists()
        )

    def test_explains_artifact_boundary(self) -> None:
        self.assertIn("outputs/", self.readme)
        self.assertIn("reports/", self.readme)
        self.assertIn("checkpoints/", self.readme)

    def test_uses_dataset_independent_project_name(self) -> None:
        self.assertIn("# Full Fine-Tuning vs. LoRA Analysis", self.readme)
        pyproject = (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('name = "fullft-lora-analysis"', pyproject)

    def test_cites_and_attributes_pubmedqa(self) -> None:
        for expected in (
            "https://pubmedqa.github.io/",
            "https://github.com/pubmedqa/pubmedqa",
            "## Citation",
            "jin2019pubmedqa",
            "not affiliated with or endorsed by",
            "THIRD_PARTY_NOTICES.md",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, self.readme)

        notices = THIRD_PARTY_NOTICES_PATH.read_text(encoding="utf-8")
        self.assertIn("Copyright (c) 2019 pubmedqa", notices)
        self.assertIn("MIT License", notices)
        self.assertIn("https://github.com/pubmedqa/pubmedqa", notices)

    def test_row_level_dataset_artifacts_are_ignored(self) -> None:
        gitignore = (REPOSITORY_ROOT / ".gitignore").read_text(encoding="utf-8")
        for expected in (
            "/outputs/**/outputs.jsonl",
            "/outputs/**/evaluations/*_predictions.jsonl",
            "/outputs/**/prediction_transitions/*.jsonl",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, gitignore.splitlines())


if __name__ == "__main__":
    unittest.main()
