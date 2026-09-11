from __future__ import annotations

import re
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
README_PATH = REPOSITORY_ROOT / "README.md"
THIRD_PARTY_NOTICES_PATH = REPOSITORY_ROOT / "THIRD_PARTY_NOTICES.md"
REPORT_PATH = REPOSITORY_ROOT / ".github/reports/Full-FT VS LoRA Report.pdf"
PROMPT_GUIDE_PATH = REPOSITORY_ROOT / ".github/guides/pubmedqa_prompt_parser.md"
EVALUATION_GUIDE_PATH = REPOSITORY_ROOT / ".github/guides/pubmedqa_evaluation.md"


class ReadmeContractTests(unittest.TestCase):
    def test_documents_physical_training_ownership_and_remaining_compatibility(self):
        architecture = (REPOSITORY_ROOT / ".github/ARCHITECTURE.md").read_text()
        for relative in (
            "train/distributed.py", "train/loop.py",
            "config/full_ft.py", "config/lora.py", "config/eval.py",
            "model/full_ft.py", "model/lora.py",
            "full_finetune.py", "lora_finetune.py",
        ):
            self.assertIn(relative, self.readme)
            self.assertIn(relative, architecture)
            self.assertTrue((REPOSITORY_ROOT / "src/pubmedqa" / relative).is_file())
        self.assertIn("기존 내부 호환 디렉토리는 제거", self.readme)
        self.assertIn("하위 패키지를 만들지", self.readme)
        for package in ("data", "model", "train", "eval", "config"):
            self.assertIn(f"{package}/", self.readme)
        guide = (
            REPOSITORY_ROOT / ".github/guides/pubmedqa_fsdp_troubleshooting.md"
        ).read_text()
        self.assertIn("train/distributed.py::TrainingSession", guide)
        # A substring assertion alone would accept a stale double-prefix path.
        for document in (self.readme, architecture, guide):
            for package in ("data", "model", "train", "eval", "config"):
                self.assertNotIn(f"{package}/{package}/", document)
        self.assertNotIn("runtime/training.py::TrainingSession", guide)

    def test_documents_functional_training_and_bounded_state_ownership(self) -> None:
        architecture = (REPOSITORY_ROOT / ".github/ARCHITECTURE.md").read_text()
        for expected in (
            "Modular Monolith",
            "Pipeline / Data-flow",
            "Conceptual Cohesion",
            "Locality",
            "Module-based OOP",
        ):
            self.assertIn(expected, self.readme)
            self.assertIn(expected, architecture)
        for expected in (
            "run_training",
            "TrainingSession",
            "EvaluationSettings",
            "RunFiles",
            "AdapterHistory",
            "compatibility",
        ):
            self.assertIn(expected, architecture)
        self.assertIn("src/pubmedqa/train/pipeline.py", self.readme)
        self.assertNotIn("기존 strategy override", self.readme)
        self.assertNotIn("Strategy classes bind extracted functions", architecture)

    def test_documents_distributed_fixes_and_metric_provenance(self) -> None:
        for expected in (
            "PUBMEDQA_CONTROL_TIMEOUT_SECONDS",
            "token_mean",
            "schema_version=2",
        ):
            self.assertIn(expected, self.readme)
        guide = (
            REPOSITORY_ROOT / ".github/guides/pubmedqa_fsdp_troubleshooting.md"
        ).read_text()
        for expected in (
            "float32",
            "CPU",
            "continue_on_error",
            "rank0_only",
            "optimizer.pt",
        ):
            self.assertIn(expected, guide)
        self.assertNotIn("Model state is gathered only on rank 0", guide)

    def test_documents_executable_offline_checks_and_training_owners(self) -> None:
        for path in (
            "scripts/test_pubmedqa_offline.py",
            "src/pubmedqa/train/loop.py",
            "src/pubmedqa/data/supervised.py",
            "src/pubmedqa/model/lora.py",
        ):
            self.assertIn(path, self.readme)
            self.assertTrue((REPOSITORY_ROOT / path).is_file())
        self.assertIn("ruff check", self.readme)
        self.assertIn("coverage run", self.readme)

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
        self.assertIn(".github/reports/Full-FT%20VS%20LoRA%20Report.pdf", self.readme)
        self.assertTrue(REPORT_PATH.is_file())
        self.assertIn(".github/ARCHITECTURE.md", self.readme)
        image_paths = re.findall(
            r"!\[[^\]]*\]\((\.github/assets/readme/[^)]+)\)", self.readme
        )
        self.assertGreaterEqual(len(image_paths), 3)
        for relative_path in image_paths:
            with self.subTest(path=relative_path):
                self.assertTrue((REPOSITORY_ROOT / relative_path).is_file())

    def test_private_docs_are_ignored_and_not_linked(self) -> None:
        gitignore = (REPOSITORY_ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("/docs/", gitignore.splitlines())
        self.assertNotIn("](docs/", self.readme)

    def test_superseded_combined_plot_cli_is_removed(self) -> None:
        self.assertFalse(
            (REPOSITORY_ROOT / "scripts/plot_pubmedqa_learning_curves.py").exists()
        )
        self.assertFalse(
            (
                REPOSITORY_ROOT
                / "src/pubmedqa/reporting/figures/all_learning_curves.py"
            ).exists()
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

    def test_documents_prompt_instruction_and_prediction_parsing(self) -> None:
        normalized_readme = " ".join(self.readme.split())
        for expected in (
            "hard constraint가 아니며",
            "생성된 응답을 parser로 처리",
            "최종 prediction label",
            "invalid prediction",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, normalized_readme)

        prompt_guide = PROMPT_GUIDE_PATH.read_text(encoding="utf-8")
        evaluation_guide = EVALUATION_GUIDE_PATH.read_text(encoding="utf-8")
        self.assertNotIn(
            "출력은 반드시 `yes`, `no`, `maybe` 중 하나여야 합니다", prompt_guide
        )
        self.assertIn("출력하도록 prompt에서 지시", prompt_guide)
        self.assertIn("direct-answer-only 형식을 유도", prompt_guide)
        self.assertIn("최종 prediction label", evaluation_guide)
        self.assertIn("invalid prediction", evaluation_guide)


if __name__ == "__main__":
    unittest.main()
