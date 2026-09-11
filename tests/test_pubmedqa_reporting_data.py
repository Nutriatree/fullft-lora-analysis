from __future__ import annotations

import json
import subprocess
import sys
import random
import tempfile
import unittest
from pathlib import Path

from pubmedqa.data.prepare import PosttrainSplitConfig, prepare_posttrain_splits
from pubmedqa.data.prepare import split_balanced_artificial
from pubmedqa.data.summary import read_jsonl_summary
from pubmedqa.eval.reports import StudyLayout
from pubmedqa.eval.reports import write_report_manifest
from pubmedqa.eval.reports import StudyArtifactReader


class PubMedQAReportingDataTest(unittest.TestCase):
    def test_report_layout_is_separate_from_experiment_outputs(self) -> None:
        layout = StudyLayout.from_study_dir(
            Path("outputs/pubmedqa_train/study_single_epoch1_used")
        )

        self.assertEqual("study_single_epoch1_used", layout.study_id)
        self.assertEqual(
            Path("reports/pubmedqa/study_single_epoch1_used/figures/main"),
            layout.main_figures_dir,
        )
        self.assertEqual(
            Path("reports/pubmedqa/study_single_epoch1_used/figures/appendix"),
            layout.appendix_figures_dir,
        )

    def test_artifact_reader_reports_missing_and_malformed_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reader = StudyArtifactReader(root)
            with self.assertRaisesRegex(FileNotFoundError, "Study artifact not found"):
                reader.read_json(Path("missing.json"))

            malformed = root / "bad.jsonl"
            malformed.write_text('{"ok": 1}\nnot-json\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "bad.jsonl:2"):
                reader.read_jsonl(Path("bad.jsonl"))

    def test_artifact_reader_validates_expected_run_directories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir = root / "Qwen_Qwen3-1.7B"
            (model_dir / "full-ft").mkdir(parents=True)
            reader = StudyArtifactReader(root)

            with self.assertRaisesRegex(FileNotFoundError, "lora"):
                reader.require_run_directories(
                    "Qwen_Qwen3-1.7B",
                    ("full-ft", "lora"),
                )

    def test_balanced_split_is_seeded_disjoint_and_balanced(self) -> None:
        rows = [
            {"pubid": f"yes-{index}", "final_decision": "yes"} for index in range(8)
        ] + [{"pubid": f"no-{index}", "final_decision": "no"} for index in range(8)]

        first = split_balanced_artificial(
            rows,
            train_size=8,
            validation_size=4,
            rng=random.Random(42),
        )
        second = split_balanced_artificial(
            rows,
            train_size=8,
            validation_size=4,
            rng=random.Random(42),
        )

        self.assertEqual(first, second)
        train, validation = first
        self.assertFalse(
            {row["pubid"] for row in train} & {row["pubid"] for row in validation}
        )
        self.assertEqual({"yes": 4, "no": 4}, _counts(train))
        self.assertEqual({"yes": 2, "no": 2}, _counts(validation))

    def test_report_manifest_records_input_and_derived_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            study_dir = root / "outputs" / "study"
            figure_dir = root / "reports" / "study" / "figures" / "main"
            figure_dir.mkdir(parents=True)
            (figure_dir / "figure.png").write_bytes(b"png")
            layout = StudyLayout(
                study_dir=study_dir,
                report_root=root / "reports" / "study",
            )

            payload = json.loads(
                write_report_manifest(layout).read_text(encoding="utf-8")
            )

            self.assertEqual(str(study_dir), payload["input_study_dir"])
            self.assertEqual(["figures/main/figure.png"], payload["figure_files"])

    def test_posttrain_service_writes_disjoint_balanced_splits_and_metadata(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artificial_train = root / "source" / "train.jsonl"
            artificial_validation = root / "source" / "validation.jsonl"
            labeled_test = root / "labeled" / "test.jsonl"
            artificial_train.parent.mkdir(parents=True)
            labeled_test.parent.mkdir(parents=True)
            artificial = [
                {"pubid": f"{label}-{index}", "final_decision": label}
                for label in ("yes", "no")
                for index in range(8)
            ]
            artificial_train.write_text(
                "".join(json.dumps(row) + "\n" for row in artificial[:8]),
                encoding="utf-8",
            )
            artificial_validation.write_text(
                "".join(json.dumps(row) + "\n" for row in artificial[8:]),
                encoding="utf-8",
            )
            labeled_test.write_text(
                json.dumps({"pubid": "test-1", "final_decision": "maybe"}) + "\n",
                encoding="utf-8",
            )
            output_dir = root / "posttrain"

            metadata_path = prepare_posttrain_splits(
                PosttrainSplitConfig(
                    output_dir=output_dir,
                    pqa_artificial_train_source=artificial_train,
                    pqa_artificial_validation_source=artificial_validation,
                    pqa_labeled_test_source=labeled_test,
                    artificial_train_size=8,
                    artificial_validation_size=4,
                    seed=42,
                )
            )

            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            self.assertEqual(
                {"yes": 4, "no": 4},
                metadata["splits"]["pqa_artificial/train"]["label_counts"],
            )
            self.assertEqual(
                {"yes": 2, "no": 2},
                metadata["splits"]["pqa_artificial/validation"]["label_counts"],
            )
            self.assertFalse(metadata["policy"]["uses_folds"])

    def test_data_summary_reports_malformed_jsonl_line(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.jsonl"
            path.write_text('{"pubid": "1"}\ninvalid\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "broken.jsonl:2"):
                read_jsonl_summary(path)

    def test_lightweight_data_and_reporting_imports_do_not_load_ml_or_plot_frameworks(
        self,
    ) -> None:
        process = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; import pubmedqa.data; import pubmedqa.eval.reports; "
                    "assert 'datasets' not in sys.modules; "
                    "assert 'matplotlib' not in sys.modules; "
                    "assert 'torch' not in sys.modules"
                ),
            ],
            check=False,
            capture_output=True,
            text=True,
            env={"PYTHONPATH": "src"},
        )
        self.assertEqual(0, process.returncode, process.stderr)

    def test_plot_scripts_own_their_flow_and_share_only_reporting_primitives(
        self,
    ) -> None:
        for path in Path("scripts").glob("plot_pubmedqa*.py"):
            source = path.read_text(encoding="utf-8")
            self.assertIn("--output-dir", source, path.name)
            self.assertIn("def generate_", source, path.name)
            self.assertNotIn("pubmedqa.eval.plot_", source, path.name)


def _counts(rows: list[dict]) -> dict[str, int]:
    return {
        label: sum(row["final_decision"] == label for row in rows)
        for label in ("yes", "no")
    }


if __name__ == "__main__":
    unittest.main()
