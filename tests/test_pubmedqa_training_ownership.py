"""Architectural regression checks for conceptual locality and narrow state."""

import ast
from dataclasses import fields
import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from helpers.tiny_training import make_config
from pubmedqa.train.distributed import RuntimeSettings, TrainingSession
from pubmedqa.train.artifacts import RunFiles
from pubmedqa.train import full_ft as analysis
from pubmedqa.train.lora import AdapterHistory


class TrainingOwnershipTest(unittest.TestCase):
    def test_records_are_defined_beside_their_producer_and_legacy_imports_match(self):
        import pubmedqa.train as legacy

        for module, names in {
            "train.loop": ("TrainStepLog",),
            "eval.validation": ("EvalPrediction", "EvalMetrics", "EvalResult"),
            "train.checkpoints": ("CheckpointRecord",),
            "train.full_ft": ("LayerwiseReference", "LayerwiseUpdateRecord"),
            "train.artifacts": ("TrainingSummary",),
        }.items():
            owner = importlib.import_module(f"pubmedqa.{module}")
            for name in names:
                with self.subTest(name=name):
                    record = getattr(owner, name)
                    self.assertEqual(owner.__name__, record.__module__)
                    self.assertIs(record, getattr(legacy, name))

    def test_core_functions_have_no_trainer_owner_or_compatibility_imports(self):
        for module in (
            "train/pipeline",
            "eval/validation",
            "train/checkpoints",
            "train/full_ft",
            "train/lora",
        ):
            tree = ast.parse(Path(f"src/pubmedqa/{module}.py").read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertNotIn(
                        node.module,
                        (
                            "pubmedqa.training.contracts",
                            "pubmedqa.full_finetune",
                            "pubmedqa.eval.inference",
                        ),
                    )
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    args = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
                    self.assertFalse(
                        {"self", "trainer", "owner", "context"} & {a.arg for a in args}
                    )

    def test_bounded_records_cannot_become_a_whole_run_container(self):
        for record in (RuntimeSettings, TrainingSession, RunFiles, AdapterHistory):
            self.assertFalse(
                {
                    "model",
                    "optimizer",
                    "scheduler",
                    "dataset",
                    "examples",
                    "environment",
                }
                & {item.name for item in fields(record)}
            )
        self.assertEqual(
            {"adapter_metrics_history", "module_share_history"},
            {item.name for item in fields(AdapterHistory)},
        )

    def test_analysis_direct_call_needs_only_model_paths_and_session(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = make_config(Path(directory))
            files = RunFiles.from_config(cfg)
            session = TrainingSession.from_config(cfg)
            model = torch.nn.Module()
            model.self_attn = torch.nn.Module()
            model.self_attn.q_proj = torch.nn.Linear(2, 2, bias=False)
            references = analysis.capture_layerwise_references(
                model, files=files, session=session
            )
            snapshots = {}
            analysis.write_layerwise_update_artifacts(
                model=model,
                files=files,
                session=session,
                checkpoint_kind="probe",
                checkpoint_percent=0,
                epoch=0,
                global_step=0,
                checkpoint_dir=Path(directory) / "probe",
                references=references,
                previous_snapshots=snapshots,
                previous_incremental_updates={},
            )
            torch.testing.assert_close(
                snapshots["self_attn.q_proj.weight"], model.self_attn.q_proj.weight
            )
            self.assertTrue(
                (Path(directory) / "probe" / "layerwise_updates.jsonl").is_file()
            )

    def test_borrowed_group_survives_pipeline_initialization_failure(self):
        from pubmedqa.train.pipeline import run_training
        from helpers.tiny_training import environment

        with tempfile.TemporaryDirectory() as directory:
            cfg = make_config(Path(directory))
            session = TrainingSession.from_config(cfg)
            session.control_group = object()
            with (
                patch.object(
                    session, "initialize", side_effect=ValueError("initialization")
                ),
                patch("torch.distributed.destroy_process_group") as destroy,
            ):
                with self.assertRaisesRegex(ValueError, "initialization"):
                    run_training(cfg, environment(), session=session)
            destroy.assert_not_called()
            self.assertIsNone(session.control_group)
