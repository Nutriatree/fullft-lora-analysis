"""Dynamic checks for training resource ownership and lifetime."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from helpers.tiny_training import make_config
from pubmedqa.train.distributed import TrainingSession
from pubmedqa.train.artifacts import RunFiles
from pubmedqa.train import analysis


class TrainingOwnershipTest(unittest.TestCase):
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
