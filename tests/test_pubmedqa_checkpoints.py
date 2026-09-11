from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch
from pubmedqa.train.distributed import TrainingSession, RuntimeSettings
from pubmedqa.train.checkpoints import remove_evaluation_snapshot
from pubmedqa.eval.validation import evaluate_checkpoint_on_main, EvaluationSettings


class CheckpointOperationsTest(unittest.TestCase):
    def test_checkpoint_selection_prefers_f1_then_accuracy_then_loss(self):
        from pubmedqa.train.checkpoints import _is_better_checkpoint

        best = SimpleNamespace(
            validation_macro_f1=0.5, validation_accuracy=0.6, validation_loss=1.0
        )
        for f1, acc, loss, expected in [
            (0.6, 0, 9, True),
            (0.5, 0.7, 9, True),
            (0.5, 0.6, 0.9, True),
            (0.5, 0.6, 1, False),
            (0.4, 1, 0, False),
        ]:
            self.assertEqual(
                expected,
                _is_better_checkpoint(
                    SimpleNamespace(
                        validation_macro_f1=f1,
                        validation_accuracy=acc,
                        validation_loss=loss,
                    ),
                    best,
                ),
            )

    def test_snapshot_cleanup_is_scoped_and_rank_zero_only(self):
        with tempfile.TemporaryDirectory() as directory:
            owner = TrainingSession(RuntimeSettings(), torch.device("cpu"))
            owner.rank = 1
            snapshot = Path(directory) / "snapshot"
            snapshot.mkdir()
            remove_evaluation_snapshot(snapshot, session=owner)
            self.assertTrue(snapshot.exists())
            owner.rank = 0
            remove_evaluation_snapshot(snapshot, session=owner)
            remove_evaluation_snapshot(snapshot, session=owner)
            self.assertFalse(snapshot.exists())
            self.assertTrue(Path(directory).exists())

    def test_checkpoint_loading_failure_is_propagated(self):
        owner = TrainingSession(RuntimeSettings(), torch.device("cpu"))
        owner.rank = 0
        owner.load_model_and_tokenizer = Mock(side_effect=ValueError("bad checkpoint"))
        with self.assertRaisesRegex(RuntimeError, "bad checkpoint"):
            evaluate_checkpoint_on_main(
                session=owner,
                settings=EvaluationSettings(
                    torch.device("cpu"), torch.float32, 1, None, 1, 0, True
                ),
                evaluations_dir=Path("unused"),
                load_model=owner.load_model_and_tokenizer,
                checkpoint_dir=Path("missing"),
                examples=[],
                split_name="test",
            )
