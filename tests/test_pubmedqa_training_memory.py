from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch

from helpers.distributed import FakeRanks
from pubmedqa.train.distributed import TrainingSession


class TrainingMemoryTest(unittest.TestCase):
    def test_remote_peak_collected_and_sorted(self):
        ranks = FakeRanks()

        def worker(rank):
            owner = SimpleNamespace(
                rank=rank,
                local_rank=rank,
                world_size=2,
                device=torch.device("cpu"),
                control_group=ranks.group,
                metadata=lambda: {"world_size": 2},
            )
            return TrainingSession.collect_memory(
                owner,
                idle_memory={},
                loaded_memory={},
                peak_train_memory={
                    "max_allocated_gb": [8, 12][rank],
                    "max_reserved_gb": None,
                },
            )

        with patch("torch.distributed.all_gather_object", ranks.gather):
            results = ranks.run(worker)
        for result in results:
            self.assertEqual([0, 1], [record["rank"] for record in result["per_rank"]])
            self.assertEqual(12, result["max_peak_allocated_gb"])

    def test_training_windows_exclude_evaluation_and_test(self):
        from pubmedqa.train.loop import TrainingMemory

        counter = {"peak": 0}
        tracker = TrainingMemory(
            torch.device("cpu"),
            sample=lambda _: {
                "max_allocated_gb": counter["peak"],
                "max_reserved_gb": None,
            },
        )
        tracker.start_window()
        counter["peak"] = 8
        tracker.finish_window()
        counter["peak"] = 20  # evaluation outside the training window
        tracker.start_window()
        counter["peak"] = 12
        tracker.finish_window()
        counter["peak"] = 30  # final test must never overwrite the frozen maximum
        self.assertEqual(12, tracker.snapshot()["max_allocated_gb"])

    def test_duplicate_or_missing_rank_is_rejected(self):
        owner = SimpleNamespace(
            rank=0,
            local_rank=0,
            world_size=2,
            device=torch.device("cpu"),
            control_group=object(),
            metadata=lambda: {},
        )
        for records in ([{"rank": 0}, {"rank": 0}], [{"rank": 0}, None]):
            with patch(
                "torch.distributed.all_gather_object",
                side_effect=lambda out, *a, **kw: out.__setitem__(slice(None), records),
            ):
                with self.assertRaisesRegex(RuntimeError, "rank"):
                    TrainingSession.collect_memory(
                        owner, idle_memory={}, loaded_memory={}, peak_train_memory={}
                    )
