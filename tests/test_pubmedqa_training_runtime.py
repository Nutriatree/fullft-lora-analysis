from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch

from pubmedqa.train import distributed as training


class TrainingRuntimeTest(unittest.TestCase):
    def test_distributed_modes_require_launcher_environment(self):
        for mode in ("ddp", "fsdp"):
            owner = SimpleNamespace(
                config=SimpleNamespace(distributed_mode=mode),
                device=torch.device("cpu"),
            )
            with (
                patch.dict("os.environ", {}, clear=True),
                self.assertRaisesRegex(RuntimeError, "torchrun"),
            ):
                training.TrainingSession.initialize(owner)
            with (
                patch.dict(
                    "os.environ", {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "2"}
                ),
                self.assertRaisesRegex(RuntimeError, "requires CUDA"),
            ):
                training.TrainingSession.initialize(owner)

    def test_cpu_reduction_and_metadata(self):
        owner = SimpleNamespace(
            device=torch.device("cpu"),
            fsdp_enabled=False,
            config=SimpleNamespace(distributed_mode="single", fsdp_cpu_offload=False),
        )
        self.assertEqual(3, training.TrainingSession.all_reduce_int(owner, 3))
        owner.rank, owner.local_rank, owner.world_size = 0, 0, 1
        metadata = training.TrainingSession.metadata(owner)
        self.assertEqual(1, metadata["world_size"])

    def test_missing_rank_zero_message_is_an_error(self):
        owner = SimpleNamespace(is_main_process=False)
        with self.assertRaisesRegex(RuntimeError, "produced no rank-0 result"):
            training.TrainingSession.run_on_main_process(owner, lambda: None, operation_name="probe")
