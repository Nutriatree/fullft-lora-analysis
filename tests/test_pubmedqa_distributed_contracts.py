from types import SimpleNamespace
import unittest
from unittest.mock import patch

from helpers.distributed import FakeRanks
from pubmedqa.train.distributed import run_on_rank_zero


class DistributedContractsTest(unittest.TestCase):
    def test_study_continue_and_abort_are_shared_and_fatal_errors_never_continue(self):
        from contextlib import nullcontext, redirect_stdout
        from dataclasses import replace
        from pathlib import Path
        import io
        import tempfile
        from unittest.mock import Mock
        from pubmedqa.train.study import ExperimentStudy, execute_study
        from pubmedqa.experiment_runs import DEFAULT_PATHS, DEFAULT_SHARED_DEFAULTS
        from pubmedqa.train.distributed import run_rank_local
        from helpers.tiny_training import environment

        for recoverable, keep in ((True, False), (True, True), (False, True)):
            with tempfile.TemporaryDirectory() as directory:
                ranks = FakeRanks()
                study = ExperimentStudy(
                    "probe",
                    ("F1", "L1"),
                    DEFAULT_PATHS,
                    replace(DEFAULT_SHARED_DEFAULTS, distributed_mode="ddp"),
                    Path(directory),
                    Path(directory),
                    {},
                    environment(),
                    keep,
                )
                counts = [0, 0]

                def run(config, environment, *, session):
                    rank = ranks.local.rank
                    counts[rank] += 1
                    if not recoverable:
                        raise RuntimeError("fatal collective failure")

                    def local():
                        if rank == 1:
                            raise OSError("local input failure")

                    return run_rank_local(
                        local, control_group=ranks.group, operation_name="run preflight"
                    )

                fake_os = SimpleNamespace(
                    environ=SimpleNamespace(get=lambda *args: str(ranks.local.rank))
                )
                with (
                    patch("pubmedqa.train.study.os", fake_os),
                    patch(
                        "pubmedqa.train.study.distributed_control_group",
                        side_effect=lambda _: nullcontext(ranks.group),
                    ),
                    patch("pubmedqa.train.study.run_training", run),
                    patch(
                        "pubmedqa.train.study.TrainingSession.from_config",
                        side_effect=lambda _: SimpleNamespace(control_group=None),
                    ),
                    patch("torch.distributed.broadcast_object_list", ranks.broadcast),
                    patch("torch.distributed.all_gather_object", ranks.gather),
                    patch("torch.distributed.get_world_size", return_value=2),
                    redirect_stdout(io.StringIO()),
                ):
                    results = ranks.run(lambda _: execute_study(study))
                self.assertTrue(
                    all(isinstance(result, RuntimeError) for result in results)
                )
                self.assertEqual([2 if recoverable and keep else 1] * 2, counts)
                self.assertEqual(ranks.traces[0], ranks.traces[1])

    def test_writer_failure_finishes_on_both_ranks_without_barriers(self):
        from contextlib import nullcontext
        from unittest.mock import Mock
        from pubmedqa.train.checkpoints import save_model_files_to_directory
        from pubmedqa.train.distributed import TrainingSession

        for failure in ("save_pretrained", "tokenizer"):
            ranks = FakeRanks()

            def worker(rank):
                owner = SimpleNamespace(
                    is_main_process=rank == 0,
                    fsdp_enabled=True,
                    control_group=ranks.group,
                    _unwrap_model=lambda model: model,
                )
                owner.run_on_main_process = lambda fn, **kw: (
                    TrainingSession.run_on_main_process(owner, fn, **kw)
                )
                model, tokenizer = Mock(), Mock()
                if failure == "tokenizer":
                    tokenizer.save_pretrained.side_effect = OSError("injected")
                else:
                    model.save_pretrained.side_effect = OSError("injected")
                save_model_files_to_directory(
                    session=owner,
                    model=model,
                    tokenizer=tokenizer,
                    checkpoint_dir="unused",
                )

            with (
                patch("torch.distributed.broadcast_object_list", ranks.broadcast),
                patch("pubmedqa.train.checkpoints.FullyShardedDataParallel") as fsdp,
            ):
                fsdp.state_dict_type.side_effect = lambda *_, **kw: nullcontext()
                results = ranks.run(worker)
            self.assertTrue(all(isinstance(x, RuntimeError) for x in results))
            self.assertEqual(str(results[0]), str(results[1]))
            self.assertEqual([["broadcast"]] * 2, ranks.traces)

    def test_standalone_owns_control_group_and_closes_once(self):
        import torch
        from pubmedqa.train import distributed as training

        owner = SimpleNamespace(
            config=SimpleNamespace(distributed_mode="ddp"),
            device=torch.device("cuda:1"),
            control_group=None,
            _owns_process_group=False,
        )
        with (
            patch.dict(
                "os.environ", {"RANK": "1", "LOCAL_RANK": "1", "WORLD_SIZE": "2"}
            ),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.set_device") as set_device,
            patch("torch.distributed.is_initialized", return_value=False),
            patch.object(training, "initialize_control_group", return_value="gloo"),
        ):
            training.TrainingSession.initialize(owner)
        set_device.assert_called_once_with(1)
        self.assertTrue(owner._owns_process_group)
        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.destroy_process_group") as destroy,
        ):
            training.TrainingSession.close(owner)
            training.TrainingSession.close(owner)
        from unittest.mock import call

        self.assertEqual([call("gloo"), call()], destroy.call_args_list)

    def test_borrowed_control_group_is_not_destroyed(self):
        from pubmedqa.train import distributed as training

        owner = SimpleNamespace(
            control_group="borrowed",
            _owns_control_group=False,
            _owns_process_group=False,
        )
        with patch("torch.distributed.destroy_process_group") as destroy:
            training.TrainingSession.close(owner)
        destroy.assert_not_called()

    def test_control_group_creation_failure_cleans_owned_default(self):
        from pubmedqa.train.distributed import initialize_control_group

        with (
            patch.dict(
                "os.environ", {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "2"}
            ),
            patch("torch.cuda.is_available", return_value=True),
            patch("torch.cuda.set_device"),
            patch("torch.distributed.is_initialized", return_value=False),
            patch("torch.distributed.init_process_group"),
            patch("torch.distributed.new_group", side_effect=OSError("gloo failed")),
            patch("torch.distributed.destroy_process_group") as destroy,
        ):
            with self.assertRaisesRegex(OSError, "gloo failed"):
                initialize_control_group("fsdp")
        destroy.assert_called_once_with()

    def test_training_result_explicit_control_group(self):
        from pubmedqa.train.distributed import TrainingSession

        ranks = FakeRanks()
        with patch("torch.distributed.broadcast_object_list", ranks.broadcast):
            results = ranks.run(
                lambda rank: TrainingSession.run_on_main_process(
                    SimpleNamespace(
                        is_main_process=rank == 0, control_group=ranks.group
                    ),
                    lambda: 12,
                    operation_name="evaluation",
                )
            )
        self.assertEqual([12, 12], results)

    def test_local_failure_is_shared_for_either_rank(self):
        from pubmedqa.train.distributed import (
            run_rank_local,
            SynchronizedOperationError,
        )

        for failed_rank in (0, 1):
            ranks = FakeRanks()

            def work(rank):
                def operation():
                    if rank == failed_rank:
                        raise ValueError("bad batch")
                    return rank

                return run_rank_local(
                    operation, control_group=ranks.group, operation_name="preflight"
                )

            with (
                patch("torch.distributed.all_gather_object", ranks.gather),
                patch("torch.distributed.get_world_size", return_value=2),
            ):
                results = ranks.run(work)
            self.assertTrue(
                all(isinstance(x, SynchronizedOperationError) for x in results)
            )
            self.assertEqual(str(results[0]), str(results[1]))

    def test_fsdp_state_collection_on_nonwriter(self):
        from contextlib import nullcontext
        from unittest.mock import Mock
        from pubmedqa.train.checkpoints import save_model_files_to_directory

        for rank in (0, 1):
            model, tokenizer = Mock(), Mock()
            owner = SimpleNamespace(
                is_main_process=rank == 0,
                fsdp_enabled=True,
                _unwrap_model=lambda model: model,
            )
            owner.run_on_main_process = lambda fn, **_: fn() if rank == 0 else None
            with patch(
                "pubmedqa.train.checkpoints.FullyShardedDataParallel"
            ) as fsdp:
                fsdp.state_dict_type.return_value = nullcontext()
                save_model_files_to_directory(
                    session=owner,
                    model=model,
                    tokenizer=tokenizer,
                    checkpoint_dir="unused",
                )
            model.state_dict.assert_called_once()
            self.assertEqual(1 if rank == 0 else 0, model.save_pretrained.call_count)

    def test_control_fixture_broadcasts_result_and_error_to_both_ranks(self):
        for fail in (False, True):
            ranks = FakeRanks()

            def operation():
                if fail:
                    raise OSError("injected write failure")
                return {"value": 7}

            with patch("torch.distributed.broadcast_object_list", ranks.broadcast):
                results = ranks.run(
                    lambda rank: run_on_rank_zero(
                        operation,
                        is_main_process=rank == 0,
                        control_group=ranks.group,
                        operation_name="fixture",
                    )
                )
            if fail:
                self.assertTrue(all(isinstance(item, RuntimeError) for item in results))
                self.assertEqual(str(results[0]), str(results[1]))
            else:
                self.assertEqual([{"value": 7}] * 2, results)
            self.assertEqual([["broadcast"]] * 2, ranks.traces)

    def test_current_zero_target_and_reference_precision_characterization(self):
        # D3/D4 baseline is intentionally documented, not an expected fix yet.
        import torch

        original = torch.tensor([0.1234567], dtype=torch.float32)
        self.assertGreater((original - original.half().float()).norm().item(), 0)
        self.assertEqual(0, (torch.tensor([[-100]])[:, 1:] != -100).sum().item())

    def test_control_fixture_collects_remote_rank(self):
        ranks = FakeRanks()

        def operation(rank):
            result = [None] * ranks.size
            ranks.gather(result, {"rank": rank}, ranks.group)
            return result

        self.assertEqual([[{"rank": 0}, {"rank": 1}]] * 2, ranks.run(operation))
