import unittest
from types import SimpleNamespace

import torch

from pubmedqa.train.artifacts import RunFiles
from pubmedqa.train.distributed import TrainingSession
from pubmedqa.train.analysis import (
    capture_layerwise_references,
    write_layerwise_update_artifacts,
)


class TrainingAnalysisTest(unittest.TestCase):
    def test_fsdp_full_access_normalizes_prefix_and_nonwriter_skips_empty_shard(self):
        import tempfile
        from contextlib import contextmanager
        from dataclasses import replace
        from pathlib import Path
        from unittest.mock import patch

        from helpers.tiny_training import make_config

        with tempfile.TemporaryDirectory() as directory:
            cfg = make_config(Path(directory))
            session = TrainingSession.from_config(cfg)
            files = RunFiles.from_config(cfg)
            model = torch.nn.Module()
            model.self_attn = torch.nn.Module()
            model.self_attn.q_proj = torch.nn.Linear(2, 2, bias=False)
            references = capture_layerwise_references(
                model, session=session, files=files, enabled=cfg.track_layerwise_updates
            )
            parameter = model.self_attn.q_proj.weight
            full = parameter.detach().clone()
            wrapped = torch.nn.Module()
            wrapped._fsdp_wrapped_module = model
            parameter.data = torch.empty(0)
            cfg = replace(cfg, distributed_mode="fsdp")
            entries = []

            @contextmanager
            def summon(model, **kwargs):
                self.assertTrue(kwargs["rank0_only"])
                self.assertFalse(kwargs["writeback"])
                entries.append(session.rank)
                parameter.data = full.clone() if session.rank == 0 else torch.empty(0)
                try:
                    yield
                finally:
                    parameter.data = torch.empty(0)

            with (
                patch(
                    "pubmedqa.train.distributed._is_fsdp_model",
                    return_value=True,
                ),
                patch(
                    "pubmedqa.train.distributed.FullyShardedDataParallel.summon_full_params",
                    summon,
                ),
            ):
                for rank in (0, 1):
                    session.rank = rank
                    destination = Path(directory) / f"rank-{rank}"
                    previous = {}
                    write_layerwise_update_artifacts(
                        session=session,
                        files=files,
                        enabled=cfg.track_layerwise_updates,
                        model=wrapped,
                        checkpoint_kind="probe",
                        checkpoint_percent=0,
                        epoch=0,
                        global_step=0,
                        checkpoint_dir=destination,
                        references=references if rank == 0 else [],
                        previous_snapshots=previous,
                        previous_incremental_updates={},
                    )
                    self.assertEqual(rank == 0, destination.exists())
                    if rank == 0:
                        torch.testing.assert_close(
                            previous[references[0].parameter_name], full
                        )
            self.assertEqual([0, 1], entries)

    def test_reference_preserves_dtype_and_independent_small_updates(self):
        import json
        import tempfile
        from pathlib import Path

        from helpers.tiny_training import make_config

        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with tempfile.TemporaryDirectory() as directory:
                cfg = make_config(Path(directory))
                session = TrainingSession.from_config(cfg)
                files = RunFiles.from_config(cfg)
                model = torch.nn.Module()
                model.self_attn = torch.nn.Module()
                model.self_attn.q_proj = torch.nn.Linear(2, 2, bias=False, dtype=dtype)
                with torch.no_grad():
                    model.self_attn.q_proj.weight.fill_(0.1234567)
                references = capture_layerwise_references(
                    model,
                    session=session,
                    files=files,
                    enabled=cfg.track_layerwise_updates,
                )
                self.assertEqual(dtype, references[0].base_tensor.dtype)
                previous, increments = {}, {}
                for step in (0, 1, 2):
                    if step == 1:
                        with torch.no_grad():
                            model.self_attn.q_proj.weight.add_(0.0001)
                    write_layerwise_update_artifacts(
                        session=session,
                        files=files,
                        enabled=cfg.track_layerwise_updates,
                        model=model,
                        checkpoint_kind="probe",
                        checkpoint_percent=step,
                        epoch=0,
                        global_step=step,
                        checkpoint_dir=Path(directory) / f"checkpoint-{step}",
                        references=references,
                        previous_snapshots=previous,
                        previous_incremental_updates=increments,
                    )
                    if step == 0:
                        torch.testing.assert_close(
                            previous["self_attn.q_proj.weight"].float(),
                            references[0].base_tensor.float(),
                            rtol=0,
                            atol=0,
                        )
                    if step == 2:
                        self.assertEqual(
                            0, increments["self_attn.q_proj.weight"].abs().sum().item()
                        )
                    rows = [
                        json.loads(row)
                        for row in (
                            Path(directory)
                            / f"checkpoint-{step}"
                            / "layerwise_updates.jsonl"
                        )
                        .read_text()
                        .splitlines()
                    ]
                    expected = (
                        model.self_attn.q_proj.weight.detach().float()
                        - references[0].base_tensor.float()
                    )
                    self.assertAlmostEqual(
                        expected.norm().item(), rows[0]["update_norm"]
                    )
                self.assertEqual(
                    tuple(expected.shape),
                    tuple(previous["self_attn.q_proj.weight"].shape),
                )

    def test_tracking_disabled_and_nonwriter_allocate_no_references(self):
        import tempfile
        from dataclasses import replace
        from pathlib import Path

        from helpers.tiny_training import make_config

        with tempfile.TemporaryDirectory() as directory:
            cfg = make_config(Path(directory))
            session = TrainingSession.from_config(cfg)
            files = RunFiles.from_config(cfg)
            model = torch.nn.Module()
            model.q_proj = torch.nn.Linear(2, 2)
            for rank, enabled in ((0, False), (1, True)):
                session.rank = rank
                cfg = replace(cfg, track_layerwise_updates=enabled)
                self.assertEqual(
                    [],
                    capture_layerwise_references(
                        model,
                        session=session,
                        files=files,
                        enabled=cfg.track_layerwise_updates,
                    ),
                )
                self.assertFalse(files.layerwise_dir.exists())

    def test_lora_delta_matches_independent_matrix_product(self):
        from pubmedqa.train.lora_analysis import _compute_lora_delta_tensor

        module = SimpleNamespace(
            lora_A={"default": SimpleNamespace(weight=torch.tensor([[1.0, 2.0]]))},
            lora_B={"default": SimpleNamespace(weight=torch.tensor([[3.0], [4.0]]))},
            scaling={"default": 2.0},
        )
        a, b, delta, scaling = _compute_lora_delta_tensor(module)
        torch.testing.assert_close(delta, torch.tensor([[6.0, 12.0], [8.0, 16.0]]))
        self.assertEqual(2.0, scaling)
        self.assertFalse(delta.requires_grad)

    def test_effective_rank_and_zero_update_boundaries(self):
        from pubmedqa.train.analysis import cosine_similarity
        from pubmedqa.train.lora_analysis import _effective_rank_from_singular_values

        self.assertEqual(0.0, _effective_rank_from_singular_values(torch.tensor([])))
        self.assertEqual(0.0, _effective_rank_from_singular_values(torch.zeros(2)))
        self.assertAlmostEqual(
            2.0, _effective_rank_from_singular_values(torch.ones(2)), places=6
        )
        self.assertIsNone(cosine_similarity(torch.zeros(2), torch.ones(2)))

    def test_empty_analysis_groups_and_missing_layers(self):
        from pubmedqa.train.analysis import (
            parse_layer_index,
            summarize_by_component,
            summarize_by_layer,
        )

        self.assertEqual({}, summarize_by_component([]))
        self.assertEqual({}, summarize_by_layer([]))
        self.assertIsNone(parse_layer_index("model.layers.bad.weight"))
        self.assertIsNone(parse_layer_index("model.embedding.weight"))
