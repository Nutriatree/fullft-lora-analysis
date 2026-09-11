import tempfile
import unittest
from contextlib import nullcontext
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from helpers.tiny_training import environment, make_config, model_options, with_lora

from pubmedqa.model.loading import load_full_model
from pubmedqa.model.lora import AdapterOptions, load_lora_model
from pubmedqa.train import distributed as training
from pubmedqa.train.checkpoints import save_model_files_to_directory


class DistributedLoadingTest(unittest.TestCase):
    def test_two_rank_engine_checkpoint_evaluation_and_finalize_trace(self):
        from threading import Lock

        from helpers.distributed import FakeRanks
        from helpers.wrappers import CPUWrapper

        class FakeDDP(CPUWrapper):
            pass

        class FakeFSDP(CPUWrapper):
            def named_parameters(self, *args, **kwargs):
                for name, parameter in self.module.named_parameters(*args, **kwargs):
                    yield f"_fsdp_wrapped_module.{name}", parameter

        for mode in ("ddp", "fsdp"):
            with tempfile.TemporaryDirectory() as directory:
                cfg = make_config(
                    Path(directory),
                    distributed_mode=mode,
                    train_batch_size=1,
                    gradient_accumulation_steps=2,
                )
                ranks = FakeRanks()
                load_lock = Lock()

                from pubmedqa.train import pipeline

                original_load = pipeline.load_full_model
                original_examples = pipeline.load_local_jsonl

                def load(source, **options):
                    # Model/datasets contexts are process-global in this thread-only fixture.
                    with load_lock:
                        return original_load(source, **options)

                def examples(path):
                    with load_lock:
                        return original_examples(path)

                def worker(rank):
                    owner = training.TrainingSession.from_config(cfg)
                    owner.rank = owner.local_rank = rank
                    owner.world_size = 2
                    owner.control_group = ranks.group
                    owner.initialize = lambda: None

                    def reduce_int(value):
                        values = [None, None]
                        ranks.gather(values, value, ranks.group)
                        return sum(values)

                    owner.all_reduce_int = reduce_int
                    return pipeline.run_training(cfg, environment(), session=owner)

                with (
                    patch.object(pipeline, "load_full_model", load),
                    patch.object(pipeline, "load_local_jsonl", examples),
                    patch.object(training, "DistributedDataParallel", FakeDDP),
                    patch.object(training, "FullyShardedDataParallel", FakeFSDP),
                    patch(
                        "pubmedqa.train.checkpoints.FullyShardedDataParallel",
                        FakeFSDP,
                    ),
                    patch(
                        "pubmedqa.train.distributed.FullyShardedDataParallel",
                        FakeFSDP,
                    ),
                    patch("torch.distributed.broadcast_object_list", ranks.broadcast),
                    patch("torch.distributed.all_gather_object", ranks.gather),
                    patch("torch.distributed.get_world_size", return_value=2),
                ):
                    results = ranks.run(worker)
                for result in results:
                    self.assertNotIsInstance(result, Exception, str(result))
                    self.assertEqual(1, result.optimizer_steps)
                self.assertEqual(
                    results[0].best_validation_loss, results[1].best_validation_loss
                )
                self.assertEqual(ranks.traces[0], ranks.traces[1])

    def test_half_precision_fsdp_peft_has_uniform_fp32_master_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            cfg = make_config(Path(directory), distributed_mode="fsdp")
            for dtype in (torch.float16, torch.bfloat16):
                config = with_lora(
                    replace(cfg, dtype=dtype),
                    target_modules=("q_proj",),
                    target_layers=(0,),
                    layer_scope="selected",
                )
                session = training.TrainingSession.from_config(config)
                _, model = load_lora_model(
                    config.model_name,
                    options=model_options(config),
                    adapter=AdapterOptions.from_config(config),
                )
                self.assertEqual({torch.float32}, {p.dtype for p in model.parameters()})
                with patch.object(training, "FullyShardedDataParallel") as fsdp:
                    session.wrap_model(model)
                self.assertEqual(
                    dtype, fsdp.call_args.kwargs["mixed_precision"].param_dtype
                )

    def test_nine_method_mode_load_step_save_reload_contracts(self):
        from helpers.wrappers import CPUWrapper

        from pubmedqa.train.loop import train_epoch

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = make_config(root)
            for method in ("full", "lora", "selective"):
                expected = None
                for mode in ("single", "ddp", "fsdp"):
                    with self.subTest(method=method, mode=mode):
                        cfg = replace(config, distributed_mode=mode)
                        if method == "full":
                            load_model = partial(
                                load_full_model, options=model_options(cfg)
                            )
                        else:
                            cfg = with_lora(
                                cfg,
                                target_modules=("q_proj", "v_proj"),
                                target_layers=(0,) if method == "selective" else (),
                                layer_scope="selected"
                                if method == "selective"
                                else "all",
                            )
                            load_model = partial(
                                load_lora_model,
                                options=model_options(cfg),
                                adapter=AdapterOptions.from_config(cfg),
                            )
                        session = training.TrainingSession.from_config(cfg)
                        torch.manual_seed(123)
                        tokenizer, original = load_model(cfg.model_name)
                        before = {
                            n: p.clone()
                            for n, p in original.named_parameters()
                            if not p.requires_grad
                        }
                        with (
                            patch.object(
                                training, "DistributedDataParallel", CPUWrapper
                            ),
                            patch.object(
                                training, "FullyShardedDataParallel", CPUWrapper
                            ),
                            patch(
                                "pubmedqa.train.checkpoints.FullyShardedDataParallel",
                                CPUWrapper,
                            ),
                        ):
                            model = session.wrap_model(original)
                            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
                            scheduler = torch.optim.lr_scheduler.LambdaLR(
                                optimizer, lambda _: 1
                            )
                            batch = dict(
                                input_ids=torch.tensor([[3, 4, 5]]),
                                labels=torch.tensor([[-100, 4, 5]]),
                                attention_mask=torch.ones(1, 3),
                            )
                            steps = list(
                                train_epoch(
                                    model=model,
                                    loader=[batch],
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    epoch=1,
                                    global_step=0,
                                    accumulation_steps=1,
                                    max_grad_norm=1,
                                    device=torch.device("cpu"),
                                    autocast_context=nullcontext,
                                    ddp_enabled=mode == "ddp",
                                    fsdp_enabled=mode == "fsdp",
                                )
                            )
                            destination = root / f"{method}-{mode}"
                            save_model_files_to_directory(
                                session=session,
                                model=model,
                                tokenizer=tokenizer,
                                checkpoint_dir=destination,
                            )
                        _, restored = load_model(str(destination))
                        restored.eval()
                        logits = restored(batch["input_ids"]).logits.detach()
                        if expected is None:
                            expected = logits
                        torch.testing.assert_close(logits, expected)
                        for name, value in original.named_parameters():
                            if name in before:
                                torch.testing.assert_close(
                                    value, before[name], rtol=0, atol=0
                                )
                        self.assertEqual(1, len(steps))

    def test_fsdp_load_preserves_cpu_weights_before_wrap(self):
        with tempfile.TemporaryDirectory() as directory:
            config = make_config(Path(directory))
            _, reference = load_full_model(
                config.model_name, options=model_options(config)
            )
            config = replace(config, distributed_mode="fsdp")
            session = training.TrainingSession.from_config(config)
            session.device = torch.device("cuda:0")
            _, model = load_full_model(
                config.model_name, options=model_options(config, device=session.device)
            )
            for actual, expected in zip(model.parameters(), reference.parameters()):
                self.assertEqual("cpu", actual.device.type)
                torch.testing.assert_close(actual, expected)
            with patch.object(training, "FullyShardedDataParallel") as wrapper:
                session.wrap_model(model)
            policy = wrapper.call_args.kwargs["auto_wrap_policy"]
            self.assertTrue(policy(model.model.layers[0], False, 1))
            self.assertEqual(["LlamaDecoderLayer"], session._fsdp_layer_classes)

    def test_fsdp_unsupported_structure_fails_explicitly(self):
        owner = SimpleNamespace(
            ddp_enabled=False,
            fsdp_enabled=True,
            local_rank=0,
            config=SimpleNamespace(dtype=torch.float32, fsdp_cpu_offload=False),
        )
        with self.assertRaisesRegex(ValueError, "Transformer"):
            training.TrainingSession.wrap_model(owner, torch.nn.Linear(2, 2))

    def test_fsdp_clipping_uses_global_method(self):
        from test_pubmedqa_training_loop import ScalarLoss

        from pubmedqa.train.loop import train_epoch

        model = ScalarLoss()
        # Local shard norms 3 and 4 require the wrapper's global norm 5.
        model.clip_grad_norm_ = Mock(return_value=torch.tensor(5.0))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
        batch = dict(
            input_ids=torch.ones(1, 2),
            labels=torch.ones(1, 2),
            attention_mask=torch.ones(1, 2),
        )
        with patch("pubmedqa.train.loop.clip_grad_norm_") as generic:
            steps = list(
                train_epoch(
                    model=model,
                    loader=[batch],
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=1,
                    global_step=0,
                    accumulation_steps=1,
                    max_grad_norm=1,
                    device=torch.device("cpu"),
                    autocast_context=nullcontext,
                    ddp_enabled=False,
                    fsdp_enabled=True,
                )
            )
        generic.assert_not_called()
        model.clip_grad_norm_.assert_called_once_with(1)
        self.assertEqual(5, steps[0].log.gradient_norm)
