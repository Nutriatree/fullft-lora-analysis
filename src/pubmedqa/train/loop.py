"""Shared optimizer-step loop and its memory measurements for Full FT and LoRA."""

from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Iterator

import torch
from torch.nn.utils import clip_grad_norm_

from pubmedqa.data.supervised import validate_targets
from pubmedqa.model.device import memory_snapshot
from pubmedqa.train.distributed import run_rank_local


class TrainingMemory:
    def __init__(self, device, *, sample=memory_snapshot):
        self.device = device
        self.sample = sample
        self.peaks = {"max_allocated_gb": None, "max_reserved_gb": None}

    def start_window(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)

    def finish_window(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        snapshot = self.sample(self.device)
        for field in self.peaks:
            value = snapshot[field]
            if value is not None:
                self.peaks[field] = max(self.peaks[field] or 0.0, value)

    def snapshot(self):
        return dict(self.peaks)


@dataclass(frozen=True)
class TrainStepLog:
    global_step: int
    epoch: int
    step_in_epoch: int
    learning_rate: float
    train_loss: float
    gradient_norm: float
    batch_size: int
    input_tokens: int
    target_tokens: int
    step_time_seconds: float
    samples_per_second: float
    tokens_per_second: float


def require_finite(value, name):
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"nonfinite {name}")


@dataclass(frozen=True)
class OptimizerStep:
    log: TrainStepLog
    loss_total: float
    micro_batches: int


def train_epoch(
    *,
    model: torch.nn.Module,
    loader: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    global_step: int,
    accumulation_steps: int,
    max_grad_norm: float,
    device: torch.device,
    autocast_context: Callable,
    ddp_enabled: bool,
    fsdp_enabled: bool = False,
    memory_tracker: Any = None,
    control_group: Any = None,
) -> Iterator[OptimizerStep]:
    """Run one epoch, preserving the historical partial-accumulation scaling.

    Batches contain [batch, sequence] tensors. Labels use -100 for prompt/pad
    tokens. Even a short final accumulation window divides loss by the configured
    accumulation size; changing that normalization would change experiment results.
    """
    if accumulation_steps <= 0 or len(loader) == 0:
        raise ValueError("accumulation_steps and loader length must be positive")
    model.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    samples = input_tokens = target_tokens = micro_batches = 0
    loss_total = 0.0
    if memory_tracker is not None:
        memory_tracker.start_window()
    iterator = run_rank_local(
        lambda: iter(loader), control_group=control_group, operation_name="loader init"
    )
    for batch_index in range(1, len(loader) + 1):

        def next_batch():
            batch = next(iterator)
            validate_targets(batch["labels"])
            return {key: value.to(device) for key, value in batch.items()}

        batch = run_rank_local(
            next_batch, control_group=control_group, operation_name="batch preflight"
        )
        should_step = batch_index % accumulation_steps == 0 or batch_index == len(
            loader
        )
        # FSDP CPU offload only supports accumulation inside no_sync. This can
        # retain unsharded gradients until the final microbatch (extra GPU RAM).
        sync = (
            model.no_sync()
            if (ddp_enabled or fsdp_enabled) and not should_step
            else nullcontext()
        )
        with sync:
            with autocast_context():
                outputs = model(**batch)
                raw_loss = outputs.loss
                loss = raw_loss / accumulation_steps
            run_rank_local(
                partial(
                    require_finite, raw_loss.detach(), "training loss before backward"
                ),
                control_group=control_group,
                operation_name="loss guard",
            )
            loss.backward()
        loss_total += float(raw_loss.detach().item())
        micro_batches += 1
        samples += int(batch["input_ids"].shape[0])
        input_tokens += int(batch["attention_mask"].sum().item())
        target_tokens += int((batch["labels"] != -100).sum().item())
        del outputs, raw_loss, loss, batch
        if not should_step:
            continue

        # FULL_SHARD must reduce shard norms globally on every rank.
        norm = (
            model.clip_grad_norm_(max_grad_norm)
            if fsdp_enabled
            else clip_grad_norm_(model.parameters(), max_grad_norm)
        )
        gradient_norm = float(norm.detach().item())

        def update():
            require_finite(norm, "gradient norm before optimizer step")
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        run_rank_local(
            update, control_group=control_group, operation_name="optimizer update"
        )
        global_step += 1
        elapsed = time.perf_counter() - started
        log = TrainStepLog(
            global_step=global_step,
            epoch=epoch,
            step_in_epoch=batch_index,
            learning_rate=float(scheduler.get_last_lr()[0]),
            train_loss=loss_total / max(1, micro_batches),
            gradient_norm=gradient_norm,
            batch_size=samples,
            input_tokens=input_tokens,
            target_tokens=target_tokens,
            step_time_seconds=elapsed,
            samples_per_second=samples / elapsed if elapsed > 0 else 0.0,
            tokens_per_second=input_tokens / elapsed if elapsed > 0 else 0.0,
        )
        if memory_tracker is not None:
            memory_tracker.finish_window()
        yield OptimizerStep(log=log, loss_total=loss_total, micro_batches=micro_batches)
        # The caller may have evaluated/saved while the generator was paused.
        if memory_tracker is not None:
            memory_tracker.start_window()
        samples = input_tokens = target_tokens = micro_batches = 0
        loss_total = 0.0
        started = time.perf_counter()
