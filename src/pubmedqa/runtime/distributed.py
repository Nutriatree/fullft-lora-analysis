"""Distributed-process lifecycle and rank-zero operation helpers."""

from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import timedelta
from typing import Any, Callable, Iterator, TypeVar, cast

import torch
import torch.distributed as dist

T = TypeVar("T")
DISTRIBUTED_MODES = ("single", "ddp", "fsdp")


def initialize_control_group(mode: str) -> Any | None:
    """Initialize NCCL workers and a long-timeout Gloo control group."""

    if mode == "single":
        return None
    if mode not in DISTRIBUTED_MODES:
        raise ValueError(f"Unsupported distributed mode: {mode!r}")

    required = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise RuntimeError(
            f"{mode.upper()} mode must be launched with torchrun; missing environment "
            f"variables: {', '.join(missing)}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError(f"{mode.upper()} mode requires CUDA.")

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    return dist.new_group(backend="gloo", timeout=timedelta(hours=24))


def destroy_process_groups(control_group: Any | None) -> None:
    """Destroy an optional control group followed by the default group."""

    if not dist.is_initialized():
        return
    try:
        if control_group is not None:
            dist.destroy_process_group(control_group)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@contextmanager
def distributed_control_group(mode: str) -> Iterator[Any | None]:
    """Own the complete lifecycle of the process groups used by one experiment study."""

    control_group = initialize_control_group(mode) if mode != "single" else None
    try:
        yield control_group
    finally:
        if mode != "single":
            destroy_process_groups(control_group)


def run_on_rank_zero(
    operation: Callable[[], T],
    *,
    is_main_process: bool,
    control_group: Any | None,
    operation_name: str,
) -> T:
    """Execute once and broadcast a serializable result or error to every rank."""

    payload: list[dict[str, Any] | None] = [None]
    if is_main_process:
        try:
            payload[0] = {"ok": True, "result": operation()}
        except Exception as exc:
            payload[0] = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    if control_group is not None:
        dist.broadcast_object_list(payload, src=0, group=control_group)

    message = payload[0]
    if message is None:
        raise RuntimeError(f"{operation_name} produced no rank-0 result.")
    if not message["ok"]:
        raise RuntimeError(
            f"{operation_name} failed on rank 0: "
            f"{message['error_type']}: {message['error']}"
        )
    return cast(T, message["result"])
