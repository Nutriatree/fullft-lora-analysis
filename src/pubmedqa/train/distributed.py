"""Rank execution, distributed model lifetime and full-parameter analysis contexts."""

from __future__ import annotations

import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field, fields
from datetime import timedelta
from functools import partial, wraps
from typing import Any, Callable, Iterator, TypeVar, cast

import torch
import torch.distributed as dist
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel

from pubmedqa.model.device import supports_cuda_amp as _supports_cuda_amp

T = TypeVar("T")


DISTRIBUTED_MODES = ("single", "ddp", "fsdp")


class SynchronizedOperationError(RuntimeError):
    """All live ranks acknowledged a local (non-collective) operation failure."""


def control_timeout_seconds() -> int:
    value = int(os.environ.get("PUBMEDQA_CONTROL_TIMEOUT_SECONDS", "86400"))
    if value <= 0:
        raise ValueError("PUBMEDQA_CONTROL_TIMEOUT_SECONDS must be positive")
    return value


def run_rank_local(
    operation: Callable[[], T], *, control_group: Any | None, operation_name: str
) -> T:
    """Agree on local failures before entering the next tensor collective.

    Never wrap forward/backward/FSDP collection here: a peer may already be
    blocked inside NCCL. Such exceptions are fatal and must escape the study.
    """
    if control_group is None:
        return operation()
    result = None
    error = None
    try:
        result = operation()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    errors = [None] * dist.get_world_size(group=control_group)
    dist.all_gather_object(errors, error, group=control_group)
    failed = [f"rank {rank}: {error}" for rank, error in enumerate(errors) if error]
    if failed:
        raise SynchronizedOperationError(
            f"{operation_name} failed; " + "; ".join(failed)
        )
    return cast(T, result)


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
    owns_default = not dist.is_initialized()
    timeout = timedelta(seconds=control_timeout_seconds())
    if owns_default:
        dist.init_process_group(backend="nccl")
    try:
        return dist.new_group(backend="gloo", timeout=timeout)
    except Exception:
        if owns_default:
            dist.destroy_process_group()
        raise


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

    owns_default = mode != "single" and not dist.is_initialized()
    control_group = initialize_control_group(mode) if mode != "single" else None
    try:
        yield control_group
    finally:
        if mode != "single" and dist.is_initialized():
            if owns_default:
                destroy_process_groups(control_group)
            elif control_group is not None:
                dist.destroy_process_group(control_group)


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
        raise SynchronizedOperationError(
            f"{operation_name} failed on rank 0: "
            f"{message['error_type']}: {message['error']}"
        )
    return cast(T, message["result"])


try:
    from torch.distributed.fsdp import (
        CPUOffload,
        FullStateDictConfig,
        FullyShardedDataParallel,
        MixedPrecision,
        ShardingStrategy,
        StateDictType,
    )
except ImportError:  # pragma: no cover - older torch builds
    CPUOffload = None
    FullStateDictConfig = None
    FullyShardedDataParallel = None
    MixedPrecision = None
    ShardingStrategy = None
    StateDictType = None


@dataclass(frozen=True)
class RuntimeSettings:
    """Only execution policy, never datasets, output paths or model ownership."""

    distributed_mode: str = "single"
    dtype: Any = torch.float32
    seed: int = 42
    cpu_threads: int = 1
    fsdp_cpu_offload: bool = False
    attn_implementation: str | None = None


@dataclass
class TrainingSession:
    """Own communication resources; borrowed study groups remain study-owned."""

    config: RuntimeSettings
    device: torch.device
    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    control_group: Any = None
    _owns_process_group: bool = False
    _owns_control_group: bool = False
    _fsdp_layer_classes: list[str] = field(default_factory=list)

    @classmethod
    def from_config(cls, config):
        from pubmedqa.model.device import resolve_device

        settings = RuntimeSettings(
            **{
                item.name: getattr(config, item.name)
                for item in fields(RuntimeSettings)
            }
        )
        return cls(settings, resolve_device(config.device))

    @property
    def is_main_process(self):
        return self.rank == 0

    @property
    def fsdp_enabled(self):
        return self.config.distributed_mode == "fsdp"

    @property
    def ddp_enabled(self):
        return self.config.distributed_mode == "ddp"

    def run_on_main_process(
        self,
        operation: Callable[[], T],
        *,
        operation_name: str,
    ) -> T:
        return run_on_rank_zero(
            operation,
            is_main_process=self.is_main_process,
            control_group=getattr(self, "control_group", None),
            operation_name=operation_name,
        )

    def run_rank_local(self, operation: Callable[[], T], *, operation_name: str) -> T:
        return run_rank_local(
            operation,
            control_group=getattr(self, "control_group", None),
            operation_name=operation_name,
        )

    def autocast_context(self):
        enabled = _supports_cuda_amp(self.config.dtype, self.device)
        if not enabled:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.config.dtype, enabled=True)

    def set_runtime(self) -> None:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
        torch.manual_seed(self.config.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(self.config.seed)
        torch.set_num_threads(max(1, self.config.cpu_threads))
        try:
            torch.set_num_interop_threads(max(1, min(4, self.config.cpu_threads)))
        except RuntimeError:
            pass

    def initialize(self) -> None:
        if self.config.distributed_mode == "single":
            self.rank = 0
            self.local_rank = 0
            self.world_size = 1
            return

        required = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
        missing = [name for name in required if name not in os.environ]
        if missing:
            raise RuntimeError(
                f"{self.config.distributed_mode.upper()} mode must be launched with torchrun; "
                f"missing environment variables: {', '.join(missing)}"
            )
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError(
                f"{self.config.distributed_mode.upper()} mode requires CUDA."
            )
        self.rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device(f"cuda:{self.local_rank}")
        if getattr(self, "control_group", None) is None:
            self._owns_process_group = not dist.is_initialized()
            self.control_group = initialize_control_group(self.config.distributed_mode)
            self._owns_control_group = True

    def close(self) -> None:
        # A study lends its control group to runners; only its owner destroys it.
        try:
            if getattr(self, "_owns_control_group", False) and dist.is_initialized():
                dist.destroy_process_group(self.control_group)
        finally:
            self._owns_control_group = False
            self.control_group = None
            if self._owns_process_group and dist.is_initialized():
                dist.destroy_process_group()
            self._owns_process_group = False

    def barrier(self) -> None:
        if getattr(self, "control_group", None) is not None:
            dist.barrier(group=self.control_group)

    def all_reduce_int(self, value: int) -> int:
        if not dist.is_initialized():
            return value
        tensor = torch.tensor(value, device=self.device, dtype=torch.long)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return int(tensor.item())

    def metadata(self) -> dict[str, Any]:
        return {
            "mode": self.config.distributed_mode,
            "world_size": self.world_size,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "sharding_strategy": ("FULL_SHARD" if self.fsdp_enabled else None),
            "fsdp_cpu_offload": (
                self.config.fsdp_cpu_offload if self.fsdp_enabled else None
            ),
            "control_timeout_seconds": control_timeout_seconds(),
            "control_backend": "gloo" if self.world_size > 1 else None,
            "fsdp_auto_wrap_layer_classes": getattr(self, "_fsdp_layer_classes", []),
            "fsdp_initialization": "cpu_pretrained_per_transformer_unit"
            if self.fsdp_enabled
            else None,
            "evaluation_device": "cpu" if self.fsdp_enabled else str(self.device),
            "master_dtype": "torch.float32"
            if self.fsdp_enabled
            else str(getattr(self.config, "dtype", None)),
            "evaluation_dtype": "torch.float32"
            if self.fsdp_enabled
            else str(getattr(self.config, "dtype", None)),
            "evaluation_attention": "eager"
            if self.fsdp_enabled
            else getattr(self.config, "attn_implementation", None),
        }

    def collect_memory(
        self,
        *,
        idle_memory: dict[str, float | None],
        loaded_memory: dict[str, float | None],
        peak_train_memory: dict[str, float | None],
    ) -> dict[str, Any]:
        local_record = {
            "rank": self.rank,
            "local_rank": self.local_rank,
            "device": str(self.device),
            "gpu_name": torch.cuda.get_device_name(self.device)
            if self.device.type == "cuda"
            else None,
            "idle_memory": idle_memory,
            "model_loaded_memory": loaded_memory,
            "peak_training_memory": peak_train_memory,
            "memory_measured": self.device.type == "cuda",
        }
        records = [local_record]
        if getattr(self, "control_group", None) is not None:
            records = [None] * self.world_size
            dist.all_gather_object(records, local_record, group=self.control_group)
        if any(record is None for record in records) or sorted(
            record["rank"] for record in records
        ) != list(range(self.world_size)):
            raise RuntimeError(
                "Incomplete or duplicate distributed memory rank records"
            )
        records.sort(key=lambda record: record["rank"])
        return {
            **self.metadata(),
            "per_rank": records,
            "max_peak_allocated_gb": max(
                (
                    record["peak_training_memory"]["max_allocated_gb"] or 0.0
                    for record in records
                ),
                default=0.0,
            ),
            "max_peak_reserved_gb": max(
                (
                    record["peak_training_memory"]["max_reserved_gb"] or 0.0
                    for record in records
                ),
                default=0.0,
            ),
        }

    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        if self.ddp_enabled:
            return DistributedDataParallel(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                broadcast_buffers=False,
            )
        if self.fsdp_enabled:
            # CPU pretrained weights are materialized/sharded per Transformer unit.
            # Root-only wrapping would copy the entire model to CUDA first. Keep
            # embeddings/shared head at the root to preserve tied-weight ownership.
            declared = set(getattr(model, "_no_split_modules", None) or ())
            layer_classes = {
                type(module)
                for module in model.modules()
                if type(module).__name__ in declared
            }
            if not layer_classes:
                raise ValueError(
                    "FSDP requires declared Transformer _no_split_modules boundaries"
                )
            self._fsdp_layer_classes = sorted(cls.__name__ for cls in layer_classes)
            if getattr(self.config, "attn_implementation", None) is not None:
                model.set_attn_implementation(self.config.attn_implementation)
            if (
                FullyShardedDataParallel is None
                or MixedPrecision is None
                or CPUOffload is None
                or ShardingStrategy is None
            ):
                raise RuntimeError("FSDP is not available in this torch build.")
            mixed_precision = None
            if self.config.dtype in {torch.float16, torch.bfloat16}:
                mixed_precision = MixedPrecision(
                    param_dtype=self.config.dtype,
                    reduce_dtype=self.config.dtype,
                    buffer_dtype=self.config.dtype,
                )
            return FullyShardedDataParallel(
                model,
                device_id=self.local_rank,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                cpu_offload=CPUOffload(offload_params=self.config.fsdp_cpu_offload),
                mixed_precision=mixed_precision,
                use_orig_params=True,
                auto_wrap_policy=partial(
                    transformer_auto_wrap_policy, transformer_layer_cls=layer_classes
                ),
            )
        return model

    def move_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {key: value.to(self.device) for key, value in batch.items()}


def _is_fsdp_model(model: torch.nn.Module) -> bool:
    return FullyShardedDataParallel is not None and isinstance(
        model, FullyShardedDataParallel
    )


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def canonical_parameter_name(name):
    return ".".join(part for part in name.split(".") if part != "_fsdp_wrapped_module")


def rank_zero_analysis(operation):
    @wraps(operation)
    def wrapped(model, *args, session, enabled=True, **kwargs):
        if not enabled:
            return []
        context = nullcontext()
        if _is_fsdp_model(model):
            # All ranks unshard; only rank 0 retains CPU full values. Never
            # evaluate/forward under summon_full_params or inspect local shards.
            context = FullyShardedDataParallel.summon_full_params(
                model,
                recurse=True,
                writeback=False,
                rank0_only=True,
                offload_to_cpu=True,
            )
        with context:
            return session.run_rank_local(
                lambda: (
                    operation(*args, model=model, **kwargs)
                    if session.is_main_process
                    else []
                ),
                operation_name=operation.__name__,
            )

    return wrapped


def tensor_bytes(values):
    return sum(value.numel() * value.element_size() for value in values)
