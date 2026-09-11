"""Fail closed at external I/O boundaries; numerical CPU operations stay real."""

from contextlib import ExitStack, contextmanager
from unittest.mock import patch
import torch
from torch.overrides import TorchFunctionMode


class CpuOnly(TorchFunctionMode):
    """Catch explicit CUDA/MPS allocations as well as tensor device transfers."""

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        devices = [kwargs.get("device")]
        if func.__name__ == "to":
            devices.extend(
                arg for arg in args[1:] if isinstance(arg, (str, torch.device))
            )
        if func.__name__ == "cuda":
            raise AssertionError("Offline CPU test forbids CUDA")
        for device in devices:
            # Transformers constructs shape-only meta parameters before loading
            # local weights. Meta has no storage or accelerator execution.
            if device is not None and torch.device(device).type not in {"cpu", "meta"}:
                raise AssertionError(f"Offline CPU test forbids device {device}")
        return func(*args, **kwargs)


@contextmanager
def offline_cpu():
    # Patching the socket boundary also catches non-Hugging Face HTTP clients.
    # Seed setup may call CUDA's lazy seed API even on CPU, so block allocation,
    # not seed registration. Real process groups are never needed in this suite.
    with ExitStack() as stack:
        stack.enter_context(CpuOnly())
        for target in (
            "socket.socket.connect",
            "socket.create_connection",
            "torch.cuda._lazy_init",
            "torch.distributed.init_process_group",
            "torch.mps.synchronize",
        ):
            stack.enter_context(
                patch(
                    target,
                    side_effect=AssertionError(f"Offline CPU test forbids {target}"),
                )
            )
        yield
