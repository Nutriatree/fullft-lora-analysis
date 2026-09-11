"""Inference backend protocol and Torch/MLX adapters."""

from __future__ import annotations

import gc
import importlib.util
import platform
from typing import Any, Protocol, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pubmedqa.config import EnvironmentConfig
from pubmedqa.config.eval import ModelRuntimeConfig

DEFAULT_MLX_MODEL_MAP: dict[str, str] = {
    "Qwen/Qwen3-0.6B": "mlx-community/Qwen3-0.6B-bf16",
    "meta-llama/Llama-3.2-1B-Instruct": "mlx-community/Llama-3.2-1B-Instruct-bf16",
    "Qwen/Qwen3-1.7B": "mlx-community/Qwen3-1.7B-bf16",
    "Qwen/Qwen3-4B": "mlx-community/Qwen3-4B-bf16",
    "google/gemma-3-4b-it": "mlx-community/gemma-3-4b-it-bf16",
}


def is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() in {"arm64", "aarch64"}


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def resolve_backend(requested: str) -> str:
    requested = requested.strip().lower()
    if requested not in {"auto", "torch", "mlx"}:
        raise ValueError("PUBMEDQA_BACKEND must be one of: auto, torch, mlx")

    if requested == "mlx":
        if not is_apple_silicon():
            raise RuntimeError("MLX backend requires Apple Silicon.")
        if not module_available("mlx_lm"):
            raise RuntimeError("MLX backend requested but mlx-lm is not installed.")
        return "mlx"

    if requested == "torch":
        return "torch"

    # auto
    if is_apple_silicon() and module_available("mlx_lm"):
        return "mlx"
    return "torch"


def resolve_torch_device(requested: str) -> torch.device:
    requested = requested.strip().lower()
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"CUDA device requested but CUDA is unavailable: {requested}"
            )
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS device requested but MPS is unavailable.")
        return device

    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_torch_dtype(device: torch.device, requested: str) -> torch.dtype:
    requested = requested.strip().lower()
    if requested == "auto":
        if device.type == "cuda":
            return torch.bfloat16
        if device.type == "mps":
            return torch.float16
        return torch.float32

    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if requested not in mapping:
        raise ValueError("PUBMEDQA_DTYPE must be one of: auto, bf16, fp16, fp32")
    return mapping[requested]


def resolve_attention(device: torch.device, requested: str | None) -> str | None:
    if requested is None:
        return None
    requested = requested.strip().lower()
    if requested == "auto":
        return "sdpa" if device.type == "cuda" else None
    return requested


def resolve_batch_size(
    model_name: str, backend: str, device: str, requested: int | None
) -> int:
    if requested is not None:
        return max(1, requested)

    # MLX generate() is kept sequential here for compatibility and deterministic screening.
    if backend == "mlx":
        return 1

    if device.startswith("cuda"):
        return 8

    if device == "mps":
        # Conservative defaults for an M1 Pro 32 GB. Override with PUBMEDQA_BATCH_SIZE if desired.
        if "0.6B" in model_name or "1B-Instruct" in model_name:
            return 16
        if "1.7B" in model_name:
            return 8
        if "3B-Instruct" in model_name:
            return 4
        if "4B" in model_name or "4b" in model_name:
            return 2
        return 2

    return 1


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()


def empty_device_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def reset_device_memory_stats(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def get_device_memory_gb(device: torch.device) -> float | None:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024**3)
    if (
        device.type == "mps"
        and hasattr(torch, "mps")
        and hasattr(torch.mps, "current_allocated_memory")
    ):
        # MPS exposes current allocated memory, not the same peak metric as CUDA.
        return torch.mps.current_allocated_memory() / (1024**3)
    return None


class InferenceBackend(Protocol):
    backend_name: str
    resolved_model_name: str
    tokenizer: Any
    device_label: str
    dtype_label: str

    def generate(self, prompts: Sequence[str]) -> list[str]: ...
    def memory_gb(self) -> float | None: ...
    def release(self) -> None: ...


class TorchBackend:
    backend_name = "torch"

    def __init__(
        self, runtime: ModelRuntimeConfig, environment: EnvironmentConfig
    ) -> None:
        self.runtime = runtime
        self.environment = environment
        self.resolved_model_name = runtime.model_name
        self.device = resolve_torch_device(runtime.device)
        self.device_label = str(self.device)
        self.dtype = resolve_torch_dtype(self.device, runtime.dtype)
        self.dtype_label = str(self.dtype).replace("torch.", "")
        self.attn_implementation = resolve_attention(
            self.device, runtime.attn_implementation
        )
        self.batch_size = resolve_batch_size(
            runtime.model_name, "torch", self.device.type, runtime.batch_size
        )

        common_kwargs: dict[str, Any] = {
            "token": environment.hf_token,
            "trust_remote_code": runtime.trust_remote_code,
        }
        model_kwargs: dict[str, Any] = {
            **common_kwargs,
            "torch_dtype": self.dtype,
        }
        if self.attn_implementation is not None:
            model_kwargs["attn_implementation"] = self.attn_implementation

        if runtime.model_name.startswith("google/gemma-3-"):
            try:
                from transformers import AutoModelForMultimodalLM, AutoProcessor
            except ImportError as exc:
                raise RuntimeError(
                    "Gemma 3 requires a recent transformers version with AutoModelForMultimodalLM."
                ) from exc
            processor = AutoProcessor.from_pretrained(
                runtime.model_name, **common_kwargs
            )
            tokenizer = getattr(processor, "tokenizer", None)
            if tokenizer is None:
                raise RuntimeError("Gemma 3 processor does not expose a tokenizer.")
            model = AutoModelForMultimodalLM.from_pretrained(
                runtime.model_name, **model_kwargs
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                runtime.model_name, **common_kwargs
            )
            model = AutoModelForCausalLM.from_pretrained(
                runtime.model_name, **model_kwargs
            )

        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise RuntimeError(
                    f"Tokenizer for {runtime.model_name} has neither pad_token_id nor eos_token_id."
                )
            tokenizer.pad_token = tokenizer.eos_token

        model.to(self.device)
        model.eval()
        self.tokenizer = tokenizer
        self.model = model

        empty_device_cache(self.device)
        reset_device_memory_stats(self.device)
        synchronize_device(self.device)

    def generate(self, prompts: Sequence[str]) -> list[str]:
        outputs: list[str] = []
        batch_size = max(1, self.batch_size)

        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            tokenize_kwargs: dict[str, Any] = {
                "return_tensors": "pt",
                "padding": True,
                "truncation": self.runtime.max_input_tokens is not None,
            }
            if self.runtime.max_input_tokens is not None:
                tokenize_kwargs["max_length"] = self.runtime.max_input_tokens

            encoded = self.tokenizer(batch_prompts, **tokenize_kwargs)
            encoded = {
                key: value.to(self.device)
                for key, value in encoded.items()
                if isinstance(value, torch.Tensor)
            }
            input_width = encoded["input_ids"].shape[1]

            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": self.runtime.max_new_tokens,
                "do_sample": False,
                "pad_token_id": self.tokenizer.pad_token_id,
                "use_cache": True,
            }
            if self.tokenizer.eos_token_id is not None:
                generation_kwargs["eos_token_id"] = self.tokenizer.eos_token_id

            with torch.inference_mode():
                generated = self.model.generate(**encoded, **generation_kwargs)

            response_ids = generated[:, input_width:]
            outputs.extend(
                text.strip()
                for text in self.tokenizer.batch_decode(
                    response_ids, skip_special_tokens=True
                )
            )

        synchronize_device(self.device)
        return outputs

    def memory_gb(self) -> float | None:
        return get_device_memory_gb(self.device)

    def release(self) -> None:
        del self.model
        del self.tokenizer
        gc.collect()
        empty_device_cache(self.device)


class MLXBackend:
    backend_name = "mlx"

    def __init__(
        self, runtime: ModelRuntimeConfig, environment: EnvironmentConfig
    ) -> None:
        del (
            environment
        )  # HF auth is handled by the local/HF tooling used by mlx-lm/mlx-vlm.
        if not is_apple_silicon():
            raise RuntimeError("MLX backend requires Apple Silicon.")

        self.runtime = runtime
        self.resolved_model_name = DEFAULT_MLX_MODEL_MAP.get(
            runtime.model_name, runtime.model_name
        )
        self.device_label = "mlx"
        self.dtype_label = "model-native"
        self.batch_size = 1
        self.is_gemma_vlm = runtime.model_name.startswith("google/gemma-3-")

        if self.is_gemma_vlm:
            if not module_available("mlx_vlm"):
                raise RuntimeError(
                    "Gemma 3 MLX evaluation requires mlx-vlm: pip install -U mlx-vlm"
                )
            from mlx_vlm import load as vlm_load

            self.model, self.processor = vlm_load(self.resolved_model_name)
            tokenizer = getattr(self.processor, "tokenizer", None)
            if tokenizer is None:
                raise RuntimeError(
                    "MLX-VLM Gemma processor does not expose a tokenizer."
                )
            self.tokenizer = tokenizer
        else:
            if not module_available("mlx_lm"):
                raise RuntimeError("MLX backend requires mlx-lm: pip install -U mlx-lm")
            from mlx_lm import load as lm_load

            self.model, self.tokenizer = lm_load(self.resolved_model_name)
            self.processor = None

    def generate(self, prompts: Sequence[str]) -> list[str]:
        outputs: list[str] = []

        if self.is_gemma_vlm:
            from mlx_vlm import generate as vlm_generate
            from mlx_vlm.prompt_utils import (
                apply_chat_template as vlm_apply_chat_template,
            )

            for prompt in prompts:
                # The common prompt_builder has already built a full text prompt. For Gemma 3
                # in MLX-VLM we wrap that text using its model-specific text-only template.
                formatted = vlm_apply_chat_template(
                    self.processor,
                    self.model.config,
                    prompt,
                    num_images=0,
                )
                result = vlm_generate(
                    self.model,
                    self.processor,
                    formatted,
                    max_tokens=self.runtime.max_new_tokens,
                    temperature=0.0,
                    verbose=False,
                )
                text = result.text if hasattr(result, "text") else str(result)
                outputs.append(text.strip())
            return outputs

        from mlx_lm import generate as lm_generate

        for prompt in prompts:
            text = lm_generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=self.runtime.max_new_tokens,
                verbose=False,
            )
            outputs.append(str(text).strip())
        return outputs

    def memory_gb(self) -> float | None:
        try:
            import mlx.core as mx

            if hasattr(mx, "get_active_memory"):
                return float(mx.get_active_memory()) / (1024**3)
        except Exception:
            pass
        return None

    def release(self) -> None:
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "tokenizer"):
            del self.tokenizer
        if hasattr(self, "processor"):
            del self.processor
        gc.collect()
        try:
            import mlx.core as mx

            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
        except Exception:
            pass


def create_backend(
    runtime: ModelRuntimeConfig, environment: EnvironmentConfig
) -> InferenceBackend:
    backend_name = resolve_backend(runtime.backend)
    if backend_name == "mlx":
        return MLXBackend(runtime, environment)
    return TorchBackend(runtime, environment)
