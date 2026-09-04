""""Dynamic PubMedQA baseline evaluator for NVIDIA CUDA and Apple Silicon.

Backends
--------
- torch: CUDA -> MPS -> CPU (automatic device resolution)
- mlx: Apple Silicon using mlx-lm; Gemma 3 uses mlx-vlm
- auto: prefers MLX on Apple Silicon when available, otherwise torch

The evaluation protocol is shared across backends:
- same JSONL test/validation file
- same prompt_builder
- same parser
- same metrics
- same result format
"""

from __future__ import annotations

import gc
import importlib.util
import json
import os
import platform
import re
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from pubmedqa.answer_parser import parse_pubmedqa_answer
from pubmedqa.prompt_builder import PubMedQAExample, build_tokenizer_prompt, example_from_record
from pubmedqa.runtime_settings import EVAL_CONFIG

DEFAULT_MODEL_NAME = EVAL_CONFIG.default_model_name
DEFAULT_BASE_MODELS = EVAL_CONFIG.default_base_models

# BF16 MLX conversions intended to preserve model-selection accuracy as much as possible.
DEFAULT_MLX_MODEL_MAP: dict[str, str] = {
    "Qwen/Qwen3-0.6B": "mlx-community/Qwen3-0.6B-bf16",
    "meta-llama/Llama-3.2-1B-Instruct": "mlx-community/Llama-3.2-1B-Instruct-bf16",
    "Qwen/Qwen3-1.7B": "mlx-community/Qwen3-1.7B-bf16",
    "Qwen/Qwen3-4B": "mlx-community/Qwen3-4B-bf16",
    "google/gemma-3-4b-it": "mlx-community/gemma-3-4b-it-bf16",
}

VALID_LABELS = ("yes", "no", "maybe")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


def _env_optional_int(name: str) -> int | None:
    value = os.getenv(name)
    if not value:
        return None
    return int(value)


def _safe_rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


def _is_apple_silicon() -> bool:
    return platform.system() == "Darwin" and platform.machine() in {"arm64", "aarch64"}


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


@dataclass(frozen=True)
class EnvironmentConfig:
    hf_token: str | None = None

    @classmethod
    def from_env(cls) -> "EnvironmentConfig":
        return cls(hf_token=os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN"))


@dataclass(frozen=True)
class ModelRuntimeConfig:
    model_name: str
    backend: str = EVAL_CONFIG.default_backend  # auto | torch | mlx
    device: str = EVAL_CONFIG.default_device  # auto | cuda:0 | mps | cpu
    dtype: str = EVAL_CONFIG.default_dtype  # auto | bf16 | fp16 | fp32
    max_new_tokens: int = EVAL_CONFIG.default_max_new_tokens
    batch_size: int | None = None
    max_input_tokens: int | None = None
    attn_implementation: str | None = EVAL_CONFIG.default_attn_implementation  # auto | sdpa | eager | ... | none
    trust_remote_code: bool = False
    cpu_threads: int = EVAL_CONFIG.default_cpu_threads
    strict_parser: bool = EVAL_CONFIG.default_strict_parser


@dataclass(frozen=True)
class EvalItem:
    index: int
    run_id: str
    model_name: str
    resolved_model_name: str
    backend: str
    condition: str
    title: str
    start_time: str
    end_time: str
    pubid: str
    question: str
    context: list[str]
    gold_label: str
    prompt: str
    response_text: str
    predicted_label: str | None
    parse_method: str | None
    parse_error: str | None


@dataclass(frozen=True)
class EvalSummary:
    run_id: str
    model_name: str
    resolved_model_name: str
    backend: str
    condition: str
    title: str
    start_time: str
    end_time: str
    elapsed_seconds: float
    num_examples: int
    num_parsed: int
    accuracy: float
    macro_f1: float
    invalid_rate: float
    examples_per_second: float
    device: str
    dtype: str
    device_memory_gb: float | None


@dataclass(frozen=True)
class CliBatchConfig:
    test_path: Path
    output_dir: Path
    models: tuple[str, ...]
    expected_test_size: int | None
    condition: str
    run_id: str
    runtime_defaults: ModelRuntimeConfig
    environment: EnvironmentConfig

    @classmethod
    def from_env(cls) -> "CliBatchConfig":
        test_path_raw = os.getenv(EVAL_CONFIG.env_test_path)
        if not test_path_raw:
            raise RuntimeError(
                f"{EVAL_CONFIG.env_test_path} is required. Point it to the model-selection/validation or test JSONL."
            )

        expected_raw = os.getenv(
            EVAL_CONFIG.env_expected_test_size,
            str(EVAL_CONFIG.default_expected_test_size),
        ).strip().lower()
        expected_test_size = None if expected_raw in {"", "none", "off"} else int(expected_raw)

        raw_models = os.getenv(EVAL_CONFIG.env_models)
        if raw_models:
            models = tuple(part.strip() for part in raw_models.split(",") if part.strip())
            if not models:
                raise ValueError(f"{EVAL_CONFIG.env_models} was set but contains no valid model names.")
        else:
            models = DEFAULT_BASE_MODELS

        attn = os.getenv(
            EVAL_CONFIG.env_attn_implementation,
            EVAL_CONFIG.default_attn_implementation,
        ).strip().lower()
        if attn in {"", "none"}:
            attn = None

        batch_raw = os.getenv(EVAL_CONFIG.env_batch_size)
        batch_size = int(batch_raw) if batch_raw else None

        runtime_defaults = ModelRuntimeConfig(
            model_name=models[0],
            backend=os.getenv(EVAL_CONFIG.env_backend, EVAL_CONFIG.default_backend).strip().lower(),
            device=os.getenv(EVAL_CONFIG.env_device, EVAL_CONFIG.default_device).strip().lower(),
            dtype=os.getenv(EVAL_CONFIG.env_dtype, EVAL_CONFIG.default_dtype).strip().lower(),
            max_new_tokens=_env_int(
                EVAL_CONFIG.env_max_new_tokens,
                EVAL_CONFIG.default_max_new_tokens,
            ),
            batch_size=batch_size,
            max_input_tokens=_env_optional_int(EVAL_CONFIG.env_max_input_tokens),
            attn_implementation=attn,
            trust_remote_code=_env_bool(EVAL_CONFIG.env_trust_remote_code, False),
            cpu_threads=_env_int(EVAL_CONFIG.env_cpu_threads, EVAL_CONFIG.default_cpu_threads),
            strict_parser=_env_bool(EVAL_CONFIG.env_strict_parser, EVAL_CONFIG.default_strict_parser),
        )

        return cls(
            test_path=Path(test_path_raw),
            output_dir=Path(os.getenv(EVAL_CONFIG.env_output_dir, str(EVAL_CONFIG.default_output_dir))),
            models=models,
            expected_test_size=expected_test_size,
            condition=os.getenv(EVAL_CONFIG.env_condition, EVAL_CONFIG.default_condition),
            run_id=os.getenv(EVAL_CONFIG.env_run_id) or datetime.now().strftime("%Y%m%d_%H%M%S"),
            runtime_defaults=runtime_defaults,
            environment=EnvironmentConfig.from_env(),
        )


def load_local_jsonl(path: Path, expected_size: int | None = None) -> list[PubMedQAExample]:
    if not path.is_file():
        raise FileNotFoundError(f"JSONL not found: {path}")

    dataset = load_dataset("json", data_files={"eval": str(path)}, split="eval")
    examples = [example_from_record(row) for row in dataset]

    if not examples:
        raise RuntimeError(f"Dataset is empty: {path}")
    if expected_size is not None and len(examples) != expected_size:
        raise RuntimeError(
            f"Expected {expected_size} examples but loaded {len(examples)} from {path}."
        )

    invalid_gold = [example.pubid for example in examples if example.final_decision not in VALID_LABELS]
    if invalid_gold:
        raise RuntimeError(f"Invalid gold labels found, e.g. {', '.join(invalid_gold[:5])}")

    return examples


def resolve_backend(requested: str) -> str:
    requested = requested.strip().lower()
    if requested not in {"auto", "torch", "mlx"}:
        raise ValueError("PUBMEDQA_BACKEND must be one of: auto, torch, mlx")

    if requested == "mlx":
        if not _is_apple_silicon():
            raise RuntimeError("MLX backend requires Apple Silicon.")
        if not _module_available("mlx_lm"):
            raise RuntimeError("MLX backend requested but mlx-lm is not installed.")
        return "mlx"

    if requested == "torch":
        return "torch"

    # auto
    if _is_apple_silicon() and _module_available("mlx_lm"):
        return "mlx"
    return "torch"


def resolve_torch_device(requested: str) -> torch.device:
    requested = requested.strip().lower()
    if requested != "auto":
        device = torch.device(requested)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {requested}")
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


def resolve_batch_size(model_name: str, backend: str, device: str, requested: int | None) -> int:
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
    if device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "current_allocated_memory"):
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

    def __init__(self, runtime: ModelRuntimeConfig, environment: EnvironmentConfig) -> None:
        self.runtime = runtime
        self.environment = environment
        self.resolved_model_name = runtime.model_name
        self.device = resolve_torch_device(runtime.device)
        self.device_label = str(self.device)
        self.dtype = resolve_torch_dtype(self.device, runtime.dtype)
        self.dtype_label = str(self.dtype).replace("torch.", "")
        self.attn_implementation = resolve_attention(self.device, runtime.attn_implementation)
        self.batch_size = resolve_batch_size(runtime.model_name, "torch", self.device.type, runtime.batch_size)

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
            processor = AutoProcessor.from_pretrained(runtime.model_name, **common_kwargs)
            tokenizer = getattr(processor, "tokenizer", None)
            if tokenizer is None:
                raise RuntimeError("Gemma 3 processor does not expose a tokenizer.")
            model = AutoModelForMultimodalLM.from_pretrained(runtime.model_name, **model_kwargs)
        else:
            tokenizer = AutoTokenizer.from_pretrained(runtime.model_name, **common_kwargs)
            model = AutoModelForCausalLM.from_pretrained(runtime.model_name, **model_kwargs)

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
                for text in self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
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

    def __init__(self, runtime: ModelRuntimeConfig, environment: EnvironmentConfig) -> None:
        del environment  # HF auth is handled by the local/HF tooling used by mlx-lm/mlx-vlm.
        if not _is_apple_silicon():
            raise RuntimeError("MLX backend requires Apple Silicon.")

        self.runtime = runtime
        self.resolved_model_name = DEFAULT_MLX_MODEL_MAP.get(runtime.model_name, runtime.model_name)
        self.device_label = "mlx"
        self.dtype_label = "model-native"
        self.batch_size = 1
        self.is_gemma_vlm = runtime.model_name.startswith("google/gemma-3-")

        if self.is_gemma_vlm:
            if not _module_available("mlx_vlm"):
                raise RuntimeError("Gemma 3 MLX evaluation requires mlx-vlm: pip install -U mlx-vlm")
            from mlx_vlm import load as vlm_load

            self.model, self.processor = vlm_load(self.resolved_model_name)
            tokenizer = getattr(self.processor, "tokenizer", None)
            if tokenizer is None:
                raise RuntimeError("MLX-VLM Gemma processor does not expose a tokenizer.")
            self.tokenizer = tokenizer
        else:
            if not _module_available("mlx_lm"):
                raise RuntimeError("MLX backend requires mlx-lm: pip install -U mlx-lm")
            from mlx_lm import load as lm_load

            self.model, self.tokenizer = lm_load(self.resolved_model_name)
            self.processor = None

    def generate(self, prompts: Sequence[str]) -> list[str]:
        outputs: list[str] = []

        if self.is_gemma_vlm:
            from mlx_vlm import generate as vlm_generate
            from mlx_vlm.prompt_utils import apply_chat_template as vlm_apply_chat_template

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


def create_backend(runtime: ModelRuntimeConfig, environment: EnvironmentConfig) -> InferenceBackend:
    backend_name = resolve_backend(runtime.backend)
    if backend_name == "mlx":
        return MLXBackend(runtime, environment)
    return TorchBackend(runtime, environment)


class PubMedQAEvaluationRunner:
    def __init__(
        self,
        *,
        run_id: str,
        model_name: str,
        condition: str,
        output_dir: Path,
        runtime: ModelRuntimeConfig,
        environment: EnvironmentConfig,
    ) -> None:
        self.run_id = run_id
        self.model_name = model_name
        self.condition = condition
        self.output_dir = Path(output_dir)
        self.runtime = runtime
        self.environment = environment

    @property
    def title(self) -> str:
        return build_title(self.run_id, self.model_name, self.condition)

    def evaluate(self, examples: Sequence[PubMedQAExample]) -> tuple[list[EvalItem], EvalSummary]:
        backend = create_backend(self.runtime, self.environment)
        started_at = current_time_iso()
        started = time.perf_counter()

        try:
            prompts = [build_tokenizer_prompt(backend.tokenizer, example) for example in examples]
            response_texts = backend.generate(prompts)
            elapsed = time.perf_counter() - started
            ended_at = current_time_iso()

            if len(response_texts) != len(examples):
                raise RuntimeError(
                    f"Backend returned {len(response_texts)} responses for {len(examples)} examples."
                )

            items: list[EvalItem] = []
            for index, (example, prompt, response_text) in enumerate(
                zip(examples, prompts, response_texts)
            ):
                parsed = parse_pubmedqa_answer(response_text, strict=self.runtime.strict_parser)
                items.append(
                    EvalItem(
                        index=index,
                        run_id=self.run_id,
                        model_name=self.model_name,
                        resolved_model_name=backend.resolved_model_name,
                        backend=backend.backend_name,
                        condition=self.condition,
                        title=self.title,
                        start_time=started_at,
                        end_time=ended_at,
                        pubid=example.pubid,
                        question=example.question,
                        context=list(example.contexts),
                        gold_label=example.final_decision or "",
                        prompt=prompt,
                        response_text=response_text,
                        predicted_label=parsed.label,
                        parse_method=parsed.method,
                        parse_error=parsed.error,
                    )
                )

            gold = [example.final_decision for example in examples]
            predicted = [item.predicted_label for item in items]
            num_parsed = sum(label in VALID_LABELS for label in predicted)

            summary = EvalSummary(
                run_id=self.run_id,
                model_name=self.model_name,
                resolved_model_name=backend.resolved_model_name,
                backend=backend.backend_name,
                condition=self.condition,
                title=self.title,
                start_time=started_at,
                end_time=ended_at,
                elapsed_seconds=elapsed,
                num_examples=len(examples),
                num_parsed=num_parsed,
                accuracy=accuracy(gold, predicted),
                macro_f1=macro_f1(gold, predicted),
                invalid_rate=1.0 - _safe_rate(num_parsed, len(examples)),
                examples_per_second=(len(examples) / elapsed) if elapsed > 0 else 0.0,
                device=backend.device_label,
                dtype=backend.dtype_label,
                device_memory_gb=backend.memory_gb(),
            )
            return items, summary
        finally:
            backend.release()

    def save_results(self, items: Sequence[EvalItem], summary: EvalSummary) -> Path:
        return save_results(self.output_dir, items, summary)


def build_title(run_id: str, model_name: str, condition: str) -> str:
    return f"{run_id}__{safe_name(model_name)}__{safe_name(condition)}"


def accuracy(gold: Sequence[str | None], predicted: Sequence[str | None]) -> float:
    if not gold:
        return 0.0
    return sum(g == p for g, p in zip(gold, predicted)) / len(gold)


def macro_f1(gold: Sequence[str | None], predicted: Sequence[str | None]) -> float:
    scores: list[float] = []
    for label in VALID_LABELS:
        tp = sum(g == label and p == label for g, p in zip(gold, predicted))
        fp = sum(g != label and p == label for g, p in zip(gold, predicted))
        fn = sum(g == label and p != label for g, p in zip(gold, predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    return sum(scores) / len(scores)


def save_results(output_dir: Path, items: Sequence[EvalItem], summary: EvalSummary) -> Path:
    run_dir = output_dir / summary.run_id / safe_name(summary.model_name) / safe_name(summary.condition)
    run_dir.mkdir(parents=True, exist_ok=True)

    write_json(run_dir / "summary.json", asdict(summary))
    write_jsonl(run_dir / "outputs.jsonl", (asdict(item) for item in items))
    write_json(run_dir / "metrics" / "ACC.json", {"name": "ACC", "value": summary.accuracy})
    write_json(run_dir / "metrics" / "Macro_F1.json", {"name": "Macro F1", "value": summary.macro_f1})
    write_json(
        run_dir / "metrics" / "Invalid_rate.json",
        {"name": "Invalid rate", "value": summary.invalid_rate},
    )
    write_json(
        run_dir / "run.json",
        {
            "run_id": summary.run_id,
            "model_name": summary.model_name,
            "resolved_model_name": summary.resolved_model_name,
            "backend": summary.backend,
            "condition": summary.condition,
            "title": summary.title,
            "start_time": summary.start_time,
            "end_time": summary.end_time,
        },
    )
    return run_dir


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False, default=str)
        file.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def current_time_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def configure_parallelism(cpu_threads: int) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    torch.set_num_threads(max(1, cpu_threads))
    try:
        torch.set_num_interop_threads(max(1, min(4, cpu_threads)))
    except RuntimeError:
        pass


def print_runtime(runtime: ModelRuntimeConfig) -> None:
    backend = resolve_backend(runtime.backend)
    print(f"[runtime] backend={backend}")
    print(f"[runtime] apple_silicon={_is_apple_silicon()}")
    print(f"[runtime] cuda_available={torch.cuda.is_available()}")
    print(f"[runtime] mps_available={torch.backends.mps.is_available()}")

    if backend == "torch":
        device = resolve_torch_device(runtime.device)
        dtype = resolve_torch_dtype(device, runtime.dtype)
        attn = resolve_attention(device, runtime.attn_implementation)
        print(f"[runtime] device={device}")
        if device.type == "cuda":
            print(f"[runtime] gpu={torch.cuda.get_device_name(0)}")
        print(f"[runtime] dtype={dtype}")
        print(f"[runtime] attention={attn or 'model-default'}")
    else:
        print("[runtime] device=mlx")
        print("[runtime] dtype=model-native")

    print(f"[runtime] cpu_threads={runtime.cpu_threads}")


def main() -> None:
    config = CliBatchConfig.from_env()
    configure_parallelism(config.runtime_defaults.cpu_threads)
    print_runtime(config.runtime_defaults)
    print(f"[runtime] hf_token_set={bool(config.environment.hf_token)}")

    examples = load_local_jsonl(config.test_path, config.expected_test_size)
    print(f"[dataset] path={config.test_path}")
    print(f"[dataset] num_examples={len(examples)}")

    summaries: list[dict[str, Any]] = []

    # Models are intentionally evaluated sequentially. Each backend performs its own
    # device-level parallelism/batching, avoiding cross-model memory contention.
    for model_name in config.models:
        runtime = replace(config.runtime_defaults, model_name=model_name)
        resolved_backend = resolve_backend(runtime.backend)

        if resolved_backend == "torch":
            device = resolve_torch_device(runtime.device)
            batch_size = resolve_batch_size(model_name, "torch", device.type, runtime.batch_size)
        else:
            batch_size = 1

        print(f"\n[model] evaluating {model_name}")
        print(f"[model] backend={resolved_backend} batch_size={batch_size}")
        if resolved_backend == "mlx":
            print(f"[model] resolved_model={DEFAULT_MLX_MODEL_MAP.get(model_name, model_name)}")

        runner = PubMedQAEvaluationRunner(
            run_id=config.run_id,
            model_name=model_name,
            condition=config.condition,
            output_dir=config.output_dir,
            runtime=runtime,
            environment=config.environment,
        )

        items, summary = runner.evaluate(examples)
        run_dir = runner.save_results(items, summary)
        summaries.append(asdict(summary))

        print(
            f"[result] acc={summary.accuracy:.4f} "
            f"macro_f1={summary.macro_f1:.4f} "
            f"invalid={summary.invalid_rate:.4f} "
            f"time={summary.elapsed_seconds:.1f}s "
            f"examples/s={summary.examples_per_second:.2f} "
            f"saved={run_dir}"
        )

    write_json(config.output_dir / config.run_id / "all_models_summary.json", summaries)


if __name__ == "__main__":
    main()
