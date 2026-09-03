"""Torch-based PubMedQA evaluation for baseline QA experiments.

Core evaluation objects are configured explicitly and receive environment
credentials via injected objects. The CLI entrypoint may still read environment
variables, but model loading, prompting, inference, metrics, and persistence do
not depend on implicit global state.
"""

from __future__ import annotations

import gc
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from pubmedqa.answer_parser import parse_pubmedqa_answer
from pubmedqa.prompt_builder import (
    PubMedQAExample,
    build_tokenizer_prompt,
    example_from_record,
)

DEFAULT_BASE_MODELS: tuple[str, ...] = (
    "Qwen/Qwen3-0.6B",
    "meta-llama/Llama-3.2-1B-Instruct",
    "Qwen/Qwen3-1.7B",
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen3-4B",
    "google/gemma-3-4b-it",
)
VALID_LABELS = ("yes", "no", "maybe")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


def _resolve_dtype(name: str) -> torch.dtype:
    normalized = name.strip().lower()
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported dtype {name!r}. Use bf16, fp16, or fp32.")
    return mapping[normalized]


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


@dataclass(frozen=True)
class EnvironmentConfig:
    hf_token: str | None = None

    @classmethod
    def from_env(cls) -> "EnvironmentConfig":
        return cls(hf_token=os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN"))


@dataclass(frozen=True)
class ModelRuntimeConfig:
    model_name: str
    torch_dtype: torch.dtype = torch.bfloat16
    device_map: str | None = None
    max_new_tokens: int = 4
    batch_size: int = 8
    max_input_tokens: int | None = None
    use_cuda: bool = True
    attn_implementation: str | None = "sdpa"
    trust_remote_code: bool = False
    cpu_threads: int = max(1, (os.cpu_count() or 1) - 2)
    strict_parser: bool = False


@dataclass(frozen=True)
class EvalItem:
    index: int
    run_id: str
    model_name: str
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
    peak_cuda_memory_gb: float | None


@dataclass
class ModelBundle:
    tokenizer: Any
    model: Any
    device: torch.device
    uses_device_map: bool


@dataclass(frozen=True)
class BatchRunContext:
    run_id: str
    model_name: str
    condition: str
    title: str
    start_time: str
    end_time: str


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
        test_path_raw = os.getenv("PUBMEDQA_TEST_PATH")
        if not test_path_raw:
            raise RuntimeError(
                "PUBMEDQA_TEST_PATH is required. Point it to the PubMedQA test JSONL."
            )

        expected_raw = os.getenv("PUBMEDQA_EXPECTED_TEST_SIZE", "500").strip().lower()
        expected_test_size = None if expected_raw in {"", "none", "off"} else int(expected_raw)

        raw_models = os.getenv("PUBMEDQA_MODELS")
        if raw_models:
            models = tuple(part.strip() for part in raw_models.split(",") if part.strip())
            if not models:
                raise ValueError("PUBMEDQA_MODELS was set but contains no valid model names.")
        else:
            models = DEFAULT_BASE_MODELS

        attn = os.getenv("PUBMEDQA_ATTN_IMPLEMENTATION", "sdpa").strip()
        if attn.lower() in {"", "none", "auto"}:
            attn = None

        runtime_defaults = ModelRuntimeConfig(
            model_name=models[0],
            torch_dtype=_resolve_dtype(os.getenv("PUBMEDQA_DTYPE", "bf16")),
            device_map=os.getenv("PUBMEDQA_DEVICE_MAP", "").strip() or None,
            max_new_tokens=_env_int("PUBMEDQA_MAX_NEW_TOKENS", 4),
            batch_size=_env_int("PUBMEDQA_BATCH_SIZE", 8),
            max_input_tokens=(
                int(os.getenv("PUBMEDQA_MAX_INPUT_TOKENS"))
                if os.getenv("PUBMEDQA_MAX_INPUT_TOKENS")
                else None
            ),
            use_cuda=_env_bool("PUBMEDQA_USE_CUDA", True),
            attn_implementation=attn,
            trust_remote_code=_env_bool("PUBMEDQA_TRUST_REMOTE_CODE", False),
            cpu_threads=_env_int("PUBMEDQA_CPU_THREADS", max(1, (os.cpu_count() or 1) - 2)),
            strict_parser=_env_bool("PUBMEDQA_STRICT_PARSER", False),
        )

        return cls(
            test_path=Path(test_path_raw),
            output_dir=Path(os.getenv("PUBMEDQA_OUTPUT_DIR", "outputs/pubmedqa_eval")),
            models=models,
            expected_test_size=expected_test_size,
            condition=os.getenv("PUBMEDQA_CONDITION", "baseline"),
            run_id=os.getenv("PUBMEDQA_RUN_ID") or datetime.now().strftime("%Y%m%d_%H%M%S"),
            runtime_defaults=runtime_defaults,
            environment=EnvironmentConfig.from_env(),
        )


def load_local_jsonl(path: Path, expected_size: int | None = None) -> list[PubMedQAExample]:
    if not path.is_file():
        raise FileNotFoundError(f"Test JSONL not found: {path}")

    dataset = load_dataset("json", data_files={"test": str(path)}, split="test")
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


def resolve_device(use_cuda: bool) -> torch.device:
    if use_cuda and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def load_model_bundle(
    runtime: ModelRuntimeConfig,
    environment: EnvironmentConfig,
) -> ModelBundle:
    device = resolve_device(runtime.use_cuda)
    uses_device_map = bool(runtime.device_map)

    common_kwargs: dict[str, Any] = {
        "token": environment.hf_token,
        "trust_remote_code": runtime.trust_remote_code,
    }
    model_kwargs: dict[str, Any] = {
        **common_kwargs,
        "torch_dtype": runtime.torch_dtype,
    }
    if runtime.attn_implementation is not None:
        model_kwargs["attn_implementation"] = runtime.attn_implementation
    if uses_device_map:
        model_kwargs["device_map"] = runtime.device_map

    if runtime.model_name.startswith("google/gemma-3-"):
        try:
            from transformers import AutoModelForMultimodalLM, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Gemma 3 requires a recent transformers version with multimodal support."
            ) from exc

        processor = AutoProcessor.from_pretrained(runtime.model_name, **common_kwargs)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("Gemma 3 AutoProcessor does not expose a tokenizer.")
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

    if not uses_device_map:
        model.to(device)
    model.eval()
    return ModelBundle(tokenizer=tokenizer, model=model, device=device, uses_device_map=uses_device_map)


class PubMedQAEvaluationRunner:
    def __init__(
        self,
        *,
        run_id: str,
        model_name: str,
        condition: str,
        output_dir: Path,
        runtime: ModelRuntimeConfig,
        environment: EnvironmentConfig | None = None,
        hf_token: str | None = None,
    ) -> None:
        self.run_id = run_id
        self.model_name = model_name
        self.condition = condition
        self.output_dir = Path(output_dir)
        self.runtime = runtime
        self.environment = environment or EnvironmentConfig(hf_token=hf_token)

    @property
    def title(self) -> str:
        return build_title(self.run_id, self.model_name, self.condition)

    def load_model(self) -> tuple[Any, Any]:
        bundle = load_model_bundle(self.runtime, self.environment)
        return bundle.tokenizer, bundle.model

    def load_bundle(self) -> ModelBundle:
        return load_model_bundle(self.runtime, self.environment)

    def load_local_jsonl(self, path: Path, expected_size: int | None = None) -> list[PubMedQAExample]:
        return load_local_jsonl(path, expected_size)

    def evaluate(
        self,
        examples: Sequence[PubMedQAExample],
        *,
        tokenizer: Any | None = None,
        model: Any | None = None,
    ) -> tuple[list[EvalItem], EvalSummary]:
        if tokenizer is None or model is None:
            bundle = self.load_bundle()
            owns_bundle = True
        else:
            bundle = ModelBundle(
                tokenizer=tokenizer,
                model=model,
                device=resolve_device(self.runtime.use_cuda),
                uses_device_map=bool(self.runtime.device_map),
            )
            owns_bundle = False

        if bundle.device.type == "cuda" and not bundle.uses_device_map:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(bundle.device)
            torch.cuda.synchronize(bundle.device)

        started_at = current_time_iso()
        started = time.perf_counter()
        items = self._generate_all(examples, bundle, start_time=started_at)
        if bundle.device.type == "cuda" and not bundle.uses_device_map:
            torch.cuda.synchronize(bundle.device)
        elapsed = time.perf_counter() - started
        ended_at = current_time_iso()

        title = self.title
        items = [
            EvalItem(
                **{
                    **asdict(item),
                    "title": title,
                    "end_time": ended_at,
                }
            )
            for item in items
        ]

        gold = [example.final_decision for example in examples]
        predicted = [item.predicted_label for item in items]
        num_parsed = sum(label in VALID_LABELS for label in predicted)
        peak_gb: float | None = None
        if bundle.device.type == "cuda" and not bundle.uses_device_map:
            peak_gb = torch.cuda.max_memory_allocated(bundle.device) / (1024**3)

        summary = EvalSummary(
            run_id=self.run_id,
            model_name=self.model_name,
            condition=self.condition,
            title=title,
            start_time=started_at,
            end_time=ended_at,
            elapsed_seconds=elapsed,
            num_examples=len(examples),
            num_parsed=num_parsed,
            accuracy=accuracy(gold, predicted),
            macro_f1=macro_f1(gold, predicted),
            invalid_rate=1.0 - _safe_rate(num_parsed, len(examples)),
            examples_per_second=(len(examples) / elapsed) if elapsed > 0 else 0.0,
            device=str(bundle.device),
            dtype=str(self.runtime.torch_dtype).replace("torch.", ""),
            peak_cuda_memory_gb=peak_gb,
        )

        if owns_bundle:
            self.release_bundle(bundle)
        return items, summary

    def save_results(self, items: Sequence[EvalItem], summary: EvalSummary) -> Path:
        return save_results(self.output_dir, items, summary)

    def _generate_all(
        self,
        examples: Sequence[PubMedQAExample],
        bundle: ModelBundle,
        *,
        start_time: str,
    ) -> list[EvalItem]:
        items: list[EvalItem] = []
        batch_size = max(1, self.runtime.batch_size)

        for batch_start in range(0, len(examples), batch_size):
            batch = examples[batch_start : batch_start + batch_size]
            prompts = [build_tokenizer_prompt(bundle.tokenizer, example) for example in batch]

            tokenize_kwargs: dict[str, Any] = {
                "return_tensors": "pt",
                "padding": True,
                "truncation": self.runtime.max_input_tokens is not None,
            }
            if self.runtime.max_input_tokens is not None:
                tokenize_kwargs["max_length"] = self.runtime.max_input_tokens

            encoded = bundle.tokenizer(prompts, **tokenize_kwargs)
            encoded = {
                key: value.to(bundle.device, non_blocking=True)
                for key, value in encoded.items()
                if isinstance(value, torch.Tensor)
            }
            input_width = encoded["input_ids"].shape[1]

            generation_kwargs: dict[str, Any] = {
                "max_new_tokens": self.runtime.max_new_tokens,
                "do_sample": False,
                "pad_token_id": bundle.tokenizer.pad_token_id,
                "use_cache": True,
            }
            if bundle.tokenizer.eos_token_id is not None:
                generation_kwargs["eos_token_id"] = bundle.tokenizer.eos_token_id

            with torch.inference_mode():
                generated = bundle.model.generate(**encoded, **generation_kwargs)

            response_ids = generated[:, input_width:]
            response_texts = bundle.tokenizer.batch_decode(response_ids, skip_special_tokens=True)

            for offset, (example, prompt, text) in enumerate(zip(batch, prompts, response_texts)):
                response_text = text.strip()
                parsed = parse_pubmedqa_answer(response_text, strict=self.runtime.strict_parser)
                items.append(
                    EvalItem(
                        index=batch_start + offset,
                        run_id=self.run_id,
                        model_name=self.model_name,
                        condition=self.condition,
                        title=self.title,
                        start_time=start_time,
                        end_time="",
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
        return items

    @staticmethod
    def release_bundle(bundle: ModelBundle) -> None:
        del bundle.model
        del bundle.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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
        json.dump(data, file, indent=2, ensure_ascii=False)
        file.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def main() -> None:
    config = CliBatchConfig.from_env()
    configure_parallelism(config.runtime_defaults.cpu_threads)

    device = resolve_device(config.runtime_defaults.use_cuda)
    print(f"[runtime] device={device}")
    print(f"[runtime] cuda_available={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[runtime] gpu={torch.cuda.get_device_name(0)}")
    print(f"[runtime] dtype={config.runtime_defaults.torch_dtype}")
    print(f"[runtime] batch_size={config.runtime_defaults.batch_size}")
    print(f"[runtime] cpu_threads={config.runtime_defaults.cpu_threads}")
    print(f"[runtime] hf_token_set={bool(config.environment.hf_token)}")

    examples = load_local_jsonl(config.test_path, config.expected_test_size)
    print(f"[dataset] test_path={config.test_path}")
    print(f"[dataset] num_test_examples={len(examples)}")

    summaries: list[dict[str, Any]] = []
    for model_name in config.models:
        runtime = ModelRuntimeConfig(
            model_name=model_name,
            torch_dtype=config.runtime_defaults.torch_dtype,
            device_map=config.runtime_defaults.device_map,
            max_new_tokens=config.runtime_defaults.max_new_tokens,
            batch_size=config.runtime_defaults.batch_size,
            max_input_tokens=config.runtime_defaults.max_input_tokens,
            use_cuda=config.runtime_defaults.use_cuda,
            attn_implementation=config.runtime_defaults.attn_implementation,
            trust_remote_code=config.runtime_defaults.trust_remote_code,
            cpu_threads=config.runtime_defaults.cpu_threads,
            strict_parser=config.runtime_defaults.strict_parser,
        )
        runner = PubMedQAEvaluationRunner(
            run_id=config.run_id,
            model_name=model_name,
            condition=config.condition,
            output_dir=config.output_dir,
            runtime=runtime,
            environment=config.environment,
        )
        print(f"\n[model] evaluating {model_name}")
        items, summary = runner.evaluate(examples)
        run_dir = runner.save_results(items, summary)
        summaries.append(asdict(summary))
        print(
            f"[result] title={summary.title} "
            f"acc={summary.accuracy:.4f} "
            f"macro_f1={summary.macro_f1:.4f} "
            f"invalid={summary.invalid_rate:.4f} "
            f"time={summary.elapsed_seconds:.1f}s "
            f"saved={run_dir}"
        )

    write_json(config.output_dir / config.run_id / "all_models_summary.json", summaries)


if __name__ == "__main__":
    main()
