"""Dynamic PubMedQA baseline evaluator for NVIDIA CUDA and Apple Silicon.

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

import os
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import torch
from datasets import load_dataset

from pubmedqa.config import EnvironmentConfig
from pubmedqa.config import env_bool as _env_bool
from pubmedqa.config import env_int as _env_int
from pubmedqa.config import env_optional_int as _env_optional_int
from pubmedqa.config.eval import EVAL_CONFIG, ModelRuntimeConfig
from pubmedqa.data.prompts import build_tokenizer_prompt
from pubmedqa.data.records import (
    VALID_LABELS,
    PubMedQAExample,
    current_time_iso,
    example_from_record,
    safe_name,
    write_json,
    write_jsonl,
)

# Preserve the historical import while data owns record loading.
from pubmedqa.data.records import load_local_jsonl as load_local_jsonl
from pubmedqa.eval.backends import (
    DEFAULT_MLX_MODEL_MAP,
    InferenceBackend,
    MLXBackend,
    TorchBackend,
    create_backend,
    empty_device_cache,
    get_device_memory_gb,
    is_apple_silicon,
    reset_device_memory_stats,
    resolve_attention,
    resolve_backend,
    resolve_batch_size,
    resolve_torch_device,
    resolve_torch_dtype,
    synchronize_device,
)
from pubmedqa.eval.metrics import (
    accuracy,
    macro_f1,
    parse_pubmedqa_answer,
    resolve_metric_labels,
)
from pubmedqa.model.device import configure_parallelism


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


DEFAULT_MODEL_NAME = EVAL_CONFIG.default_model_name
DEFAULT_BASE_MODELS = EVAL_CONFIG.default_base_models


def _safe_rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


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

        expected_raw = (
            os.getenv(
                EVAL_CONFIG.env_expected_test_size,
                str(EVAL_CONFIG.default_expected_test_size),
            )
            .strip()
            .lower()
        )
        expected_test_size = (
            None if expected_raw in {"", "none", "off"} else int(expected_raw)
        )

        raw_models = os.getenv(EVAL_CONFIG.env_models)
        if raw_models:
            models = tuple(
                part.strip() for part in raw_models.split(",") if part.strip()
            )
            if not models:
                raise ValueError(
                    f"{EVAL_CONFIG.env_models} was set but contains no valid model names."
                )
        else:
            models = DEFAULT_BASE_MODELS

        attn = (
            os.getenv(
                EVAL_CONFIG.env_attn_implementation,
                EVAL_CONFIG.default_attn_implementation,
            )
            .strip()
            .lower()
        )
        if attn in {"", "none"}:
            attn = None

        batch_raw = os.getenv(EVAL_CONFIG.env_batch_size)
        batch_size = int(batch_raw) if batch_raw else None

        runtime_defaults = ModelRuntimeConfig(
            model_name=models[0],
            backend=os.getenv(EVAL_CONFIG.env_backend, EVAL_CONFIG.default_backend)
            .strip()
            .lower(),
            device=os.getenv(EVAL_CONFIG.env_device, EVAL_CONFIG.default_device)
            .strip()
            .lower(),
            dtype=os.getenv(EVAL_CONFIG.env_dtype, EVAL_CONFIG.default_dtype)
            .strip()
            .lower(),
            max_new_tokens=_env_int(
                EVAL_CONFIG.env_max_new_tokens,
                EVAL_CONFIG.default_max_new_tokens,
            ),
            batch_size=batch_size,
            max_input_tokens=_env_optional_int(EVAL_CONFIG.env_max_input_tokens),
            attn_implementation=attn,
            trust_remote_code=_env_bool(EVAL_CONFIG.env_trust_remote_code, False),
            cpu_threads=_env_int(
                EVAL_CONFIG.env_cpu_threads, EVAL_CONFIG.default_cpu_threads
            ),
            strict_parser=_env_bool(
                EVAL_CONFIG.env_strict_parser, EVAL_CONFIG.default_strict_parser
            ),
        )

        return cls(
            test_path=Path(test_path_raw),
            output_dir=Path(
                os.getenv(
                    EVAL_CONFIG.env_output_dir, str(EVAL_CONFIG.default_output_dir)
                )
            ),
            models=models,
            expected_test_size=expected_test_size,
            condition=os.getenv(
                EVAL_CONFIG.env_condition, EVAL_CONFIG.default_condition
            ),
            run_id=os.getenv(EVAL_CONFIG.env_run_id)
            or datetime.now().strftime("%Y%m%d_%H%M%S"),
            runtime_defaults=runtime_defaults,
            environment=EnvironmentConfig.from_env(),
        )


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

    def evaluate(
        self, examples: Sequence[PubMedQAExample]
    ) -> tuple[list[EvalItem], EvalSummary]:
        backend = create_backend(self.runtime, self.environment)
        started_at = current_time_iso()
        started = time.perf_counter()

        try:
            prompts = [
                build_tokenizer_prompt(backend.tokenizer, example)
                for example in examples
            ]
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
                parsed = parse_pubmedqa_answer(
                    response_text, strict=self.runtime.strict_parser
                )
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


def save_results(
    output_dir: Path, items: Sequence[EvalItem], summary: EvalSummary
) -> Path:
    run_dir = (
        output_dir
        / summary.run_id
        / safe_name(summary.model_name)
        / safe_name(summary.condition)
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    write_json(run_dir / "summary.json", asdict(summary))
    write_jsonl(run_dir / "outputs.jsonl", (asdict(item) for item in items))
    write_json(
        run_dir / "metrics" / "ACC.json", {"name": "ACC", "value": summary.accuracy}
    )
    write_json(
        run_dir / "metrics" / "Macro_F1.json",
        {"name": "Macro F1", "value": summary.macro_f1},
    )
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


def print_runtime(runtime: ModelRuntimeConfig) -> None:
    backend = resolve_backend(runtime.backend)
    print(f"[runtime] backend={backend}")
    print(f"[runtime] apple_silicon={is_apple_silicon()}")
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
            batch_size = resolve_batch_size(
                model_name, "torch", device.type, runtime.batch_size
            )
        else:
            batch_size = 1

        print(f"\n[model] evaluating {model_name}")
        print(f"[model] backend={resolved_backend} batch_size={batch_size}")
        if resolved_backend == "mlx":
            print(
                f"[model] resolved_model={DEFAULT_MLX_MODEL_MAP.get(model_name, model_name)}"
            )

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
