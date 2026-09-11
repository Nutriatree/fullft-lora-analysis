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

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from pubmedqa.config.env import EnvironmentConfig
from pubmedqa.config.eval import ModelRuntimeConfig
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

from pubmedqa.eval.backends import create_backend
from pubmedqa.eval.metrics import (
    accuracy,
    macro_f1,
    parse_pubmedqa_answer,
    resolve_metric_labels,
)


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


def _safe_rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else numerator / denominator


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
