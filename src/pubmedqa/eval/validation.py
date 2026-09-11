"""Validation metric assembly and artifact persistence for training runs."""

from __future__ import annotations

import gc
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import DataLoader

from pubmedqa.data.prompts import build_tokenizer_prompt
from pubmedqa.data.records import (
    VALID_LABELS,
    PubMedQAExample,
    write_json,
    write_jsonl,
)
from pubmedqa.data.supervised import (
    PubMedQASupervisedDataset,
    build_eval_dataloader,
    validate_targets,
)
from pubmedqa.eval.metrics import (
    accuracy,
    classwise_f1,
    confusion_matrix,
    macro_f1,
    parse_pubmedqa_answer,
    resolve_metric_labels,
)
from pubmedqa.model.device import (
    memory_snapshot as _memory_snapshot,
)
from pubmedqa.model.device import (
    supports_cuda_amp,
)


@dataclass(frozen=True)
class EvalPrediction:
    pubid: str
    gold_label: str
    predicted_label: str | None
    response_text: str
    parse_method: str | None
    parse_error: str | None


@dataclass(frozen=True)
class EvalMetrics:
    split: str
    loss: float
    accuracy: float
    macro_f1: float
    class_f1: dict[str, float]
    confusion_matrix: dict[str, dict[str, int]]
    label_order: tuple[str, ...]
    invalid_rate: float
    num_examples: int
    num_parsed: int
    elapsed_seconds: float
    avg_latency_seconds: float
    examples_per_second: float
    tokens_per_second: float
    peak_allocated_gb: float | None
    peak_reserved_gb: float | None
    loss_reduction: str = "token_mean"


@dataclass(frozen=True)
class EvalResult:
    metrics: EvalMetrics
    predictions: list[EvalPrediction]


def build_eval_result(
    *,
    split_name: str,
    loss: float,
    predictions: Sequence[EvalPrediction],
    elapsed_seconds: float,
    input_tokens: int,
    peak_allocated_gb: float | None,
    peak_reserved_gb: float | None,
) -> EvalResult:
    """Build validation metrics using only labels present in the gold split."""

    prediction_list = list(predictions)
    gold = [item.gold_label for item in prediction_list]
    predicted = [item.predicted_label for item in prediction_list]
    metric_labels = resolve_metric_labels(gold)
    num_parsed = sum(label in VALID_LABELS for label in predicted)
    confusion_labels = list(metric_labels)
    for label in VALID_LABELS:
        if label in predicted and label not in confusion_labels:
            confusion_labels.append(label)
    confusion_labels.append("invalid")
    count = len(prediction_list)

    return EvalResult(
        metrics=EvalMetrics(
            split=split_name,
            loss=loss,
            accuracy=accuracy(gold, predicted),
            macro_f1=macro_f1(gold, predicted, labels=metric_labels),
            class_f1=classwise_f1(gold, predicted, labels=metric_labels),
            confusion_matrix=confusion_matrix(
                gold, predicted, labels=tuple(confusion_labels)
            ),
            label_order=tuple(confusion_labels),
            invalid_rate=1.0 - (num_parsed / count) if count else 0.0,
            num_examples=count,
            num_parsed=num_parsed,
            elapsed_seconds=elapsed_seconds,
            avg_latency_seconds=(elapsed_seconds / count) if count else 0.0,
            examples_per_second=(count / elapsed_seconds)
            if elapsed_seconds > 0
            else 0.0,
            tokens_per_second=(input_tokens / elapsed_seconds)
            if elapsed_seconds > 0
            else 0.0,
            peak_allocated_gb=peak_allocated_gb,
            peak_reserved_gb=peak_reserved_gb,
        ),
        predictions=prediction_list,
    )


def write_eval_result(
    evaluations_dir: Path, split_name: str, result: EvalResult
) -> None:
    """Persist the stable validation summary and prediction artifact schemas."""

    write_json(evaluations_dir / f"{split_name}_summary.json", asdict(result.metrics))
    write_jsonl(
        evaluations_dir / f"{split_name}_predictions.jsonl",
        (asdict(prediction) for prediction in result.predictions),
    )


@dataclass(frozen=True)
class EvaluationSettings:
    """Evaluation inputs only; a CPU view never mutates the training session."""

    device: torch.device
    dtype: Any
    eval_batch_size: int
    max_input_tokens: int | None
    max_new_tokens: int
    num_workers: int
    strict_parser: bool

    @classmethod
    def from_config(cls, config, device):
        fsdp = config.distributed_mode == "fsdp"
        return cls(
            torch.device("cpu") if fsdp else device,
            torch.float32 if fsdp else config.dtype,
            config.eval_batch_size,
            config.max_input_tokens,
            config.max_new_tokens,
            config.num_workers,
            config.strict_parser,
        )

    def autocast_context(self):
        from contextlib import nullcontext

        if not supports_cuda_amp(self.dtype, self.device):
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.dtype, enabled=True)


def evaluate_split(
    *,
    model,
    tokenizer,
    supervised_loader,
    examples,
    split_name,
    settings,
    evaluations_dir,
) -> EvalResult:
    was_training = model.training
    padding_side = tokenizer.padding_side
    try:
        return _evaluate_split(
            model=model,
            tokenizer=tokenizer,
            supervised_loader=supervised_loader,
            examples=examples,
            split_name=split_name,
            settings=settings,
            evaluations_dir=evaluations_dir,
        )
    finally:
        # Include failures in tokenization, forward, generation and artifact I/O.
        tokenizer.padding_side = padding_side
        model.train(was_training)


def _evaluate_split(
    *,
    settings,
    evaluations_dir,
    model: torch.nn.Module,
    tokenizer: Any,
    supervised_loader: DataLoader[Any],
    examples: Sequence[PubMedQAExample],
    split_name: str,
) -> EvalResult:
    model.eval()
    evaluation_started = time.perf_counter()
    loss_total = 0.0
    loss_count = 0
    predictions: list[EvalPrediction] = []
    total_input_tokens = 0
    total_generation_time = 0.0

    if settings.device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(settings.device)

    with torch.no_grad():
        for batch in supervised_loader:
            target_count = validate_targets(batch["labels"])
            batch = {key: value.to(settings.device) for key, value in batch.items()}
            with settings.autocast_context():
                outputs = model(**batch)
            batch_loss = float(outputs.loss.detach().item())
            if not math.isfinite(batch_loss):
                raise ValueError("nonfinite validation loss")
            # HF causal-LM loss is a mean over nonignored shifted labels. Weight
            # it by that same denominator, never by batch count or padded width.
            loss_total += batch_loss * target_count
            loss_count += target_count

        if loss_count == 0:
            raise ValueError("Empty supervised evaluation loader")
        tokenizer.padding_side = "left"
        for batch_start in range(0, len(examples), settings.eval_batch_size):
            batch_examples = examples[
                batch_start : batch_start + settings.eval_batch_size
            ]
            prompts = [
                build_tokenizer_prompt(tokenizer, example, include_answer=False)
                for example in batch_examples
            ]
            tokenize_kwargs: dict[str, Any] = {
                "return_tensors": "pt",
                "padding": True,
                "truncation": settings.max_input_tokens is not None,
            }
            if settings.max_input_tokens is not None:
                tokenize_kwargs["max_length"] = settings.max_input_tokens
            encoded = tokenizer(prompts, **tokenize_kwargs)
            encoded = {
                key: value.to(settings.device)
                for key, value in encoded.items()
                if isinstance(value, torch.Tensor)
            }
            input_width = encoded["input_ids"].shape[1]
            total_input_tokens += int(encoded["attention_mask"].sum().item())
            started = time.perf_counter()
            generated = model.generate(
                **encoded,
                max_new_tokens=settings.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
            elapsed = time.perf_counter() - started
            total_generation_time += elapsed
            responses = tokenizer.batch_decode(
                generated[:, input_width:], skip_special_tokens=True
            )
            for example, response_text in zip(batch_examples, responses):
                parsed = parse_pubmedqa_answer(
                    response_text.strip(), strict=settings.strict_parser
                )
                predictions.append(
                    EvalPrediction(
                        pubid=example.pubid,
                        gold_label=example.final_decision or "",
                        predicted_label=parsed.label,
                        response_text=response_text.strip(),
                        parse_method=parsed.method,
                        parse_error=parsed.error,
                    )
                )

    peak_memory = _memory_snapshot(settings.device)
    total_elapsed = time.perf_counter() - evaluation_started
    result = build_eval_result(
        split_name=split_name,
        loss=(loss_total / loss_count) if loss_count > 0 else 0.0,
        predictions=predictions,
        elapsed_seconds=total_elapsed,
        input_tokens=total_input_tokens,
        peak_allocated_gb=peak_memory["max_allocated_gb"],
        peak_reserved_gb=peak_memory["max_reserved_gb"],
    )
    write_eval_result(evaluations_dir, split_name, result)
    return result


def evaluate_checkpoint_on_main(
    *,
    session,
    settings,
    evaluations_dir,
    load_model,
    checkpoint_dir: Path,
    examples: Sequence[PubMedQAExample],
    split_name: str,
) -> EvalResult:
    def evaluate_checkpoint() -> EvalResult:
        eval_tokenizer, eval_model = load_model(str(checkpoint_dir))
        try:
            eval_dataset = PubMedQASupervisedDataset(examples, eval_tokenizer)
            eval_loader = build_eval_dataloader(
                eval_dataset,
                eval_tokenizer,
                batch_size=settings.eval_batch_size,
                num_workers=settings.num_workers,
                max_input_tokens=settings.max_input_tokens,
            )
            return evaluate_split(
                settings=settings,
                evaluations_dir=evaluations_dir,
                model=eval_model,
                tokenizer=eval_tokenizer,
                supervised_loader=eval_loader,
                examples=examples,
                split_name=split_name,
            )
        finally:
            del eval_model
            gc.collect()
            if session.device.type == "cuda":
                torch.cuda.empty_cache()

    return session.run_on_main_process(
        evaluate_checkpoint,
        operation_name=f"{split_name} checkpoint evaluation",
    )
