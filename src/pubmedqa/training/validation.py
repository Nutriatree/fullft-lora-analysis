"""Validation metric assembly and artifact persistence for training runs."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Sequence

from pubmedqa.domain.labels import VALID_LABELS
from pubmedqa.domain.metrics import accuracy, classwise_f1, confusion_matrix, macro_f1, resolve_metric_labels
from pubmedqa.runtime.io import write_json, write_jsonl
from pubmedqa.training.contracts import EvalMetrics, EvalPrediction, EvalResult


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
            confusion_matrix=confusion_matrix(gold, predicted, labels=tuple(confusion_labels)),
            label_order=tuple(confusion_labels),
            invalid_rate=1.0 - (num_parsed / count) if count else 0.0,
            num_examples=count,
            num_parsed=num_parsed,
            elapsed_seconds=elapsed_seconds,
            avg_latency_seconds=(elapsed_seconds / count) if count else 0.0,
            examples_per_second=(count / elapsed_seconds) if elapsed_seconds > 0 else 0.0,
            tokens_per_second=(input_tokens / elapsed_seconds) if elapsed_seconds > 0 else 0.0,
            peak_allocated_gb=peak_allocated_gb,
            peak_reserved_gb=peak_reserved_gb,
        ),
        predictions=prediction_list,
    )


def write_eval_result(evaluations_dir: Path, split_name: str, result: EvalResult) -> None:
    """Persist the stable validation summary and prediction artifact schemas."""

    write_json(evaluations_dir / f"{split_name}_summary.json", asdict(result.metrics))
    write_jsonl(
        evaluations_dir / f"{split_name}_predictions.jsonl",
        (asdict(prediction) for prediction in result.predictions),
    )
