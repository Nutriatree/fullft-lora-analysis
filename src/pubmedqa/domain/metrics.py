"""Framework-independent metrics for PubMedQA labels."""

from __future__ import annotations

from collections.abc import Sequence

from pubmedqa.domain.labels import VALID_LABELS


def resolve_metric_labels(gold: Sequence[str | None]) -> tuple[str, ...]:
    labels = tuple(label for label in VALID_LABELS if any(item == label for item in gold))
    return labels or VALID_LABELS


def accuracy(gold: Sequence[str | None], predicted: Sequence[str | None]) -> float:
    if not gold:
        return 0.0
    return sum(g == p for g, p in zip(gold, predicted)) / len(gold)


def classwise_f1(
    gold: Sequence[str | None],
    predicted: Sequence[str | None],
    *,
    labels: Sequence[str] | None = None,
) -> dict[str, float]:
    metric_labels = tuple(labels) if labels is not None else resolve_metric_labels(gold)
    scores: dict[str, float] = {}
    for label in metric_labels:
        true_positive = sum(g == label and p == label for g, p in zip(gold, predicted))
        false_positive = sum(g != label and p == label for g, p in zip(gold, predicted))
        false_negative = sum(g == label and p != label for g, p in zip(gold, predicted))
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        scores[label] = (
            0.0
            if precision + recall == 0.0
            else 2 * precision * recall / (precision + recall)
        )
    return scores


def macro_f1(
    gold: Sequence[str | None],
    predicted: Sequence[str | None],
    *,
    labels: Sequence[str] | None = None,
) -> float:
    scores = classwise_f1(gold, predicted, labels=labels)
    return sum(scores.values()) / len(scores)


def confusion_matrix(
    gold: Sequence[str | None],
    predicted: Sequence[str | None],
    *,
    labels: Sequence[str] | None = None,
) -> dict[str, dict[str, int]]:
    if labels is None:
        active_labels = list(resolve_metric_labels(gold))
        for label in VALID_LABELS:
            if label in predicted and label not in active_labels:
                active_labels.append(label)
        active_labels.append("invalid")
        labels = tuple(active_labels)

    matrix = {
        gold_label: {predicted_label: 0 for predicted_label in labels}
        for gold_label in labels
    }
    for gold_label, predicted_label in zip(gold, predicted):
        normalized_gold = gold_label if gold_label in VALID_LABELS else "invalid"
        normalized_predicted = predicted_label if predicted_label in VALID_LABELS else "invalid"
        matrix[normalized_gold][normalized_predicted] += 1
    return matrix
