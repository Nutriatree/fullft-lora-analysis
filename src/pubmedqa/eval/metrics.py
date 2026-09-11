"""Parse generated answers and compute PubMedQA metrics."""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass

from pubmedqa.data.records import LABEL_PATTERN, VALID_LABELS, normalize_label


def resolve_metric_labels(gold: Sequence[str | None]) -> tuple[str, ...]:
    labels = tuple(
        label for label in VALID_LABELS if any(item == label for item in gold)
    )
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
        normalized_predicted = (
            predicted_label if predicted_label in VALID_LABELS else "invalid"
        )
        matrix[normalized_gold][normalized_predicted] += 1
    return matrix


ANSWER_LINE_PATTERN = re.compile(r"^\s*answer\s*:\s*(yes|no|maybe)\b", re.IGNORECASE)


JSON_LABEL_KEYS = ("answer", "label", "final_decision", "prediction")


@dataclass(frozen=True)
class ParseResult:
    """Structured parse output for evaluation and error analysis."""

    label: str | None
    raw_text: str
    normalized_text: str
    matched_text: str | None
    method: str | None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.label is not None


def parse_pubmedqa_answer(text: str, *, strict: bool = False) -> ParseResult:
    """Parse a generated PubMedQA answer into yes/no/maybe.

    The preferred model output is a single label: ``yes``, ``no``, or
    ``maybe``. The parser also handles JSON snippets and common verbose
    completions so dry runs can measure parser robustness before training.
    """

    raw_text = text
    normalized = normalize_text(text)

    exact_label_result = parse_exact_label(normalized, raw_text)
    if exact_label_result.ok:
        return exact_label_result

    json_result = parse_json_answer(normalized, raw_text)
    if json_result.ok:
        return json_result

    answer_line_result = parse_answer_line(normalized, raw_text)
    if answer_line_result.ok:
        return answer_line_result

    first_label_result = parse_first_label(normalized, raw_text, strict=strict)
    if first_label_result.ok:
        return first_label_result

    return ParseResult(
        label=None,
        raw_text=raw_text,
        normalized_text=normalized,
        matched_text=None,
        method=None,
        error="No PubMedQA label found",
    )


def parse_exact_label(normalized: str, raw_text: str) -> ParseResult:
    """Parse the preferred label-only output."""

    label = normalize_label(normalized)
    if label is None:
        return ParseResult(None, raw_text, normalized, None, None)
    return ParseResult(
        label=label,
        raw_text=raw_text,
        normalized_text=normalized,
        matched_text=normalized,
        method="exact_label",
    )


def normalize_text(text: str) -> str:
    """Normalize model output before parsing."""

    text = text.strip()
    text = text.replace("\u200b", "")
    text = re.sub(r"</?s>|<\|[^>]+?\|>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def parse_json_answer(normalized: str, raw_text: str) -> ParseResult:
    """Parse JSON object outputs such as ``{"answer": "yes"}``."""

    candidates = [normalized]
    object_match = re.search(r"\{.*?\}", normalized)
    if object_match:
        candidates.append(object_match.group(0))

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        for key in JSON_LABEL_KEYS:
            if key in payload:
                label = normalize_label(payload[key], allow_embedded=True)
                if label is not None:
                    return ParseResult(
                        label=label,
                        raw_text=raw_text,
                        normalized_text=normalized,
                        matched_text=str(payload[key]),
                        method="json",
                    )

    return ParseResult(None, raw_text, normalized, None, None)


def parse_answer_line(normalized: str, raw_text: str) -> ParseResult:
    """Parse explicit answer-line outputs."""

    for line in raw_text.splitlines():
        match = ANSWER_LINE_PATTERN.search(line)
        if match:
            return ParseResult(
                label=match.group(1).lower(),
                raw_text=raw_text,
                normalized_text=normalized,
                matched_text=match.group(0).strip(),
                method="answer_line",
            )

    match = re.search(r"answer\s*:\s*(yes|no|maybe)\b", normalized, re.IGNORECASE)
    if match:
        return ParseResult(
            label=match.group(1).lower(),
            raw_text=raw_text,
            normalized_text=normalized,
            matched_text=match.group(0),
            method="answer_inline",
        )

    return ParseResult(None, raw_text, normalized, None, None)


def parse_first_label(normalized: str, raw_text: str, *, strict: bool) -> ParseResult:
    """Parse label-only or verbose outputs.

    In strict mode, fallback parsing accepts only outputs whose first token is a
    valid label. In non-strict mode, the first label anywhere in the completion
    is accepted.
    """

    if strict:
        match = re.match(r"^(yes|no|maybe)\b", normalized, re.IGNORECASE)
        method = "first_token"
    else:
        match = LABEL_PATTERN.search(normalized)
        method = "first_label"

    if match:
        return ParseResult(
            label=match.group(1).lower(),
            raw_text=raw_text,
            normalized_text=normalized,
            matched_text=match.group(0),
            method=method,
        )
    return ParseResult(None, raw_text, normalized, None, None)


def require_pubmedqa_label(text: str, *, strict: bool = False) -> str:
    """Parse a label or raise ValueError with the normalized failing text."""

    result = parse_pubmedqa_answer(text, strict=strict)
    if result.label is None:
        raise ValueError(f"{result.error}: {result.normalized_text!r}")
    return result.label
