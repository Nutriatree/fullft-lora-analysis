"""Output parsers for PubMedQA yes/no/maybe predictions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from pubmedqa.labels import LABEL_PATTERN, normalize_label

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
