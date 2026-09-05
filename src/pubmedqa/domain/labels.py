"""PubMedQA answer-label policy."""

from __future__ import annotations

import re
from typing import Any

VALID_LABELS = ("yes", "no", "maybe")
LABEL_PATTERN = re.compile(r"\b(yes|no|maybe)\b", re.IGNORECASE)


def normalize_label(value: Any, *, allow_embedded: bool = False) -> str | None:
    """Normalize a value to a PubMedQA label, or return ``None``."""

    if value is None:
        return None
    text = str(value).strip().lower()
    if text in VALID_LABELS:
        return text
    if allow_embedded:
        match = LABEL_PATTERN.search(text)
        if match:
            return match.group(1).lower()
    return None


def require_label(value: Any, *, allow_embedded: bool = False) -> str:
    """Normalize a label or raise ``ValueError``."""

    label = normalize_label(value, allow_embedded=allow_embedded)
    if label is None:
        raise ValueError(f"Invalid PubMedQA label: {value!r}")
    return label
