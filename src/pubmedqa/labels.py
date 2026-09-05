"""Compatibility facade for :mod:`pubmedqa.domain.labels`."""

from pubmedqa.domain.labels import LABEL_PATTERN, VALID_LABELS, normalize_label, require_label

__all__ = ["LABEL_PATTERN", "VALID_LABELS", "normalize_label", "require_label"]
