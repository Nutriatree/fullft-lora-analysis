"""Input contracts for external PubMedQA dataset sources."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Protocol


class CanonicalSourcePort(Protocol):
    @property
    def labeled_source_url(self) -> str: ...

    def load_artificial(self) -> OrderedDict[str, dict[str, Any]]: ...

    def load_labeled(self) -> OrderedDict[str, dict[str, Any]]: ...

    def load_official_test_labels(self) -> dict[str, str] | None: ...
