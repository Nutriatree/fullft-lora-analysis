"""Filesystem serialization and artifact naming helpers."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol


class ArtifactWriter(Protocol):
    """Persistence port used by experiment and reporting services."""

    def write_json(self, path: Path, data: Any) -> None: ...

    def write_jsonl(self, path: Path, rows: Iterable[dict[str, Any]]) -> None: ...


class FileArtifactWriter:
    """UTF-8 filesystem implementation of :class:`ArtifactWriter`."""

    def write_json(self, path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False, default=str)
            file.write("\n")

    def write_jsonl(self, path: Path, rows: Iterable[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


DEFAULT_ARTIFACT_WRITER: ArtifactWriter = FileArtifactWriter()


def write_json(path: Path, data: Any) -> None:
    DEFAULT_ARTIFACT_WRITER.write_json(path, data)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    DEFAULT_ARTIFACT_WRITER.write_jsonl(path, rows)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")


def current_time_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def count_directory_size_bytes(path: Path) -> int:
    return sum(
        file_path.stat().st_size
        for file_path in path.rglob("*")
        if file_path.is_file()
    )
