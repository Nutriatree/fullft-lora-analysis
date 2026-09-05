"""Provenance manifest for figures derived from immutable experiment studies."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from pubmedqa.reporting.layout import StudyLayout
from pubmedqa.runtime.io import current_time_iso, write_json


def write_report_manifest(layout: StudyLayout) -> Path:
    figure_files = sorted(
        path.relative_to(layout.report_root).as_posix()
        for path in layout.figures_dir.rglob("*")
        if path.is_file()
    ) if layout.figures_dir.exists() else []
    write_json(
        layout.manifest_path,
        {
            "study_id": layout.study_id,
            "input_study_dir": str(layout.study_dir),
            "report_root": str(layout.report_root),
            "generated_at": current_time_iso(),
            "figure_files": figure_files,
        },
    )
    return layout.manifest_path
