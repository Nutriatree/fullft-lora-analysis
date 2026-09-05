"""Canonical separation between immutable studies and derived report assets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DEFAULT_REPORTS_ROOT = Path("reports/pubmedqa")


@dataclass(frozen=True)
class StudyLayout:
    study_dir: Path
    report_root: Path

    @classmethod
    def from_study_dir(
        cls,
        study_dir: Path,
        *,
        reports_root: Path = DEFAULT_REPORTS_ROOT,
    ) -> "StudyLayout":
        study_path = Path(study_dir)
        return cls(study_dir=study_path, report_root=reports_root / study_path.name)

    @property
    def study_id(self) -> str:
        return self.study_dir.name

    @property
    def figures_dir(self) -> Path:
        return self.report_root / "figures"

    @property
    def main_figures_dir(self) -> Path:
        return self.figures_dir / "main"

    @property
    def appendix_figures_dir(self) -> Path:
        return self.figures_dir / "appendix"

    @property
    def manifest_path(self) -> Path:
        return self.report_root / "report_manifest.json"
