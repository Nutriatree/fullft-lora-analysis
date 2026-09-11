"""Read study artifacts and assemble report paths and manifests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from pubmedqa.data.records import current_time_iso, write_json

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


class StudyArtifactReader:
    def __init__(self, study_dir: Path) -> None:
        self.study_dir = Path(study_dir)

    def resolve(self, relative_path: Path | str) -> Path:
        path = self.study_dir / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Study artifact not found: {path}")
        return path

    def read_json(self, relative_path: Path | str) -> dict[str, Any]:
        path = self.resolve(relative_path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Malformed JSON at {path}: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"Expected a JSON object in study artifact: {path}")
        return payload

    def read_jsonl(self, relative_path: Path | str) -> list[dict[str, Any]]:
        path = self.resolve(relative_path)
        records: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Malformed JSONL at {path}:{line_number}: {exc.msg}"
                    ) from exc
                if not isinstance(record, dict):
                    raise ValueError(f"Expected a JSON object at {path}:{line_number}")
                records.append(record)
        return records

    def require_run_directories(
        self,
        model_directory: str,
        run_directories: Sequence[str],
    ) -> dict[str, Path]:
        model_dir = self.study_dir / model_directory
        missing = [name for name in run_directories if not (model_dir / name).is_dir()]
        if missing:
            raise FileNotFoundError(
                f"Missing expected run directories under {model_dir}: {', '.join(missing)}"
            )
        return {name: model_dir / name for name in run_directories}


@dataclass(frozen=True)
class RunArtifacts:
    """Typed access to the stable artifact schema of one completed training run."""

    run_dir: Path

    @property
    def reader(self) -> StudyArtifactReader:
        return StudyArtifactReader(self.run_dir)

    def summary(self) -> dict[str, Any]:
        return self.reader.read_json("summary.json")

    def evaluation_summary(self, split: str) -> dict[str, Any]:
        return self.reader.read_json(Path("evaluations") / f"{split}_summary.json")

    def optimization_timeline(self) -> list[dict[str, Any]]:
        return self.reader.read_jsonl("analysis/optimization_timeline.jsonl")

    def performance_timeline(self) -> list[dict[str, Any]]:
        return self.reader.read_json("analysis/performance_dynamics.json")[
            "checkpoint_timeline"
        ]

    def analysis(self, name: str) -> dict[str, Any]:
        return self.reader.read_json(Path("analysis") / f"{name}.json")


def write_report_manifest(layout: StudyLayout) -> Path:
    figure_files = (
        sorted(
            path.relative_to(layout.report_root).as_posix()
            for path in layout.figures_dir.rglob("*")
            if path.is_file()
        )
        if layout.figures_dir.exists()
        else []
    )
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
