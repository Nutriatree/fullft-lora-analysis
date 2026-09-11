#!/usr/bin/env python3
"""Create the RQ1 final PQA-L performance figure."""

from __future__ import annotations

import argparse
from pathlib import Path

from pubmedqa.eval.plot_rq1 import generate_rq1_performance_figure
from pubmedqa.eval.reports import StudyLayout
from pubmedqa.eval.reports import write_report_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=Path("outputs/pubmedqa_train/study_single_epoch1_used"))
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path("outputs/pubmedqa_eval/study_single_epoch1/Qwen_Qwen3-1.7B/baseline"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layout = StudyLayout.from_study_dir(args.study_dir)
    generate_rq1_performance_figure(
        args.study_dir,
        args.baseline_dir,
        args.output_dir or layout.main_figures_dir,
        dpi=args.dpi,
    )
    print(f"Updated {write_report_manifest(layout)}")


if __name__ == "__main__":
    main()
