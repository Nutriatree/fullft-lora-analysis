#!/usr/bin/env python3
"""Draw RQ-specific train/validation loss learning curves for PubMedQA."""

from __future__ import annotations

import argparse
from pathlib import Path

from pubmedqa.eval.plot_learning_curves import generate_rq_learning_curves
from pubmedqa.eval.reports import StudyLayout
from pubmedqa.eval.reports import write_report_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=Path("outputs/pubmedqa_train/study_single_epoch1_used"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--smooth-window", type=int, default=25)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layout = StudyLayout.from_study_dir(args.study_dir)
    generate_rq_learning_curves(
        args.study_dir,
        args.output_dir or layout.main_figures_dir,
        smooth_window=args.smooth_window,
        dpi=args.dpi,
    )
    print(f"Updated {write_report_manifest(layout)}")


if __name__ == "__main__":
    main()
