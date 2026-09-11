#!/usr/bin/env python3
"""Create Appendix figures for layer-wise and temporal adaptation dynamics."""

from __future__ import annotations

import argparse
from pathlib import Path

from pubmedqa.eval.plot_appendix import generate_appendix_figures
from pubmedqa.eval.reports import StudyLayout
from pubmedqa.eval.reports import write_report_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=Path("outputs/pubmedqa_train/study_single_epoch1_used"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layout = StudyLayout.from_study_dir(args.study_dir)
    generate_appendix_figures(
        args.study_dir,
        args.output_dir or layout.appendix_figures_dir,
        dpi=args.dpi,
    )
    print(f"Updated {write_report_manifest(layout)}")


if __name__ == "__main__":
    main()
