"""Shared plotting style, smoothing, annotations, and figure persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_DPI = 300
DEFAULT_FONT_SIZE = 10
DEFAULT_TITLE_SIZE = 12


def apply_report_style(
    *,
    dpi: int,
    title_size: int = DEFAULT_TITLE_SIZE,
    legend_size: int | None = None,
) -> None:
    settings: dict[str, int] = {
        "font.size": DEFAULT_FONT_SIZE,
        "axes.titlesize": title_size,
        "axes.labelsize": 10,
        "figure.dpi": dpi,
    }
    if legend_size is not None:
        settings["legend.fontsize"] = legend_size
    else:
        settings["legend.fontsize"] = DEFAULT_FONT_SIZE
    plt.rcParams.update(settings)


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="same")


def save_figure(
    figure: plt.Figure, output_dir: Path, name: str, *, dpi: int
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / name
    png_path = stem.with_suffix(".png")
    pdf_path = stem.with_suffix(".pdf")
    figure.savefig(png_path, bbox_inches="tight", dpi=dpi)
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {png_path}")
    print(f"Saved {pdf_path}")
    return png_path, pdf_path


def annotate_bars(
    axis: plt.Axes,
    bars: Iterable[Any],
    formatter: Callable[[float], str],
    *,
    offset: Callable[[float], float] | None = None,
) -> None:
    for bar in bars:
        value = float(bar.get_height())
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            (offset or (lambda item: item + 0.012))(value),
            formatter(value),
            ha="center",
            va="bottom",
            fontsize=8,
        )
