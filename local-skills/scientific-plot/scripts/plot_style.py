"""Small helpers for publication-quality scientific matplotlib figures."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}

COLUMN_WIDTH_IN = {
    "single": 3.35,
    "double": 6.85,
    "square": 3.35,
    "wide": 5.20,
}


def mm_to_in(mm: float) -> float:
    return mm / 25.4


def figure_size(width: str = "single", height_ratio: float = 0.62) -> tuple[float, float]:
    """Return a journal-friendly figure size in inches."""
    if width not in COLUMN_WIDTH_IN:
        valid = ", ".join(sorted(COLUMN_WIDTH_IN))
        raise ValueError(f"width must be one of: {valid}")
    w = COLUMN_WIDTH_IN[width]
    if width == "square":
        return (w, w)
    return (w, w * height_ratio)


def set_paper_style(font_size: int = 8, line_width: float = 1.1) -> None:
    """Apply conservative matplotlib rcParams for scientific figures."""
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "font.size": font_size,
            "axes.labelsize": font_size,
            "axes.titlesize": font_size,
            "xtick.labelsize": font_size - 1,
            "ytick.labelsize": font_size - 1,
            "legend.fontsize": font_size - 1,
            "axes.linewidth": 0.8,
            "lines.linewidth": line_width,
            "lines.markersize": 4,
            "axes.grid": True,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def savefig_all(
    fig,
    path_base: str | Path,
    formats: Iterable[str] = ("pdf", "svg", "png"),
    dpi: int = 300,
) -> list[Path]:
    """Save a matplotlib figure to several formats and return written paths."""
    path_base = Path(path_base)
    path_base.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for fmt in formats:
        out = path_base.with_suffix("." + fmt.lstrip("."))
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.03}
        if fmt.lower().lstrip(".") == "png":
            kwargs["dpi"] = dpi
        fig.savefig(out, **kwargs)
        written.append(out)
    return written
