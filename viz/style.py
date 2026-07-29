"""
Shared plotting style for probex evaluation figures.

Colors follow a validated categorical palette (fixed hue order, CVD-safe
adjacent pairs) so model identity stays consistent and distinguishable
across every figure in viz/.
"""

from __future__ import annotations

import matplotlib as mpl

MODEL_ORDER = ["IFS_ENS", "uCast", "FGN", "ArchesWeatherGen"]

MODEL_LABELS = {
    "IFS_ENS": "IFS-ENS",
    "uCast": "U-Cast",
    "FGN": "FGN",
    "ArchesWeatherGen": "ArchesWeatherGen",
}

MODEL_COLORS = {
    "IFS_ENS": "#2a78d6",
    "uCast": "#eb6834",
    "FGN": "#1baf7a",
    "ArchesWeatherGen": "#eda100",
}

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


def apply_style() -> None:
    """Apply shared rcParams. Call once before plotting."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": BASELINE,
            "axes.labelcolor": INK_PRIMARY,
            "text.color": INK_PRIMARY,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "axes.grid": True,
            "grid.color": GRIDLINE,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
            "font.size": 10,
        }
    )
