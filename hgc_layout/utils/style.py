"""Shared plotting palette and matplotlib defaults.

One hue per semantic role, used consistently across every figure:
METHOD for the proposed framework, BASELINE greys for compared methods,
GRAPH for graph structure and attention traces, CONSTRAINT for forces and
violations, and a fixed colour per entity family in layout renders.
"""
import matplotlib as mpl

PALETTE = {
    "METHOD": "#1F77E0",
    "BASELINE": "#9AA3AD",
    "GRAPH": "#8E5CE6",
    "CONSTRAINT": "#F28C28",
    "ACCENT": "#2CB37A",
    "NEUTRAL": "#C7CDD4",
}

ENTITY_COLORS = {
    "building": "#5B8FF9",
    "road": "#9AA3AD",
    "green": "#2CB37A",
    "water": "#17B3C7",
    "poi": "#F28C28",
    "other": "#C7CDD4",
}

BASELINE_SHADES = ["#C7CDD4", "#9AA3AD", "#6F7883", "#4A525B"]


def light(hex_color: str, alpha: float = 0.18) -> tuple:
    """RGBA of the same hue with low opacity, for area fills."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return (r, g, b, alpha)


def apply_style() -> None:
    """White background, no top/right spines, 300 dpi output."""
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })
