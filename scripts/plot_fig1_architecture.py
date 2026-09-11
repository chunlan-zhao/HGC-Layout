"""Draw the architecture overview: data fusion, encoder, generator, optimizer.

Usage:
    python scripts/plot_fig1_architecture.py --out outputs/demo/fig1_architecture.png
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from hgc_layout.utils.style import PALETTE, apply_style, light


def box(ax, xy, w, h, text, color, fontsize=9, weight="normal"):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                facecolor=light(color, 0.20), edgecolor=color, linewidth=1.5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, weight=weight)


def arrow(ax, start, end, color, ls="-"):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13, color=color,
                                 linewidth=1.5, linestyle=ls))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="outputs/fig1_architecture.png")
    args = ap.parse_args()
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    for i, (label, y) in enumerate((("Remote sensing", 0.74), ("POI semantics", 0.50),
                                    ("Road network", 0.26))):
        box(ax, (0.02, y - 0.07), 0.16, 0.14, label, PALETTE["NEUTRAL"])
        arrow(ax, (0.18, y), (0.24, 0.50), PALETTE["NEUTRAL"])

    box(ax, (0.24, 0.38), 0.15, 0.24, "Multi-source\nheterogeneous\ngraph", PALETTE["GRAPH"])
    box(ax, (0.43, 0.38), 0.15, 0.24, "GAT encoder\ncontrastive\npre-training", PALETTE["GRAPH"],
        weight="bold")
    box(ax, (0.62, 0.38), 0.15, 0.24, "Graph-conditioned\nautoregressive\ngenerator",
        PALETTE["METHOD"], weight="bold")
    box(ax, (0.81, 0.38), 0.17, 0.24, "Adaptive\nforce-directed\noptimizer", PALETTE["CONSTRAINT"],
        weight="bold")

    for a, b in ((0.39, 0.43), (0.58, 0.62), (0.77, 0.81)):
        arrow(ax, (a, 0.50), (b, 0.50), PALETTE["METHOD"])

    ax.add_patch(FancyArrowPatch((0.895, 0.38), (0.695, 0.30), connectionstyle="arc3,rad=0.35",
                                 arrowstyle="-|>", mutation_scale=12,
                                 color=PALETTE["CONSTRAINT"], linestyle="--", linewidth=1.3))
    ax.text(0.79, 0.235, "feedback: updated partial graph", ha="center", fontsize=8,
            color=PALETTE["CONSTRAINT"])

    box(ax, (0.24, 0.06), 0.53, 0.10, "Planning: retrieved layout rules  →  entity set, "
        "functional groups, scene boundary", PALETTE["ACCENT"], fontsize=8.5)
    arrow(ax, (0.50, 0.16), (0.31, 0.38), PALETTE["ACCENT"], ls="--")
    ax.text(0.50, 0.86, "attention coefficients and cross-attention traces retained for inspection",
            ha="center", fontsize=8.5, color=PALETTE["GRAPH"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
