"""Plot metric degradation under increasing data sparsity.

Usage:
    python scripts/plot_fig4_robustness.py --run-dir outputs/demo --out outputs/demo/fig4_robustness.png
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from hgc_layout.tables import aggregate
from hgc_layout.utils.io import read_json
from hgc_layout.utils.style import PALETTE, apply_style

METRICS = ("SPR", "COL", "RCN", "SP")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default="outputs/main")
    ap.add_argument("--out", default="outputs/fig4_robustness.png")
    args = ap.parse_args()
    apply_style()

    results = read_json(Path(args.run_dir) / "results.json")
    records = aggregate(results["robustness"], extra_keys=("dropout",))
    methods = sorted({r["method"] for r in records})
    fig, axes = plt.subplots(1, len(METRICS), figsize=(3.2 * len(METRICS), 3.0))
    for ax, metric in zip(axes, METRICS):
        for method in methods:
            rows = sorted([r for r in records if r["method"] == method], key=lambda r: r["dropout"])
            x = [100 * float(r["dropout"]) for r in rows]
            y = [float(r[metric]) for r in rows]
            colour = PALETTE["METHOD"] if "HGC" in method else PALETTE["BASELINE"]
            ax.plot(x, y, marker="o", color=colour, linewidth=1.8, markersize=4, label=method)
        ax.set_title(metric, fontsize=10)
        ax.set_xlabel("dropout (%)")
    axes[0].set_ylabel("score")
    axes[-1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
