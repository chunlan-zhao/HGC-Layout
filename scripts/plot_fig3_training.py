"""Plot the training dynamics of the three stages.

Usage:
    python scripts/plot_fig3_training.py --run-dir outputs/demo --out outputs/demo/fig3_training.png
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from hgc_layout.utils.io import read_json
from hgc_layout.utils.style import PALETTE, apply_style

STAGES = (("contrastive", "Contrastive pre-training", "GRAPH"),
          ("generator", "Generator likelihood", "METHOD"),
          ("alignment", "Alignment loss", "CONSTRAINT"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default="outputs/main")
    ap.add_argument("--out", default="outputs/fig3_training.png")
    args = ap.parse_args()
    apply_style()

    history = read_json(Path(args.run_dir) / "training_history.json")
    present = [(k, title, colour) for k, title, colour in STAGES if history.get(k)]
    fig, axes = plt.subplots(1, len(present), figsize=(3.6 * len(present), 3.0))
    axes = [axes] if len(present) == 1 else list(axes)
    for ax, (key, title, colour) in zip(axes, present):
        values = history[key]
        ax.plot(range(1, len(values) + 1), values, color=PALETTE[colour], linewidth=1.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
