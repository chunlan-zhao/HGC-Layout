"""Visualize the attention trace recorded when placing one entity.

Line width is proportional to the attention weight between the focal entity and
its top-k attended neighbours.

Usage:
    python scripts/plot_fig2_attention.py --config configs/demo.yaml --out outputs/demo/fig2_attention.png
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from hgc_layout.data.loaders import load_benchmark, split_scenes
from hgc_layout.factory import build_model
from hgc_layout.training import load_checkpoint
from hgc_layout.utils.config import load_config
from hgc_layout.utils.style import ENTITY_COLORS, PALETTE, apply_style


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/main.yaml")
    ap.add_argument("--out", default="outputs/fig2_attention.png")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--difficulty", default="hard")
    args = ap.parse_args()
    apply_style()

    cfg = load_config(args.config)
    root = (cfg["data"]["synthetic_dir"] if cfg["data"]["source"] == "synthetic"
            else cfg["data"]["benchmark_dir"])
    scenes = split_scenes(load_benchmark(root))["test"]
    scene = next((s for s in scenes if s.difficulty == args.difficulty), scenes[0])
    model = build_model(cfg)
    if Path(cfg["output"]["checkpoint"]).exists():
        load_checkpoint(model, cfg["output"]["checkpoint"])
    result = model.generate(scene, seed=cfg["seed"])

    positions = {e["id"]: np.asarray(e["position"], dtype=float) for e in result.entities}
    types = {e["id"]: e["type"] for e in result.entities}
    labels = {e["id"]: e["description"] for e in result.entities}
    focal_trace = max(result.traces, key=lambda t: len(t.neighbors))
    focal = focal_trace.entity_id
    if focal not in positions:
        focal = result.entities[0]["id"]

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    length = float(scene.boundary["length"])
    for e in result.entities:
        p = np.asarray(e["position"], dtype=float)
        ax.scatter(*p, s=34, color=ENTITY_COLORS.get(e["type"], ENTITY_COLORS["other"]),
                   zorder=3, edgecolor="white", linewidth=0.6)
    pairs = [(n, w) for n, w in focal_trace.top_k(args.top_k * 3) if n in positions and n != focal]
    pairs = pairs[: args.top_k]
    if pairs:
        wmax = max(w for _, w in pairs)
        offsets = [(6, 6), (6, -10), (-8, 8), (-8, -12), (10, 0), (-12, 0), (0, 10), (0, -14)]
        for rank, (neighbor, weight) in enumerate(pairs):
            a, b = positions[focal], positions[neighbor]
            ax.plot([a[0], b[0]], [a[1], b[1]], color=PALETTE["GRAPH"], zorder=2,
                    linewidth=0.6 + 3.4 * weight / (wmax + 1e-9), alpha=0.7)
            if rank < 5:   # labelling every neighbour turns clustered scenes into a smear
                ax.annotate(f"{labels[neighbor]} ({weight:.2f})", b, textcoords="offset points",
                            xytext=offsets[rank % len(offsets)], fontsize=7,
                            ha="left" if offsets[rank % len(offsets)][0] >= 0 else "right")
    ax.scatter(*positions[focal], s=120, color=PALETTE["METHOD"], zorder=4, marker="*",
               edgecolor="white", linewidth=0.8)
    ax.annotate(f"focal: {labels[focal]}", positions[focal], textcoords="offset points",
                xytext=(6, -12), fontsize=8, color=PALETTE["METHOD"])
    ax.set_xlim(0, length)
    ax.set_ylim(0, length)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Attention trace, scene {scene.scene_id} ({scene.difficulty})", fontsize=10)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
