"""Render generated layouts of every method side by side for one scene.

Usage:
    python scripts/plot_fig5_qualitative.py --config configs/demo.yaml --out outputs/demo/fig5_qualitative.png
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from hgc_layout.baselines.registry import build_baselines
from hgc_layout.data.loaders import load_benchmark, split_scenes
from hgc_layout.factory import build_model, build_planner_from_config
from hgc_layout.training import load_checkpoint
from hgc_layout.utils.config import load_config
from hgc_layout.utils.style import ENTITY_COLORS, apply_style


def draw(ax, entities, scene, title):
    length = float(scene.boundary["length"])
    axes = scene.roads.get("axes", {})
    for x in axes.get("x", []):
        ax.axvline(x, color=ENTITY_COLORS["road"], linewidth=2.4, alpha=0.45, zorder=1)
    for y in axes.get("y", []):
        ax.axhline(y, color=ENTITY_COLORS["road"], linewidth=2.4, alpha=0.45, zorder=1)
    for e in entities:
        p = np.asarray(e["position"], dtype=float)
        w, h = float(e["size"][0]), float(e["size"][1])
        ax.add_patch(Rectangle(p - np.array([w, h]) / 2, w, h, zorder=2,
                               facecolor=ENTITY_COLORS.get(e["type"], ENTITY_COLORS["other"]),
                               edgecolor="white", linewidth=0.6, alpha=0.9))
    ax.set_xlim(0, length)
    ax.set_ylim(0, length)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=9)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/main.yaml")
    ap.add_argument("--out", default="outputs/fig5_qualitative.png")
    ap.add_argument("--difficulty", default="hard")
    args = ap.parse_args()
    apply_style()

    cfg = load_config(args.config)
    root = (cfg["data"]["synthetic_dir"] if cfg["data"]["source"] == "synthetic"
            else cfg["data"]["benchmark_dir"])
    splits = split_scenes(load_benchmark(root))
    test = splits["test"]
    scene = next((s for s in test if s.difficulty == args.difficulty), test[0])

    planner = build_planner_from_config(cfg)
    model = build_model(cfg, planner=planner)
    if Path(cfg["output"]["checkpoint"]).exists():
        load_checkpoint(model, cfg["output"]["checkpoint"])
    baselines = build_baselines(planner, cfg["baselines"]["names"])
    for method in baselines.values():
        method.fit(splits["train"], epochs=cfg["baselines"]["epochs"], seed=cfg["seed"])

    panels = [("Reference", scene.entities),
              ("HGC-Layout", model.generate(scene, seed=cfg["seed"]).entities)]
    panels += [(name, m.generate(scene, seed=cfg["seed"])) for name, m in baselines.items()]
    fig, axes = plt.subplots(1, len(panels), figsize=(2.6 * len(panels), 3.0))
    for ax, (title, entities) in zip(axes, panels):
        draw(ax, entities, scene, title)
    fig.suptitle(f"Scene {scene.scene_id} ({scene.difficulty}, {scene.category})", fontsize=10)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
