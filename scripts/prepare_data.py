"""Generate the synthetic benchmark or validate a real benchmark directory.

Usage:
    python scripts/prepare_data.py --data synthetic --n-scenes 100
    python scripts/prepare_data.py --data real --benchmark-dir data/benchmark
"""
import argparse
import logging
from pathlib import Path

from hgc_layout.data.loaders import load_benchmark, save_scene
from hgc_layout.data.synthetic import generate_benchmark


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", choices=["synthetic", "real"], default="synthetic")
    ap.add_argument("--n-scenes", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="data/synthetic")
    ap.add_argument("--benchmark-dir", default="data/benchmark")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.data == "real":
        scenes = load_benchmark(args.benchmark_dir)
        counts = {}
        for s in scenes:
            counts[s.split] = counts.get(s.split, 0) + 1
        logging.info("Benchmark validated: %d scenes %s", len(scenes), counts)
        return

    scenes = generate_benchmark(args.n_scenes, seed=args.seed)
    out = Path(args.out_dir)
    for scene in scenes:
        save_scene(scene, out / scene.split / f"{scene.scene_id}.json")
    logging.info("Synthetic demo data; numbers are illustrative only.")
    logging.info("Wrote %d scenes to %s (train/val/test by tile)", len(scenes), out)


if __name__ == "__main__":
    main()
