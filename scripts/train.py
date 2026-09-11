"""Train the three stages and save a checkpoint.

Usage:
    python scripts/train.py --config configs/demo.yaml
"""
import argparse
import logging

from hgc_layout.data.loaders import load_benchmark, split_scenes
from hgc_layout.factory import build_model
from hgc_layout.training import save_checkpoint, train_all
from hgc_layout.utils.config import load_config
from hgc_layout.utils.io import write_json
from hgc_layout.utils.seed import set_seed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/main.yaml")
    ap.add_argument("--checkpoint")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    root = (cfg["data"]["synthetic_dir"] if cfg["data"]["source"] == "synthetic"
            else cfg["data"]["benchmark_dir"])
    scenes = split_scenes(load_benchmark(root))
    logging.info("scenes: %s", {k: len(v) for k, v in scenes.items()})
    if cfg["data"]["source"] == "synthetic":
        logging.info("Synthetic demo data; numbers are illustrative only.")

    model = build_model(cfg)
    history = train_all(model, scenes["train"], cfg)
    path = save_checkpoint(model, args.checkpoint or cfg["output"]["checkpoint"])
    write_json(history.as_dict(), f'{cfg["output"]["dir"]}/training_history.json')
    logging.info("checkpoint written to %s", path)


if __name__ == "__main__":
    main()
