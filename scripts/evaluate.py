"""Run the experiments on the test split and write the per-scene records.

Usage:
    python scripts/evaluate.py --config configs/demo.yaml
    python scripts/evaluate.py --config configs/main.yaml --only comparison robustness
"""
import argparse
import logging
from pathlib import Path

from hgc_layout import experiments
from hgc_layout.baselines.registry import build_baselines
from hgc_layout.data.loaders import load_benchmark, split_scenes
from hgc_layout.factory import build_model, build_planner_from_config, core_variants, module_variants
from hgc_layout.stats.tests import paired_comparisons
from hgc_layout.tables import per_scene_series
from hgc_layout.training import load_checkpoint
from hgc_layout.utils.config import load_config
from hgc_layout.utils.io import write_json
from hgc_layout.utils.seed import set_seed

log = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/main.yaml")
    ap.add_argument("--checkpoint")
    ap.add_argument("--out-dir")
    ap.add_argument("--only", nargs="*", help="subset of the evaluation.run list")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    out_dir = Path(args.out_dir or cfg["output"]["dir"])
    root = (cfg["data"]["synthetic_dir"] if cfg["data"]["source"] == "synthetic"
            else cfg["data"]["benchmark_dir"])
    scenes = split_scenes(load_benchmark(root))
    test, train = scenes["test"], scenes["train"]
    for name, split in (("test", test), ("train", train)):
        if not split:
            raise SystemExit(f"The {name} split is empty under {root}; regenerate the benchmark "
                             "with more scenes or check the split field of each scene JSON.")
    if cfg["data"]["source"] == "synthetic":
        log.info("Synthetic demo data; numbers are illustrative only.")

    planner = build_planner_from_config(cfg)
    model = build_model(cfg, planner=planner)
    checkpoint = args.checkpoint or cfg["output"]["checkpoint"]
    if Path(checkpoint).exists():
        load_checkpoint(model, checkpoint)
        log.info("loaded checkpoint %s", checkpoint)
    else:
        log.warning("checkpoint %s not found; evaluating untrained modules", checkpoint)

    baselines = build_baselines(planner, cfg["baselines"]["names"])
    for name, method in baselines.items():
        method.fit(train, epochs=cfg["baselines"]["epochs"], seed=cfg["seed"])
        log.info("fitted baseline %s", name)

    seeds = cfg["seeds"]
    sp_model = cfg["evaluation"].get("sp_model")
    requested = args.only or cfg["evaluation"]["run"]
    results = {}

    if "comparison" in requested:
        rows = experiments.comparison(model, baselines, test, seeds, sp_model)
        results["comparison"] = rows
        series = per_scene_series(rows, cfg["evaluation"]["primary_metrics"])
        results["significance"] = paired_comparisons(
            series, experiments.METHOD_NAME, cfg["baselines"]["names"],
            cfg["evaluation"]["primary_metrics"], cfg["evaluation"]["alpha"])
    if "core_ablation" in requested:
        results["core_ablation"] = experiments.core_ablation(
            model, core_variants(cfg, model), test, seeds, sp_model)
        results["module_ablation"] = experiments.core_ablation(
            model, module_variants(cfg, model), test, seeds, sp_model)
    if "optimizer_ablation" in requested:
        results["optimizer_ablation"] = experiments.optimizer_ablation(model, test, seeds, sp_model)
    if "data_source" in requested:
        results["data_source"] = experiments.data_source_ablation(model, test, seeds, sp_model)
    if "robustness" in requested:
        results["robustness"] = experiments.robustness(
            model, baselines["FW-FD"], test, seeds,
            cfg["evaluation"]["dropout_levels"], sp_model)
    if "editing" in requested:
        results["editing"] = experiments.editing(
            model, test, seeds, cfg["evaluation"]["editing_scenes"])
    if "reliability" in requested:
        results["reliability"] = experiments.sp_reliability(
            model, baselines, test[: cfg["evaluation"]["sp_reliability_scenes"]],
            cfg["evaluation"]["sp_reliability_runs"], cfg["seed"], sp_model)

    write_json(results, out_dir / "results.json")
    log.info("wrote %s (%d blocks)", out_dir / "results.json", len(results))


if __name__ == "__main__":
    main()
