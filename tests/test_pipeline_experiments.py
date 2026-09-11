import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from hgc_layout import experiments
from hgc_layout.baselines.registry import BASELINE_NAMES, build_baselines
from hgc_layout.data.loaders import save_scene
from hgc_layout.editing.commands import apply_edit, parse_command
from hgc_layout.factory import build_model, core_variants, module_variants
from hgc_layout.metrics.evaluate import METRIC_NAMES
from hgc_layout.tables import aggregate, markdown_table, per_scene_series, significance_table
from hgc_layout.training import (load_checkpoint, pretrain, save_checkpoint, train_adaptive_weights,
                                 train_generator)
from hgc_layout.graph.contrastive import pretrain_reconstruction
from hgc_layout.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def test_pipeline_generates_and_optimises(model, scene):
    result = model.generate(scene, seed=0)
    assert len(result.entities) > 0
    assert result.iterations > 0
    assert result.graph is not None and result.plan is not None
    assert len(result.traces) == len(result.entities)


def test_ablation_switches_all_run(demo_config, planner, scene):
    for overrides in ({"use_graph_conditioning": False}, {"use_adaptive_weights": False},
                      {"use_optimizer": False}, {"feature_blocks": ["poi"]},
                      {"use_poi_nodes": False}, {"use_road_edges": False},
                      {"use_rule_retrieval": False}, {"use_global_context": False},
                      {"use_hierarchical_groups": False}):
        variant = build_model(demo_config, planner=planner, **overrides)
        assert len(variant.generate(scene, seed=0).entities) > 0


def test_variant_factories(demo_config, model):
    assert set(core_variants(demo_config, model)) == {"w/o graph-conditioned gen.",
                                                      "w/o adaptive force weights"}
    assert len(module_variants(demo_config, model)) == 5


def test_training_stages_run(demo_config, planner, scenes, tmp_path):
    model = build_model(demo_config, planner=planner)
    contrastive = pretrain(model, scenes[:3], epochs=2, batch_size=4)
    generator = train_generator(model, scenes[:3], epochs=2)
    alignment = train_adaptive_weights(model, scenes[:2], epochs=1, unroll=4, warmup=2)
    assert len(contrastive) == 2 and len(generator) == 2 and len(alignment) == 1
    assert all(np.isfinite(v) for v in contrastive + generator + alignment)
    path = save_checkpoint(model, tmp_path / "ckpt.pt")
    restored = build_model(demo_config, planner=planner)
    load_checkpoint(restored, path)
    a = next(iter(restored.encoder.state_dict().values()))
    b = next(iter(model.encoder.state_dict().values()))
    assert torch.allclose(a, b)


def test_reconstruction_pretraining_ablation(planner, scenes, model):
    graphs = [model.build_graph(s, model.plan_scene(s)) for s in scenes[:2]]
    history = pretrain_reconstruction(graphs, model.encoder, epochs=2, log_every=0)
    assert len(history["loss"]) == 2


def test_baselines_train_and_generate(planner, scenes, scene):
    baselines = build_baselines(planner)
    assert set(baselines) == set(BASELINE_NAMES)
    for name, method in baselines.items():
        method.fit(scenes[:3], epochs=2, seed=0)
        entities = method.generate(scene, seed=0)
        assert len(entities) > 0
        for e in entities:
            assert np.all(np.isfinite(np.asarray(e["position"])))
    edges = baselines["GraphRNN-Layout"].predicted_edges(scene)
    assert isinstance(edges, list)


def test_editing_commands(model, scene):
    result = model.generate(scene, seed=0)
    assert parse_command("move the supermarket to north").operation == "move"
    assert parse_command("delete the playground").operation == "delete"
    added, elapsed = apply_edit(model, scene, result.entities, parse_command("add a kiosk"), seed=0)
    assert len(added) == len(result.entities) + 1 and elapsed > 0
    removed, _ = apply_edit(model, scene, result.entities,
                            parse_command(f'delete {result.entities[0]["description"]}'), seed=0)
    assert len(removed) == len(result.entities) - 1


def test_experiment_blocks(model, planner, scenes):
    test_scenes, seeds = scenes[:2], [0]
    baselines = build_baselines(planner, ["LayoutGPT", "FW-FD"])
    for method in baselines.values():
        method.fit(scenes[:3], epochs=1, seed=0)
    rows = experiments.comparison(model, baselines, test_scenes, seeds)
    assert {r["method"] for r in rows} == {"HGC-Layout", "LayoutGPT", "FW-FD"}
    assert all(set(METRIC_NAMES) <= set(r) for r in rows)
    robust = experiments.robustness(model, baselines["FW-FD"], test_scenes, seeds, [0.0, 0.5])
    assert {r["dropout"] for r in robust} == {0.0, 0.5}
    edits = experiments.editing(model, test_scenes, seeds, n_scenes=1)
    assert {r["operation"] for r in edits} == {"add", "delete", "move"}
    report = experiments.sp_reliability(model, baselines, test_scenes, n_runs=3)
    assert report["n_runs"] == 3 and "ICC(A,k)" in report
    assert len(experiments.optimizer_ablation(model, test_scenes[:1], seeds)) == 4
    assert len(experiments.data_source_ablation(model, test_scenes[:1], seeds)) == 4
    assert len(experiments.core_ablation(model, {}, test_scenes[:1], seeds)) == 1
    assert experiments.checkpoint_sweep(model, test_scenes[:1], {}, [1])


def test_tables_render(model, planner, scenes):
    rows = experiments.comparison(model, build_baselines(planner, ["LayoutGPT"]), scenes[:2], [0])
    records = aggregate(rows, with_std=True)
    text = markdown_table(records, ["SPR", "COL", "SP"], title="t", with_std=True)
    assert text.startswith("**t**") and "±" in text
    series = per_scene_series(rows, ["SPR"])
    assert set(series) == {"HGC-Layout", "LayoutGPT"}
    assert len(series["HGC-Layout"]["SPR"]) == 2
    formatted = significance_table([{"metric": "SPR", "baseline": "LayoutGPT", "W": 1.0,
                                     "p_raw": 0.01, "p_holm": 0.02, "n": 20, "significant": True}])
    assert formatted[0]["significant"] == "yes"
    assert "no" in markdown_table(
        [{"metric": "SPR", "baseline": "b", "significant": "no"}], ["significant"],
        key_columns=("metric", "baseline"), bold_best=False)


def test_configs_parse_and_scripts_run(tmp_path, scenes):
    for name in ("main.yaml", "demo.yaml", "real.yaml"):
        cfg = load_config(ROOT / "configs" / name)
        assert cfg["evaluation"]["run"]
    data_dir = tmp_path / "scenes"
    for s in scenes[:4]:
        save_scene(s, data_dir / s.split / f"{s.scene_id}.json")
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "prepare_data.py"),
                          "--data", "real", "--benchmark-dir", str(data_dir)],
                         capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0 and "Benchmark validated" in out.stdout + out.stderr


def test_end_to_end_scripts(tmp_path):
    cfg = load_config(ROOT / "configs" / "demo.yaml")
    cfg["data"]["n_scenes"] = 8
    cfg["data"]["synthetic_dir"] = str(tmp_path / "scenes")
    cfg["data"]["rules"] = str(ROOT / "data" / "rules" / "layout_rules.yaml")
    cfg["graph"]["pretrain_epochs"] = 1
    cfg["generator"]["epochs"] = 1
    cfg["optimizer"]["epochs"] = 1
    cfg["optimizer"]["max_iter"] = 20
    cfg["baselines"]["epochs"] = 1
    cfg["baselines"]["names"] = ["LayoutGPT"]
    cfg["seeds"] = [0]
    cfg["evaluation"]["run"] = ["comparison", "editing"]
    cfg["evaluation"]["editing_scenes"] = 1
    cfg["output"]["dir"] = str(tmp_path / "out")
    cfg["output"]["checkpoint"] = str(tmp_path / "ckpt.pt")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(dict(cfg)), encoding="utf-8")

    def run(script, *args):
        result = subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args],
                                capture_output=True, text=True, cwd=ROOT)
        assert result.returncode == 0, result.stderr[-2000:]

    run("prepare_data.py", "--data", "synthetic", "--n-scenes", "8",
        "--out-dir", str(tmp_path / "scenes"))
    run("train.py", "--config", str(config_path))
    run("evaluate.py", "--config", str(config_path))
    run("make_tables.py", "--run-dir", str(tmp_path / "out"))
    run("plot_fig1_architecture.py", "--out", str(tmp_path / "out" / "fig1.png"))
    run("plot_fig2_attention.py", "--config", str(config_path),
        "--out", str(tmp_path / "out" / "fig2.png"))
    run("plot_fig3_training.py", "--run-dir", str(tmp_path / "out"),
        "--out", str(tmp_path / "out" / "fig3.png"))
    run("plot_fig5_qualitative.py", "--config", str(config_path),
        "--out", str(tmp_path / "out" / "fig5.png"))

    results = json.loads((tmp_path / "out" / "results.json").read_text())
    assert {"comparison", "significance", "editing"} <= set(results)
    assert (tmp_path / "out" / "all_tables.md").exists()
    for name in ("fig1.png", "fig2.png", "fig3.png", "fig5.png"):
        assert (tmp_path / "out" / name).stat().st_size > 0
