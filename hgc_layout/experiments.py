"""Experiment drivers: comparison, ablations, robustness, editing, reliability.

Every routine returns per-scene records so that the tables and the paired
significance tests operate on the same numbers.
"""
import logging
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from hgc_layout.baselines.registry import build_baselines
from hgc_layout.data.schema import Scene
from hgc_layout.data.sparsity import degrade_scene
from hgc_layout.editing.commands import apply_edit, parse_command
from hgc_layout.metrics.evaluate import METRIC_NAMES, evaluate_layout
from hgc_layout.metrics.reliability import reliability_report
from hgc_layout.metrics.semantic import semantic_plausibility
from hgc_layout.optimization.optimizer import AdaptiveOptimizer
from hgc_layout.optimization.weights import AdaptiveWeights, FixedWeights
from hgc_layout.pipeline import HGCLayout

log = logging.getLogger(__name__)

METHOD_NAME = "HGC-Layout"

EDIT_TEMPLATES = ("add a bus stop", "delete the playground", "move the supermarket to north")


def _record(method: str, scene: Scene, seed: int, scores: Dict[str, float],
            **extra) -> Dict[str, object]:
    row = {"method": method, "scene_id": scene.scene_id, "difficulty": scene.difficulty,
           "seed": seed}
    row.update(scores)
    row.update(extra)
    return row


def run_method(method, scenes: Sequence[Scene], seeds: Sequence[int], label: str,
               sp_model: Optional[str] = None) -> List[Dict[str, object]]:
    """Generate and evaluate one method over every scene and seed."""
    rows: List[Dict[str, object]] = []
    for seed in seeds:
        for scene in scenes:
            start = time.perf_counter()
            entities = (method.generate(scene, seed=seed).entities
                        if isinstance(method, HGCLayout) else method.generate(scene, seed=seed))
            elapsed = time.perf_counter() - start
            rows.append(_record(label, scene, seed,
                                evaluate_layout(entities, scene, elapsed, sp_model=sp_model)))
    return rows


def comparison(model: HGCLayout, baselines: Dict[str, object], scenes: Sequence[Scene],
               seeds: Sequence[int], sp_model: Optional[str] = None) -> List[Dict[str, object]]:
    """Main comparison of the framework against every baseline."""
    rows = run_method(model, scenes, seeds, METHOD_NAME, sp_model)
    for name, method in baselines.items():
        rows += run_method(method, scenes, seeds, name, sp_model)
    return rows


def core_ablation(model: HGCLayout, variants: Dict[str, HGCLayout], scenes: Sequence[Scene],
                  seeds: Sequence[int], sp_model: Optional[str] = None) -> List[Dict[str, object]]:
    """Full framework against the ablated variants supplied by the caller."""
    rows = run_method(model, scenes, seeds, "Full HGC-Layout", sp_model)
    for label, variant in variants.items():
        rows += run_method(variant, scenes, seeds, label, sp_model)
    return rows


def optimizer_ablation(model: HGCLayout, scenes: Sequence[Scene], seeds: Sequence[int],
                       sp_model: Optional[str] = None) -> List[Dict[str, object]]:
    """Adaptive optimizer against three stripped-down solvers."""
    variants: Dict[str, HGCLayout] = {}
    fixed = _clone(model)
    fixed.weights = FixedWeights()
    fixed.optimizer = AdaptiveOptimizer(fixed.weights, model.optimizer.lambdas,
                                        step_size=model.optimizer.step_size,
                                        max_iter=model.optimizer.max_iter)
    variants["w/o adaptive force weights"] = fixed

    no_deadlock = _clone(model)
    no_deadlock.optimizer = AdaptiveOptimizer(model.weights, model.optimizer.lambdas,
                                              step_size=model.optimizer.step_size,
                                              max_iter=model.optimizer.max_iter,
                                              use_deadlock_evasion=False)
    variants["w/o deadlock evasion"] = no_deadlock

    distance_only = _clone(model)
    distance_only.weights = AdaptiveWeights(model.encoder.out_dim, use_graph_embeddings=False)
    distance_only.optimizer = AdaptiveOptimizer(distance_only.weights, model.optimizer.lambdas,
                                                step_size=model.optimizer.step_size,
                                                max_iter=model.optimizer.max_iter)
    variants["distance-only MLP weights"] = distance_only

    rows = run_method(model, scenes, seeds, "Full HGC-Layout", sp_model)
    for label, variant in variants.items():
        rows += run_method(variant, scenes, seeds, label, sp_model)
    return rows


def data_source_ablation(model: HGCLayout, scenes: Sequence[Scene], seeds: Sequence[int],
                         sp_model: Optional[str] = None) -> List[Dict[str, object]]:
    """Contribution of each input modality, with the encoder held fixed."""
    blocks = {"Full (RS + POI + Road)": None, "RS only": ["rs"], "POI only": ["poi"],
              "Road only": ["graph"]}
    rows: List[Dict[str, object]] = []
    for label, subset in blocks.items():
        variant = _clone(model)
        variant.feature_blocks = subset
        rows += run_method(variant, scenes, seeds, label, sp_model)
    return rows


def robustness(model: HGCLayout, reference, scenes: Sequence[Scene], seeds: Sequence[int],
               levels: Sequence[float] = (0.0, 0.25, 0.50, 0.75),
               sp_model: Optional[str] = None) -> List[Dict[str, object]]:
    """Both methods under progressive dropout of POI records and road segments."""
    rows: List[Dict[str, object]] = []
    for level in levels:
        for seed in seeds:
            rng = np.random.default_rng(seed)
            degraded = [degrade_scene(s, level, rng) for s in scenes]
            for label, method in ((METHOD_NAME, model), (reference.name, reference)):
                for original, scene in zip(scenes, degraded):
                    start = time.perf_counter()
                    entities = (method.generate(scene, seed=seed).entities
                                if isinstance(method, HGCLayout) else method.generate(scene, seed=seed))
                    elapsed = time.perf_counter() - start
                    scores = evaluate_layout(entities, original, elapsed, sp_model=sp_model)
                    rows.append(_record(label, original, seed, scores, dropout=level))
    return rows


def editing(model: HGCLayout, scenes: Sequence[Scene], seeds: Sequence[int],
            n_scenes: int = 10, templates: Sequence[str] = EDIT_TEMPLATES
            ) -> List[Dict[str, object]]:
    """Completion time per editing operation on simple and complex scenes."""
    rows: List[Dict[str, object]] = []
    chosen = list(scenes)[:n_scenes]
    for seed in seeds:
        for scene in chosen:
            result = model.generate(scene, seed=seed)
            complexity = "simple" if len(result.entities) < 10 else "complex"
            for template in templates:
                command = parse_command(template)
                _, elapsed = apply_edit(model, scene, result.entities, command, seed=seed)
                rows.append({"scene_id": scene.scene_id, "seed": seed, "operation": command.operation,
                             "complexity": complexity, "time": elapsed})
    return rows


def sp_reliability(model: HGCLayout, baselines: Dict[str, object], scenes: Sequence[Scene],
                   n_runs: int = 3, seed: int = 0, sp_model: Optional[str] = None
                   ) -> Dict[str, object]:
    """Repeat the semantic-plausibility scoring and report its agreement."""
    layouts: List[Tuple[str, Scene, List[Dict]]] = []
    for scene in scenes:
        layouts.append((METHOD_NAME, scene, model.generate(scene, seed=seed).entities))
        for name, method in baselines.items():
            layouts.append((name, scene, method.generate(scene, seed=seed)))
    ratings = np.array([[semantic_plausibility(entities, scene, model=sp_model)["score"]
                         for _ in range(n_runs)] for _, scene, entities in layouts])
    report = reliability_report(ratings)
    report["n_subjects"] = int(ratings.shape[0])
    report["n_runs"] = int(n_runs)
    report["backend"] = semantic_plausibility(layouts[0][2], layouts[0][1],
                                              model=sp_model)["backend"]
    return report


def checkpoint_sweep(model: HGCLayout, scenes: Sequence[Scene], histories: Dict[str, List[float]],
                     checkpoints: Sequence[int], seed: int = 0,
                     sp_model: Optional[str] = None) -> List[Dict[str, object]]:
    """Downstream quality at successive pre-training checkpoints.

    The caller supplies a checkpoint list; the model must already be restored
    to each checkpoint by the training script before this routine is called for
    that checkpoint, which keeps this function free of file handling.
    """
    rows = []
    for scene in scenes:
        scores = evaluate_layout(model.generate(scene, seed=seed).entities, scene, 0.0,
                                 sp_model=sp_model)
        rows.append(_record(METHOD_NAME, scene, seed, scores, epochs=checkpoints[-1]))
    return rows


def _clone(model: HGCLayout) -> HGCLayout:
    """Shallow copy sharing the trained modules but allowing switch overrides."""
    clone = HGCLayout(model.planner, encoder=model.encoder, decoder=model.decoder,
                      weights=model.weights, optimizer=model.optimizer, delta=model.delta,
                      use_graph_conditioning=model.use_graph_conditioning,
                      use_adaptive_weights=model.use_adaptive_weights,
                      use_optimizer=model.use_optimizer, device=model.device,
                      feature_blocks=model.feature_blocks, use_poi_nodes=model.use_poi_nodes,
                      use_road_edges=model.use_road_edges,
                      use_rule_retrieval=model.use_rule_retrieval,
                      use_global_context=model.use_global_context,
                      use_hierarchical_groups=model.use_hierarchical_groups)
    return clone
