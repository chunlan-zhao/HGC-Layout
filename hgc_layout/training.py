"""Training procedures: contrastive pre-training, generator, adaptive weights.

The three stages are trained in sequence. Contrastive pre-training is
unsupervised. The generator is supervised with the teacher-forced negative
log-likelihood of the reference layout on the training split. The adaptive
weights are trained with the alignment loss of Eq. (5) through a truncated
unroll of the optimizer: the first `warmup` steps run as a forward warm-up,
the state is detached, and gradients flow only through the remaining steps.
"""
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from hgc_layout.data.schema import Scene
from hgc_layout.generation.generator import LayoutGenerator
from hgc_layout.graph.construction import build_graph
from hgc_layout.graph.contrastive import pretrain_encoder
from hgc_layout.optimization.constraints import CONSTRAINT_TYPES
from hgc_layout.optimization.optimizer import OptimizerState
from hgc_layout.optimization.violations import alignment_loss, violation_measures
from hgc_layout.pipeline import HGCLayout, scene_tensors
from hgc_layout.utils.io import ensure_dir

log = logging.getLogger(__name__)


@dataclass
class TrainingHistory:
    """Loss curves of the three stages."""
    contrastive: List[float] = field(default_factory=list)
    generator: List[float] = field(default_factory=list)
    alignment: List[float] = field(default_factory=list)

    def as_dict(self) -> Dict[str, List[float]]:
        return {"contrastive": self.contrastive, "generator": self.generator,
                "alignment": self.alignment}


def pretrain(model: HGCLayout, scenes: Sequence[Scene], epochs: int = 200, batch_size: int = 10,
             temperature: float = 0.5, lr: float = 1e-3, weight_decay: float = 1e-5,
             feature_dropout: float = 0.20, edge_removal: float = 0.20,
             edge_addition: float = 0.05, seed: int = 0) -> List[float]:
    """Stage one: contrastive pre-training of the GAT encoder."""
    graphs = [model.build_graph(s, model.plan_scene(s)) for s in scenes]
    history = pretrain_encoder(graphs, model.encoder, epochs=epochs, batch_size=batch_size,
                               temperature=temperature, lr=lr, weight_decay=weight_decay,
                               feature_dropout=feature_dropout, edge_removal=edge_removal,
                               edge_addition=edge_addition, seed=seed, device=model.device)
    return history["loss"]


def train_generator(model: HGCLayout, scenes: Sequence[Scene], epochs: int = 30, lr: float = 1e-4,
                    clip_norm: float = 1.0, seed: int = 0) -> List[float]:
    """Stage two: teacher-forced likelihood of the reference layouts."""
    torch.manual_seed(seed)
    generator = LayoutGenerator(model.encoder, model.decoder, model.device)
    opt = torch.optim.Adam(model.decoder.parameters(), lr=lr)
    cached = []
    for scene in scenes:
        plan = model.plan_scene(scene)
        graph = model.build_graph(scene, plan)
        targets = _aligned_targets(plan, scene)
        if targets:
            cached.append((graph, plan, targets))
    history: List[float] = []
    model.decoder.train()
    for epoch in range(epochs):
        losses = []
        for graph, plan, targets in cached:
            opt.zero_grad()
            loss = generator.nll(graph, plan, targets)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), clip_norm)
            opt.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)) if losses else float("nan"))
        log.info("generator epoch %d/%d loss %.4f", epoch + 1, epochs, history[-1])
    model.decoder.eval()
    return history


def _aligned_targets(plan, scene: Scene) -> Dict[str, Tuple[np.ndarray, float]]:
    """Pair planned entities with reference entities of the same label, in order."""
    pool: Dict[str, List[Dict]] = {}
    for e in scene.entities:
        pool.setdefault(e["description"], []).append(e)
    targets: Dict[str, Tuple[np.ndarray, float]] = {}
    for entity in plan.entities:
        candidates = pool.get(entity["description"])
        if candidates:
            ref = candidates.pop(0)
            targets[entity["id"]] = (np.asarray(ref["position"], dtype=float),
                                     float(ref["orientation"] or 0.0))
    return targets


def train_adaptive_weights(model: HGCLayout, scenes: Sequence[Scene], epochs: int = 20,
                           lr: float = 1e-3, unroll: int = 40, warmup: int = 20,
                           beta: float = 0.01, clip_norm: float = 1.0, tol: float = 1e-3,
                           seed: int = 0) -> List[float]:
    """Stage three: alignment loss through a truncated unroll of the optimizer.

    Reference coordinates never enter the objective; only residual constraint
    violations of the converged layout and the unit prior on the weights do.
    """
    torch.manual_seed(seed)
    params = [p for p in model.weights.parameters()]
    if not params:
        return []
    opt = torch.optim.Adam(params, lr=lr)
    cached = []
    for scene in scenes:
        result = model.generate(scene, seed=seed)
        plan, graph = result.plan, result.graph
        cached.append((scene, plan, graph, result.entities))
    history: List[float] = []
    for epoch in range(epochs):
        losses = []
        for scene, plan, graph, entities in cached:
            tensors = scene_tensors(entities, scene, plan, graph, model.encoder, model.device)
            state = OptimizerState(tensors.pos.clone(), tensors.theta.clone(), tensors.size.clone())
            with torch.no_grad():
                for _ in range(warmup):
                    state, _, _ = model.optimizer.step(tensors, state)
            state = OptimizerState(state.pos.detach(), state.theta.detach(), state.size.detach())
            weights: List[torch.Tensor] = []
            residual = None
            for _ in range(max(unroll - warmup, 1)):
                state, residual, w = model.optimizer.step(tensors, state)
                weights = w
                if float(residual.detach()) < tol:
                    break
            measures = violation_measures(
                state.pos, state.size, tensors.extent, tensors.pairs, tensors.related_pairs,
                tensors.parent_pairs, tensors.road_points, tensors.admissible,
                tensors.frontage_mask, tensors.required, tensors.setback)
            loss = alignment_loss(measures, weights, model.optimizer.lambdas, beta)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, clip_norm)
            opt.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)) if losses else float("nan"))
        log.info("alignment epoch %d/%d loss %.4f", epoch + 1, epochs, history[-1])
    return history


def train_all(model: HGCLayout, train_scenes: Sequence[Scene], cfg) -> TrainingHistory:
    """Run the three stages with the durations given in the config."""
    history = TrainingHistory()
    history.contrastive = pretrain(
        model, train_scenes, epochs=cfg["graph"]["pretrain_epochs"],
        batch_size=cfg["graph"]["batch_size"], temperature=cfg["graph"]["temperature"],
        lr=cfg["graph"]["lr"], weight_decay=cfg["graph"]["weight_decay"],
        feature_dropout=cfg["graph"]["feature_dropout"], edge_removal=cfg["graph"]["edge_removal"],
        edge_addition=cfg["graph"]["edge_addition"], seed=cfg["seed"])
    history.generator = train_generator(
        model, train_scenes, epochs=cfg["generator"]["epochs"], lr=cfg["generator"]["lr"],
        seed=cfg["seed"])
    history.alignment = train_adaptive_weights(
        model, train_scenes, epochs=cfg["optimizer"]["epochs"], lr=cfg["optimizer"]["lr"],
        unroll=cfg["optimizer"]["unroll"], warmup=cfg["optimizer"]["warmup"],
        beta=cfg["optimizer"]["beta"], tol=cfg["optimizer"]["tol"], seed=cfg["seed"])
    return history


def save_checkpoint(model: HGCLayout, path) -> Path:
    """Persist encoder, decoder and adaptive-weight parameters."""
    path = Path(path)
    ensure_dir(path.parent)
    state = {"encoder": model.encoder.state_dict(), "decoder": model.decoder.state_dict()}
    if list(model.weights.parameters()):
        state["weights"] = model.weights.state_dict()
    torch.save(state, path)
    return path


def load_checkpoint(model: HGCLayout, path, strict: bool = True) -> HGCLayout:
    """Restore parameters saved by `save_checkpoint`."""
    state = torch.load(Path(path), map_location=model.device, weights_only=True)
    model.encoder.load_state_dict(state["encoder"], strict=strict)
    model.decoder.load_state_dict(state["decoder"], strict=strict)
    if "weights" in state and list(model.weights.parameters()):
        model.weights.load_state_dict(state["weights"], strict=strict)
    return model
