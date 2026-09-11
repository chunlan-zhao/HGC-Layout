"""FW-FD: fixed-weight force-directed baseline.

Entities are seeded group by group around zone centres and refined by the same
force-directed solver used by the full method, with every adaptive weight
pinned to one, restricted to the generic constraint subset that transfers to
the outdoor setting: collision, boundary and proximity.

This follows the optimizer stage of HOG-Layout (Jiang et al., CVPR 2026) but is
NOT a reimplementation of that system: its retrieval-augmented planner and
vision-language placement stage are not reproduced here.
"""
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from hgc_layout.baselines.base import BaselineMethod, to_entities
from hgc_layout.data.schema import Scene
from hgc_layout.optimization.constraints import LAMBDA_DEFAULT
from hgc_layout.optimization.optimizer import AdaptiveOptimizer
from hgc_layout.optimization.weights import FixedWeights
from hgc_layout.planning.planner import TemplatePlanner

FW_FD_LAMBDAS = {"collision": 1.0, "boundary": 1.0, "proximity": 0.5, "support": 0.0, "frontage": 0.0}


class FWFD(BaselineMethod):
    name = "FW-FD"

    def __init__(self, planner: TemplatePlanner, max_iter: int = 500, step_size: float = 0.05):
        self.planner = planner
        self.optimizer = AdaptiveOptimizer(FixedWeights(), lambdas=FW_FD_LAMBDAS,
                                           step_size=step_size, max_iter=max_iter)

    def generate(self, scene: Scene, seed: int = 0) -> List[Dict]:
        from hgc_layout.pipeline import apply_state, scene_tensors
        from hgc_layout.graph.construction import build_graph, feature_width
        from hgc_layout.graph.gat import GATEncoder

        rng = np.random.default_rng(seed)
        plan = self.planner.plan(scene)
        groups = plan.groups()
        centres = {g: np.array([0.5 + 0.26 * np.cos(2 * np.pi * i / max(len(groups), 1)),
                                0.5 + 0.26 * np.sin(2 * np.pi * i / max(len(groups), 1))])
                   for i, g in enumerate(groups)}
        coords = np.stack([np.clip(centres[e["group"]] + rng.normal(0, 0.07, 2), 0.02, 0.98)
                           for e in plan.entities])
        entities = to_entities(plan, coords, rng.uniform(0, 2 * np.pi, len(plan.entities)),
                               scene.boundary)
        graph = build_graph(scene, plan)
        encoder = GATEncoder(feature_width())
        tensors = scene_tensors(entities, scene, plan, graph, encoder, torch.device("cpu"))
        state, _ = self.optimizer.run(tensors)
        return apply_state(entities, state, scene.boundary)
