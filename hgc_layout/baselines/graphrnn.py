"""Graph recurrent baseline.

A GRU rolls over the entity sequence and emits an entity-level graph: at each
step it predicts edges to the entities generated so far and a node attribute
vector. A separate multi-layer perceptron maps the node attributes to spatial
coordinates. The model sees relational structure but no geospatial priors and
no constraint solver.
"""
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn

from hgc_layout.baselines.base import (BaselineMethod, feature_dim, text_features, to_entities,
                                       targets_from_scene)
from hgc_layout.data.schema import Scene
from hgc_layout.planning.planner import TemplatePlanner


class GraphRNNLayout(BaselineMethod):
    name = "GraphRNN-Layout"

    def __init__(self, planner: TemplatePlanner, hidden: int = 128, attr_dim: int = 32,
                 lr: float = 1e-3):
        self.planner = planner
        self.rnn = nn.GRU(feature_dim(), hidden, batch_first=True)
        self.edge_head = nn.Linear(hidden, 1)
        self.attr_head = nn.Sequential(nn.Linear(hidden, attr_dim), nn.ReLU())
        self.coord_head = nn.Sequential(nn.Linear(attr_dim, hidden), nn.ReLU(),
                                        nn.Linear(hidden, 2), nn.Sigmoid())
        self.lr = lr

    def _parameters(self):
        return (list(self.rnn.parameters()) + list(self.edge_head.parameters()) +
                list(self.attr_head.parameters()) + list(self.coord_head.parameters()))

    def _sequence(self, scene: Scene) -> torch.Tensor:
        plan = self.planner.plan(scene)
        feats = np.stack([text_features(e, scene.description, k, len(plan.entities))
                          for k, e in enumerate(plan.entities)])
        return torch.as_tensor(feats, dtype=torch.float32).unsqueeze(0)

    def fit(self, scenes: Sequence[Scene], epochs: int = 30, seed: int = 0) -> None:
        torch.manual_seed(seed)
        opt = torch.optim.Adam(self._parameters(), lr=self.lr)
        cached = []
        for scene in scenes:
            plan = self.planner.plan(scene)
            targets = targets_from_scene(scene)
            ids = list(targets)
            y = np.stack([targets[e["id"]] if e["id"] in targets else targets[ids[min(k, len(ids) - 1)]]
                          for k, e in enumerate(plan.entities)]) if ids else None
            if y is None:
                continue
            cached.append((self._sequence(scene), torch.as_tensor(y, dtype=torch.float32)))
        for _ in range(epochs):
            for x, y in cached:
                opt.zero_grad()
                h, _ = self.rnn(x)
                coords = self.coord_head(self.attr_head(h[0]))
                edge_logits = self.edge_head(h[0]).squeeze(-1)
                loss = (nn.functional.mse_loss(coords, y) +
                        0.01 * edge_logits.pow(2).mean())   # keeps edge scores bounded
                loss.backward()
                opt.step()

    @torch.no_grad()
    def generate(self, scene: Scene, seed: int = 0) -> List[Dict]:
        torch.manual_seed(seed)
        plan = self.planner.plan(scene)
        h, _ = self.rnn(self._sequence(scene))
        coords = self.coord_head(self.attr_head(h[0])).numpy()
        rng = np.random.default_rng(seed)
        coords = np.clip(coords + rng.normal(0, 0.02, coords.shape), 0, 1)
        return to_entities(plan, coords, rng.uniform(0, 2 * np.pi, len(plan.entities)),
                           scene.boundary)

    def predicted_edges(self, scene: Scene, threshold: float = 0.0) -> List[List[int]]:
        """Entity-level edges implied by the recurrent edge head."""
        with torch.no_grad():
            h, _ = self.rnn(self._sequence(scene))
            scores = self.edge_head(h[0]).squeeze(-1).numpy()
        return [[i, j] for i in range(len(scores)) for j in range(i)
                if scores[i] + scores[j] > threshold]
