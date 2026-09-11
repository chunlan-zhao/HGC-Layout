"""LayoutGPT-style baseline: coordinates straight from the text prompt.

An entity- and prompt-conditioned multi-layer perceptron predicts normalized
coordinates directly, with no spatial graph, no constraint solver and no
awareness of the other entities beyond the sequence position.
"""
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn

from hgc_layout.baselines.base import (BaselineMethod, feature_dim, text_features, to_entities,
                                       training_pairs)
from hgc_layout.data.schema import Scene
from hgc_layout.planning.planner import TemplatePlanner


class LayoutGPT(BaselineMethod):
    name = "LayoutGPT"

    def __init__(self, planner: TemplatePlanner, hidden: int = 128, lr: float = 1e-3):
        self.planner = planner
        self.net = nn.Sequential(nn.Linear(feature_dim(), hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 2), nn.Sigmoid())
        self.lr = lr

    def fit(self, scenes: Sequence[Scene], epochs: int = 30, seed: int = 0) -> None:
        torch.manual_seed(seed)
        x, y = training_pairs(self.planner, scenes)
        if len(x) == 0:
            return
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        for _ in range(epochs):
            opt.zero_grad()
            loss = nn.functional.mse_loss(self.net(x), y)
            loss.backward()
            opt.step()

    @torch.no_grad()
    def generate(self, scene: Scene, seed: int = 0) -> List[Dict]:
        plan = self.planner.plan(scene)
        feats = torch.as_tensor(np.stack([
            text_features(e, scene.description, k, len(plan.entities))
            for k, e in enumerate(plan.entities)]), dtype=torch.float32)
        coords = self.net(feats).numpy()
        rng = np.random.default_rng(seed)
        coords = np.clip(coords + rng.normal(0, 0.03, coords.shape), 0, 1)
        thetas = rng.uniform(0, 2 * np.pi, len(plan.entities))
        return to_entities(plan, coords, thetas, scene.boundary)
