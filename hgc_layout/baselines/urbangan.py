"""GAN baseline conditioned on functional-zone masks and a frozen text encoder.

A conditional generator maps noise, a coarse zone-occupancy mask and a frozen
prompt embedding to normalized coordinates; a discriminator separates reference
layouts from generated ones under the same condition. Training is the standard
non-saturating GAN objective with an auxiliary reconstruction term, which keeps
the short training schedules used in the experiments stable.
"""
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from hgc_layout.baselines.base import BaselineMethod, to_entities
from hgc_layout.data.features import _hash_embedding
from hgc_layout.data.schema import Scene
from hgc_layout.planning.planner import Plan, TemplatePlanner

TEXT_DIM = 128
MASK_SIDE = 8
MAX_ENTITIES = 32
NOISE_DIM = 32


def zone_mask(plan: Plan, side: int = MASK_SIDE) -> np.ndarray:
    """Coarse occupancy mask of the functional zones implied by the plan."""
    mask = np.zeros((side, side), dtype=np.float32)
    groups = plan.groups()
    for i, g in enumerate(groups):
        cx = 0.5 + 0.26 * np.cos(2 * np.pi * i / max(len(groups), 1))
        cy = 0.5 + 0.26 * np.sin(2 * np.pi * i / max(len(groups), 1))
        n = len(plan.entities_of_group(g))
        gx, gy = int(np.clip(cx * side, 0, side - 1)), int(np.clip(cy * side, 0, side - 1))
        mask[gx, gy] += n
    total = mask.sum()
    return mask / total if total else mask


class Generator(nn.Module):
    def __init__(self, cond_dim: int, hidden: int = 256, max_entities: int = MAX_ENTITIES):
        super().__init__()
        self.max_entities = max_entities
        self.net = nn.Sequential(nn.Linear(cond_dim + NOISE_DIM, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, max_entities * 2), nn.Sigmoid())

    def forward(self, cond: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        out = self.net(torch.cat([cond, noise], dim=-1))
        return out.view(-1, self.max_entities, 2)


class Discriminator(nn.Module):
    def __init__(self, cond_dim: int, hidden: int = 256, max_entities: int = MAX_ENTITIES):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(cond_dim + max_entities * 2, hidden), nn.LeakyReLU(0.2),
                                 nn.Linear(hidden, hidden), nn.LeakyReLU(0.2), nn.Linear(hidden, 1))

    def forward(self, cond: torch.Tensor, layout: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([cond, layout.flatten(1)], dim=-1))


class UrbanGAN(BaselineMethod):
    name = "UrbanGAN"

    def __init__(self, planner: TemplatePlanner, lr: float = 2e-4, recon_weight: float = 10.0):
        self.planner = planner
        cond_dim = TEXT_DIM + MASK_SIDE * MASK_SIDE
        self.generator = Generator(cond_dim)
        self.discriminator = Discriminator(cond_dim)
        self.lr = lr
        self.recon_weight = recon_weight

    def _condition(self, scene: Scene, plan: Plan) -> torch.Tensor:
        text = _hash_embedding(scene.description, TEXT_DIM, "clip-frozen")
        mask = zone_mask(plan).reshape(-1)
        return torch.as_tensor(np.concatenate([text, mask]), dtype=torch.float32).unsqueeze(0)

    def _reference(self, scene: Scene) -> Tuple[torch.Tensor, torch.Tensor]:
        scale = np.array([float(scene.boundary["length"]), float(scene.boundary["width"])])
        coords = np.zeros((MAX_ENTITIES, 2), dtype=np.float32)
        mask = np.zeros(MAX_ENTITIES, dtype=np.float32)
        for k, e in enumerate(scene.entities[:MAX_ENTITIES]):
            coords[k] = np.asarray(e["position"], dtype=float) / scale
            mask[k] = 1.0
        return torch.as_tensor(coords).unsqueeze(0), torch.as_tensor(mask).unsqueeze(0)

    def fit(self, scenes: Sequence[Scene], epochs: int = 30, seed: int = 0) -> None:
        torch.manual_seed(seed)
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(0.5, 0.999))
        bce = nn.BCEWithLogitsLoss()
        conditions = [(self._condition(s, self.planner.plan(s)), *self._reference(s)) for s in scenes]
        for _ in range(epochs):
            for cond, real, mask in conditions:
                noise = torch.randn(1, NOISE_DIM)
                fake = self.generator(cond, noise)
                opt_d.zero_grad()
                loss_d = (bce(self.discriminator(cond, real), torch.ones(1, 1)) +
                          bce(self.discriminator(cond, fake.detach()), torch.zeros(1, 1))) / 2
                loss_d.backward()
                opt_d.step()
                opt_g.zero_grad()
                adv = bce(self.discriminator(cond, fake), torch.ones(1, 1))
                recon = (((fake - real) ** 2).mean(-1) * mask).sum() / mask.sum().clamp(min=1)
                (adv + self.recon_weight * recon).backward()
                opt_g.step()

    @torch.no_grad()
    def generate(self, scene: Scene, seed: int = 0) -> List[Dict]:
        torch.manual_seed(seed)
        plan = self.planner.plan(scene)
        cond = self._condition(scene, plan)
        coords = self.generator(cond, torch.randn(1, NOISE_DIM))[0].numpy()
        n = len(plan.entities)
        if n > MAX_ENTITIES:
            extra = np.tile(coords, (int(np.ceil(n / MAX_ENTITIES)), 1))[:n]
            coords = extra
        coords = coords[:n]
        rng = np.random.default_rng(seed)
        return to_entities(plan, coords, rng.uniform(0, 2 * np.pi, n), scene.boundary)
