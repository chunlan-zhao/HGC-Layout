"""Adaptive constraint weights of Eq. (4).

One small multi-layer perceptron per constraint type maps the concatenation of
the two node embeddings and the pair relation vector to a positive scalar. The
output is parameterised so that an untrained network emits values close to one,
which reproduces the fixed-weight formulation exactly.
"""
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hgc_layout.optimization.constraints import CONSTRAINT_TYPES

UNIT_OFFSET = 0.5413248546129181  # softplus(UNIT_OFFSET) == 1


class AdaptiveWeights(nn.Module):
    """Constraint-specific weight predictors, optionally without graph embeddings."""

    def __init__(self, embed_dim: int = 256, hidden_dim: int = 64, rel_dim: int = 4,
                 use_graph_embeddings: bool = True):
        super().__init__()
        self.use_graph_embeddings = use_graph_embeddings
        in_dim = (2 * embed_dim + rel_dim) if use_graph_embeddings else rel_dim
        self.mlps = nn.ModuleDict({
            c: nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
            for c in CONSTRAINT_TYPES
        })
        for mlp in self.mlps.values():
            nn.init.zeros_(mlp[-1].weight)
            nn.init.zeros_(mlp[-1].bias)

    def forward(self, constraint: str, h_i: Optional[torch.Tensor], h_j: Optional[torch.Tensor],
                r_ij: torch.Tensor) -> torch.Tensor:
        """Weight w_ij^(c) for a batch of pairs, strictly positive and near one at init."""
        if self.use_graph_embeddings:
            features = torch.cat([h_i, h_j, r_ij], dim=-1)
        else:
            features = r_ij
        raw = self.mlps[constraint](features).squeeze(-1)
        return F.softplus(raw + UNIT_OFFSET)

    def all_weights(self, h_i: torch.Tensor, h_j: torch.Tensor,
                    r_ij: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {c: self(c, h_i, h_j, r_ij) for c in CONSTRAINT_TYPES}


class FixedWeights(nn.Module):
    """Ablation stand-in that always returns one, as in the fixed-weight baseline."""

    def forward(self, constraint: str, h_i, h_j, r_ij: torch.Tensor) -> torch.Tensor:
        return torch.ones(r_ij.shape[:-1], dtype=r_ij.dtype, device=r_ij.device)

    def all_weights(self, h_i, h_j, r_ij):
        return {c: self(c, h_i, h_j, r_ij) for c in CONSTRAINT_TYPES}
