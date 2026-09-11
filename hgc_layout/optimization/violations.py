"""Normalized constraint violations and the alignment loss of Eq. (5).

Every measure phi_c lies in [0, 1], is averaged over the relevant entities or
pairs, and returns zero when its set is empty. Collision and distance hinges
use a Softplus approximation so that all measures are differentiable. Reference
entity coordinates never enter these measures.
"""
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

from hgc_layout.optimization.constraints import CONSTRAINT_TYPES, LAMBDA_DEFAULT

SOFT_BETA = 50.0


def _softplus_hinge(x: torch.Tensor, beta: float = SOFT_BETA) -> torch.Tensor:
    """Differentiable approximation of max(0, x)."""
    return F.softplus(x * beta) / beta


def collision_violation(pos: torch.Tensor, size: torch.Tensor, pairs: torch.Tensor) -> torch.Tensor:
    """Overlapping area divided by the footprint of the smaller entity."""
    if pairs.numel() == 0:
        return pos.new_zeros(())
    i, j = pairs[:, 0], pairs[:, 1]
    reach = (size[i, :2] + size[j, :2]) / 2
    overlap = _softplus_hinge(reach - (pos[i] - pos[j]).abs())
    area = overlap[:, 0] * overlap[:, 1]
    footprint = torch.minimum(size[i, 0] * size[i, 1], size[j, 0] * size[j, 1])
    return torch.clamp(area / (footprint + 1e-9), 0.0, 1.0).mean()


def boundary_violation(pos: torch.Tensor, size: torch.Tensor, extent: torch.Tensor) -> torch.Tensor:
    """Area outside the scene boundary divided by the entity footprint."""
    half = size[:, :2] / 2
    outside = _softplus_hinge(half - pos) + _softplus_hinge(pos + half - extent)
    inside = torch.clamp(size[:, :2] - outside, min=0.0)
    area_out = size[:, 0] * size[:, 1] - inside[:, 0] * inside[:, 1]
    return torch.clamp(area_out / (size[:, 0] * size[:, 1] + 1e-9), 0.0, 1.0).mean()


def proximity_violation(pos: torch.Tensor, pairs: torch.Tensor, required: torch.Tensor,
                        size: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Shortfall max(0, d_c - d_ij) / d_c over related pairs, edge to edge."""
    if pairs.numel() == 0:
        return pos.new_zeros(())
    i, j = pairs[:, 0], pairs[:, 1]
    d = (pos[i] - pos[j]).norm(dim=-1)
    target = required
    if size is not None:
        target = required + ((size[i, :2] + size[j, :2]) / 2).norm(dim=-1)
    return torch.clamp(_softplus_hinge(target - d) / (target + 1e-9), 0.0, 1.0).mean()


def support_violation(pos: torch.Tensor, size: torch.Tensor, parent_pairs: torch.Tensor
                      ) -> torch.Tensor:
    """Footprint area outside the designated supporting region, normalized."""
    if parent_pairs.numel() == 0:
        return pos.new_zeros(())
    i, p = parent_pairs[:, 0], parent_pairs[:, 1]
    reach = torch.clamp(size[p, :2] / 2 - size[i, :2] / 2, min=0.0)
    excess = _softplus_hinge((pos[i] - pos[p]).abs() - reach)
    outside = torch.clamp(excess[:, 0] * size[i, 1] + excess[:, 1] * size[i, 0], min=0.0)
    return torch.clamp(outside / (size[i, 0] * size[i, 1] + 1e-9), 0.0, 1.0).mean()


def frontage_violation(pos: torch.Tensor, road_points: torch.Tensor, admissible: torch.Tensor,
                       mask: torch.Tensor, setback: Optional[torch.Tensor] = None,
                       size: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Deviation from the frontage band, over the admissible distance.

    Both directions count: sitting beyond the admissible distance from the
    designated road edge, and intruding inside the prescribed setback.
    """
    if mask.sum() == 0:
        return pos.new_zeros(())
    d = (pos - road_points).norm(dim=-1)
    ratio = _softplus_hinge(d - admissible) / (admissible + 1e-9)
    if setback is not None:
        clearance = setback
        if size is not None:
            clearance = setback + (size[:, :2] / 2).norm(dim=-1)
        ratio = ratio + _softplus_hinge(clearance - d) / (clearance + 1e-9)
    return torch.clamp(ratio[mask], 0.0, 1.0).mean()


def violation_measures(pos: torch.Tensor, size: torch.Tensor, extent: torch.Tensor,
                       pairs: torch.Tensor, related_pairs: torch.Tensor,
                       parent_pairs: torch.Tensor, road_points: torch.Tensor,
                       admissible: torch.Tensor, frontage_mask: torch.Tensor,
                       required: torch.Tensor,
                       setback: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """All five phi_c for one converged layout state."""
    return {
        "collision": collision_violation(pos, size, pairs),
        "boundary": boundary_violation(pos, size, extent),
        "proximity": proximity_violation(pos, related_pairs, required, size),
        "support": support_violation(pos, size, parent_pairs),
        "frontage": frontage_violation(pos, road_points, admissible, frontage_mask, setback, size),
    }


def alignment_loss(measures: Dict[str, torch.Tensor], weights: Sequence[torch.Tensor],
                   lambdas: Optional[Dict[str, float]] = None, beta: float = 0.01
                   ) -> torch.Tensor:
    """Eq. (5): weighted residual violations plus a unit prior on the weights."""
    lambdas = lambdas or LAMBDA_DEFAULT
    total = sum(lambdas[c] * measures[c] for c in CONSTRAINT_TYPES)
    if weights:
        stacked = torch.cat([w.reshape(-1) for w in weights])
        prior = ((stacked - 1.0) ** 2).mean()
    else:
        prior = total.new_zeros(())
    return total + beta * prior
