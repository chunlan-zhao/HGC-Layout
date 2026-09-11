"""Unweighted constraint forces in the fixed-weight functional form.

For every entity pair and constraint type the module returns the planar force
vector, the vertical component acting on the height entry of the scale vector,
and the orientation increment. The adaptive weights of Eq. (4) and the base
coefficients scale these quantities in Eq. (6); with every weight equal to one
the behaviour reduces to the fixed-weight formulation.
All quantities are expressed in normalized scene units.
"""
from typing import Dict, Tuple

import torch

EPS = 1e-8


def _half_extent(size: torch.Tensor) -> torch.Tensor:
    return size[..., :2] / 2


def pair_relation_vector(pos_i: torch.Tensor, pos_j: torch.Tensor,
                         size_i: torch.Tensor, size_j: torch.Tensor) -> torch.Tensor:
    """r_ij of Eq. (4): signed axis distances and axis overlaps, shape (..., 4)."""
    delta = pos_i - pos_j
    reach = _half_extent(size_i) + _half_extent(size_j)
    overlap = torch.clamp(reach - delta.abs(), min=0.0)
    return torch.cat([delta, overlap], dim=-1)


def collision_force(pos_i, pos_j, size_i, size_j) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Repulsion proportional to the penetration depth of the two footprints."""
    delta = pos_i - pos_j
    reach = _half_extent(size_i) + _half_extent(size_j)
    overlap = torch.clamp(reach - delta.abs(), min=0.0)
    penetration = torch.minimum(overlap[..., 0], overlap[..., 1])
    active = (overlap > 0).all(dim=-1).float()
    direction = delta / (delta.norm(dim=-1, keepdim=True) + EPS)
    planar = active.unsqueeze(-1) * penetration.unsqueeze(-1) * direction
    vertical = active * penetration * 0.25
    torque = torch.zeros_like(penetration)
    return planar, vertical, torque


def required_separation(size_i: torch.Tensor, size_j: torch.Tensor,
                        required: torch.Tensor) -> torch.Tensor:
    """Required centre distance: the clearance plus both footprint half-extents.

    Expressing the requirement edge to edge keeps the proximity constraint
    compatible with the collision constraint for entities of any footprint.
    """
    return required + (_half_extent(size_i) + _half_extent(size_j)).norm(dim=-1)


def proximity_force(pos_i, pos_j, size_i, size_j, required: torch.Tensor
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Spring pulling a related pair toward the required edge-to-edge clearance."""
    delta = pos_i - pos_j
    dist = delta.norm(dim=-1)
    direction = delta / (dist.unsqueeze(-1) + EPS)
    shortfall = required_separation(size_i, size_j, required) - dist
    planar = shortfall.unsqueeze(-1) * direction
    zeros = torch.zeros_like(dist)
    return planar, zeros, zeros


def support_force(pos_i, pos_parent, size_i, size_parent, height_i, height_parent
                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pull a supported entity back onto the footprint of its parent."""
    reach = _half_extent(size_parent) - _half_extent(size_i)
    reach = torch.clamp(reach, min=0.0)
    delta = pos_i - pos_parent
    excess = torch.clamp(delta.abs() - reach, min=0.0)
    planar = -torch.sign(delta) * excess
    vertical = (height_parent - height_i) * 0.1
    torque = torch.zeros(planar.shape[:-1], dtype=planar.dtype, device=planar.device)
    return planar, vertical, torque


def frontage_force(pos_i, theta_i, road_point, road_bearing, admissible: torch.Tensor,
                   setback: torch.Tensor = None, size_i: torch.Tensor = None
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hold an entity in the frontage band of its road and align its facade.

    The entity is attracted when it sits farther than the admissible distance
    from the designated road edge and repelled when it intrudes on the control
    line, so that facing the street does not violate the prescribed setback.
    """
    delta = road_point - pos_i
    dist = delta.norm(dim=-1)
    direction = delta / (dist.unsqueeze(-1) + EPS)
    excess = torch.clamp(dist - admissible, min=0.0)
    planar = excess.unsqueeze(-1) * direction
    if setback is not None:
        clearance = setback
        if size_i is not None:
            clearance = setback + _half_extent(size_i).norm(dim=-1)
        intrusion = torch.clamp(clearance - dist, min=0.0)
        planar = planar - intrusion.unsqueeze(-1) * direction
    torque = torch.atan2(torch.sin(road_bearing - theta_i), torch.cos(road_bearing - theta_i))
    vertical = torch.zeros_like(dist)
    return planar, vertical, torque


def boundary_force(pos_i, size_i, extent: torch.Tensor
                   ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Push an entity whose footprint leaves the scene boundary back inside."""
    half = _half_extent(size_i)
    low = torch.clamp(half - pos_i, min=0.0)
    high = torch.clamp(pos_i + half - extent, min=0.0)
    planar = low - high
    zeros = torch.zeros(planar.shape[:-1], dtype=planar.dtype, device=planar.device)
    return planar, zeros, zeros


def constraint_forces(constraint: str, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dispatch to the force function of `constraint`."""
    if constraint == "collision":
        return collision_force(kwargs["pos_i"], kwargs["pos_j"], kwargs["size_i"], kwargs["size_j"])
    if constraint == "proximity":
        return proximity_force(kwargs["pos_i"], kwargs["pos_j"], kwargs["size_i"],
                               kwargs["size_j"], kwargs["required"])
    if constraint == "support":
        return support_force(kwargs["pos_i"], kwargs["pos_j"], kwargs["size_i"], kwargs["size_j"],
                             kwargs["height_i"], kwargs["height_j"])
    if constraint == "frontage":
        return frontage_force(kwargs["pos_i"], kwargs["theta_i"], kwargs["road_point"],
                              kwargs["road_bearing"], kwargs["admissible"],
                              kwargs.get("setback"), kwargs.get("size_i"))
    if constraint == "boundary":
        return boundary_force(kwargs["pos_i"], kwargs["size_i"], kwargs["extent"])
    raise ValueError(f"Unknown constraint type {constraint!r}")
