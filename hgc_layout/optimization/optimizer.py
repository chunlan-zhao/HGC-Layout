"""Adaptive hierarchical optimization with graph-aware forces.

Constraint force magnitudes are modulated by weights predicted from the
pre-trained node embeddings (Eq. 4), summed over neighbours and constraint
types (Eq. 6), and integrated with explicit Euler steps. Deadlocks are detected
from cumulative and net displacement and resolved by a perturbation whose
magnitude scales with the average graph attention among the deadlocked
entities. Optimization stops when the mean residual force falls below
epsilon_conv or after I_max iterations.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from hgc_layout.optimization.constraints import CONSTRAINT_TYPES, LAMBDA_DEFAULT
from hgc_layout.optimization.forces import constraint_forces, pair_relation_vector
from hgc_layout.optimization.weights import AdaptiveWeights


@dataclass
class OptimizerState:
    """Positions, orientations and scales in normalized scene units."""
    pos: torch.Tensor            # (N, 2)
    theta: torch.Tensor          # (N,)
    size: torch.Tensor           # (N, 3), the third entry is the height / storeys
    history: List[float] = field(default_factory=list)
    iterations: int = 0
    deadlocks: int = 0

    def clone(self) -> "OptimizerState":
        return OptimizerState(self.pos.clone(), self.theta.clone(), self.size.clone(),
                              list(self.history), self.iterations, self.deadlocks)


@dataclass
class SceneTensors:
    """Everything the optimizer needs about one scene, in normalized units."""
    pos: torch.Tensor
    theta: torch.Tensor
    size: torch.Tensor
    embeddings: torch.Tensor
    pairs: torch.Tensor                 # (P, 2) neighbour pairs
    related_pairs: torch.Tensor         # (R, 2) pairs with a planned relation
    parent_pairs: torch.Tensor          # (S, 2) entity, parent
    road_points: torch.Tensor           # (N, 2) nearest road point per entity
    road_bearing: torch.Tensor          # (N,)
    frontage_mask: torch.Tensor         # (N,) bool
    extent: torch.Tensor                # (2,)
    required: torch.Tensor              # scalar, required separation
    admissible: torch.Tensor            # scalar, admissible frontage distance
    setback: Optional[torch.Tensor] = None   # scalar, prescribed control-line setback
    attention: Optional[torch.Tensor] = None   # (N,) mean attention per entity


class AdaptiveOptimizer:
    """Force-directed solver with learned, graph-conditioned force weights."""

    def __init__(self, weights: AdaptiveWeights, lambdas: Optional[Dict[str, float]] = None,
                 step_size: float = 0.05, max_iter: int = 500, tol: float = 1e-3,
                 deadlock_cumulative: float = 0.10, deadlock_net: float = 0.20,
                 use_deadlock_evasion: bool = True, perturbation_scale: float = 0.02,
                 seed: int = 0):
        self.weights = weights
        self.lambdas = lambdas or LAMBDA_DEFAULT
        self.step_size = step_size
        self.max_iter = max_iter
        self.tol = tol
        self.deadlock_cumulative = deadlock_cumulative
        self.deadlock_net = deadlock_net
        self.use_deadlock_evasion = use_deadlock_evasion
        self.perturbation_scale = perturbation_scale
        self.generator = torch.Generator().manual_seed(seed)

    def total_forces(self, scene: SceneTensors, state: OptimizerState
                     ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Eq. (6): planar force, vertical force and torque for every entity."""
        n = state.pos.size(0)
        planar = torch.zeros_like(state.pos)
        vertical = torch.zeros(n, dtype=state.pos.dtype, device=state.pos.device)
        torque = torch.zeros_like(vertical)
        used_weights: List[torch.Tensor] = []
        i, j = (scene.pairs[:, 0], scene.pairs[:, 1]) if scene.pairs.numel() else (None, None)

        if i is not None:
            r_ij = pair_relation_vector(state.pos[i], state.pos[j], state.size[i], state.size[j])
            h_i, h_j = scene.embeddings[i], scene.embeddings[j]
            for c in ("collision", "proximity"):
                kwargs = {"pos_i": state.pos[i], "pos_j": state.pos[j],
                          "size_i": state.size[i], "size_j": state.size[j],
                          "required": scene.required}   # proximity is edge to edge
                v, vz, dtheta = constraint_forces(c, **kwargs)
                if c == "proximity":
                    mask = _pair_mask(scene.pairs, scene.related_pairs)
                    v, vz, dtheta = v * mask.unsqueeze(-1), vz * mask, dtheta * mask
                w = self.weights(c, h_i, h_j, r_ij)
                used_weights.append(w)
                scale = self.lambdas[c] * w
                planar.index_add_(0, i, scale.unsqueeze(-1) * v)
                vertical.index_add_(0, i, scale * vz)
                torque.index_add_(0, i, scale * dtheta)

        if scene.parent_pairs.numel():
            a, p = scene.parent_pairs[:, 0], scene.parent_pairs[:, 1]
            r_ap = pair_relation_vector(state.pos[a], state.pos[p], state.size[a], state.size[p])
            v, vz, dtheta = constraint_forces(
                "support", pos_i=state.pos[a], pos_j=state.pos[p], size_i=state.size[a],
                size_j=state.size[p], height_i=state.size[a, 2], height_j=state.size[p, 2])
            w = self.weights("support", scene.embeddings[a], scene.embeddings[p], r_ap)
            used_weights.append(w)
            scale = self.lambdas["support"] * w
            planar.index_add_(0, a, scale.unsqueeze(-1) * v)
            vertical.index_add_(0, a, scale * vz)
            torque.index_add_(0, a, scale * dtheta)

        r_self = pair_relation_vector(state.pos, scene.road_points, state.size, state.size)
        w_front = self.weights("frontage", scene.embeddings, scene.embeddings, r_self)
        used_weights.append(w_front)
        v, vz, dtheta = constraint_forces(
            "frontage", pos_i=state.pos, theta_i=state.theta, road_point=scene.road_points,
            road_bearing=scene.road_bearing, admissible=scene.admissible,
            setback=scene.setback, size_i=state.size)
        mask = scene.frontage_mask.to(v.dtype)
        scale = self.lambdas["frontage"] * w_front * mask
        planar = planar + scale.unsqueeze(-1) * v
        torque = torque + scale * dtheta

        w_bound = self.weights("boundary", scene.embeddings, scene.embeddings, r_self)
        used_weights.append(w_bound)
        v, vz, dtheta = constraint_forces("boundary", pos_i=state.pos, size_i=state.size,
                                          extent=scene.extent)
        scale = self.lambdas["boundary"] * w_bound
        planar = planar + scale.unsqueeze(-1) * v
        return planar, vertical, torque, used_weights

    def step(self, scene: SceneTensors, state: OptimizerState
             ) -> Tuple[OptimizerState, torch.Tensor, List[torch.Tensor]]:
        """One explicit Euler update of positions, orientations and heights."""
        planar, vertical, torque, weights = self.total_forces(scene, state)
        residual = (planar.norm(dim=-1) + vertical.abs() + torque.abs()).mean()
        new = OptimizerState(
            pos=state.pos + self.step_size * planar,
            theta=(state.theta + self.step_size * torque) % (2 * np.pi),
            size=torch.cat([state.size[:, :2],
                            torch.clamp(state.size[:, 2:3] + self.step_size * vertical.unsqueeze(-1),
                                        min=0.0)], dim=-1),
            history=state.history + [float(residual.detach())],
            iterations=state.iterations + 1,
            deadlocks=state.deadlocks)
        return new, residual, weights

    def run(self, scene: SceneTensors, state: Optional[OptimizerState] = None,
            max_iter: Optional[int] = None, track_grad: bool = False
            ) -> Tuple[OptimizerState, List[torch.Tensor]]:
        """Iterate to convergence or I_max, with deadlock evasion."""
        state = state or OptimizerState(scene.pos.clone(), scene.theta.clone(), scene.size.clone())
        max_iter = max_iter or self.max_iter
        cumulative = torch.zeros_like(state.pos[:, 0])
        start = state.pos.clone()
        last_weights: List[torch.Tensor] = []
        context = torch.enable_grad() if track_grad else torch.no_grad()
        with context:
            for _ in range(max_iter):
                previous = state.pos
                state, residual, last_weights = self.step(scene, state)
                cumulative = cumulative + (state.pos - previous).norm(dim=-1)
                if self.use_deadlock_evasion:
                    net = (state.pos - start).norm(dim=-1)
                    stuck = (cumulative > self.deadlock_cumulative) & (net < self.deadlock_net)
                    if bool(stuck.any()):
                        state = self._perturb(scene, state, stuck)
                        cumulative = torch.zeros_like(cumulative)
                        start = state.pos.clone()
                if float(residual.detach()) < self.tol:
                    break
        return state, last_weights

    def _perturb(self, scene: SceneTensors, state: OptimizerState,
                 stuck: torch.Tensor) -> OptimizerState:
        """Inject noise scaled by the average attention among deadlocked entities."""
        attention = scene.attention if scene.attention is not None else torch.ones_like(stuck, dtype=state.pos.dtype)
        magnitude = self.perturbation_scale * attention[stuck].mean().clamp(min=1e-3)
        noise = torch.randn(int(stuck.sum()), 2, generator=self.generator,
                            dtype=state.pos.dtype) * magnitude
        pos = state.pos.clone()
        pos[stuck] = pos[stuck] + noise
        return OptimizerState(pos, state.theta, state.size, state.history,
                              state.iterations, state.deadlocks + 1)


def _pair_mask(pairs: torch.Tensor, subset: torch.Tensor) -> torch.Tensor:
    """1 for pairs that also appear in `subset`, 0 otherwise."""
    if subset.numel() == 0:
        return torch.zeros(pairs.size(0), dtype=torch.float32)
    keys = {(int(a), int(b)) for a, b in subset.tolist()}
    keys |= {(b, a) for a, b in keys}
    return torch.tensor([1.0 if (int(a), int(b)) in keys else 0.0 for a, b in pairs.tolist()],
                        dtype=torch.float32)
