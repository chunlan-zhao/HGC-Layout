"""Common interface for the compared methods.

Every baseline receives the same text description and the same planned entity
list, so that differences come from the placement mechanism rather than from
the entity inventory.
"""
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from hgc_layout.data.features import _hash_embedding
from hgc_layout.data.schema import Scene
from hgc_layout.planning.planner import Plan, TemplatePlanner

TEXT_DIM = 128


class BaselineMethod:
    """Interface shared by the baselines and by the framework wrapper."""

    name = "baseline"

    def fit(self, scenes: Sequence[Scene], epochs: int = 0, seed: int = 0) -> None:
        """Train on the training split; no-op for training-free methods."""

    def generate(self, scene: Scene, seed: int = 0) -> List[Dict]:
        raise NotImplementedError


def plan_entities(planner: TemplatePlanner, scene: Scene) -> Plan:
    return planner.plan(scene)


def text_features(entity: Dict, description: str, index: int, n_entities: int) -> np.ndarray:
    """Entity and prompt embedding plus sequence position, shared by baselines."""
    e = _hash_embedding(f'{entity["category"]} {entity["description"]}', TEXT_DIM, "base")
    d = _hash_embedding(description, TEXT_DIM, "prompt")
    extra = np.array([index / max(n_entities - 1, 1), float(entity["size"][0]) / 100.0,
                      float(entity["size"][1]) / 100.0, float(n_entities) / 30.0],
                     dtype=np.float32)
    return np.concatenate([e, d, extra]).astype(np.float32)


def feature_dim() -> int:
    return 2 * TEXT_DIM + 4


def targets_from_scene(scene: Scene) -> Dict[str, np.ndarray]:
    scale = np.array([float(scene.boundary["length"]), float(scene.boundary["width"])])
    return {e["id"]: np.asarray(e["position"], dtype=float) / scale for e in scene.entities}


def to_entities(plan: Plan, coords: np.ndarray, thetas: Optional[np.ndarray],
                boundary: Dict[str, float]) -> List[Dict]:
    """Attach normalized coordinates to planned entities, clipped to the boundary."""
    scale = np.array([float(boundary["length"]), float(boundary["width"])])
    out = []
    for k, entity in enumerate(plan.entities):
        e = dict(entity)
        p = np.clip(coords[k], 0.0, 1.0) * scale
        half = np.asarray(e["size"][:2], dtype=float) / 2
        e["position"] = [float(v) for v in np.clip(p, half, scale - half)]
        e["orientation"] = float(thetas[k] % (2 * np.pi)) if thetas is not None else 0.0
        out.append(e)
    return out


def training_pairs(planner: TemplatePlanner, scenes: Sequence[Scene]):
    """Yield (features, target) arrays for supervised baselines."""
    xs, ys = [], []
    for scene in scenes:
        plan = planner.plan(scene)
        targets = targets_from_scene(scene)
        ids = list(targets)
        for k, entity in enumerate(plan.entities):
            key = entity["id"] if entity["id"] in targets else (ids[k] if k < len(ids) else None)
            if key is None:
                continue
            xs.append(text_features(entity, scene.description, k, len(plan.entities)))
            ys.append(targets[key])
    if not xs:
        return torch.zeros(0, feature_dim()), torch.zeros(0, 2)
    return (torch.as_tensor(np.stack(xs), dtype=torch.float32),
            torch.as_tensor(np.stack(ys), dtype=torch.float32))
