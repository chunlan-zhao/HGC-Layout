"""Constraint vocabulary shared by the forces, the weights and the losses.

Frontage denotes alignment of an entity to its designated road edge or
building control line, the outdoor counterpart of an indoor wall-adjacency
constraint. All geometry is handled in normalized scene units, so the base
coefficients are dimensionless and comparable across scenes of different size.
"""
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

CONSTRAINT_TYPES = ("collision", "proximity", "support", "frontage", "boundary")

LAMBDA_DEFAULT: Dict[str, float] = {
    "collision": 1.00,
    "boundary": 1.00,
    "proximity": 0.50,
    "support": 0.50,
    "frontage": 0.30,
}

REQUIRED_DISTANCE_DEFAULT = 0.06   # normalized units, overridden by retrieved rules


def pair_relation(relations: Sequence[Dict]) -> Dict[Tuple[str, str], str]:
    """Map of unordered entity pairs to the planned relation type between them."""
    out: Dict[Tuple[str, str], str] = {}
    for r in relations:
        key = tuple(sorted((r["source"], r["target"])))
        out[key] = r["type"]
    return out


def neighbor_sets(entity_ids: Sequence[str], relations: Dict[Tuple[str, str], str],
                  k_nearest: int = 6, positions: Optional[np.ndarray] = None
                  ) -> Dict[str, List[str]]:
    """Entities sharing a planned relationship, padded with nearest neighbours.

    The planned relations define N(i); when a scene lists few of them, the
    k nearest entities are added so that collision and boundary forces still
    have a neighbourhood to act over.
    """
    out: Dict[str, List[str]] = {e: [] for e in entity_ids}
    for (a, b) in relations:
        if a in out and b in out:
            out[a].append(b)
            out[b].append(a)
    if positions is not None:
        index = {e: i for i, e in enumerate(entity_ids)}
        for e in entity_ids:
            d = np.linalg.norm(positions - positions[index[e]], axis=1)
            order = np.argsort(d)[1:k_nearest + 1]
            for j in order:
                other = entity_ids[j]
                if other not in out[e]:
                    out[e].append(other)
                if e not in out[other]:
                    out[other].append(e)
    return {k: sorted(set(v)) for k, v in out.items()}
