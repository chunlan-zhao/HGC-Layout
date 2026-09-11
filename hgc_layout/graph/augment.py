"""Multi-view augmentation for graph contrastive pre-training.

Feature dropout masks a fraction rho of node feature channels; edge
perturbation removes a fraction of edges and inserts a few spurious ones,
standing in for missing or noisy road and proximity links.
"""
import copy
from typing import Tuple

import numpy as np

from hgc_layout.graph.construction import EDGE_TYPES, GeoGraph


def augment_view(graph: GeoGraph, rng: np.random.Generator, feature_dropout: float = 0.20,
                 edge_removal: float = 0.20, edge_addition: float = 0.05) -> GeoGraph:
    """Return an independently corrupted view of `graph`."""
    view = copy.copy(graph)
    mask = (rng.random(graph.features.shape[1]) > feature_dropout).astype(np.float32)
    view.features = graph.features * mask[None, :]
    kept = [e for e in graph.edges if rng.random() > edge_removal]
    n_add = int(round(edge_addition * max(len(graph.edges), 1)))
    n = graph.n_nodes
    spurious = []
    for _ in range(n_add):
        i, j = int(rng.integers(0, n)), int(rng.integers(0, n))
        if i != j:
            spurious.append((i, j, EDGE_TYPES[int(rng.integers(0, len(EDGE_TYPES)))]))
    view.edges = kept + spurious
    return view
