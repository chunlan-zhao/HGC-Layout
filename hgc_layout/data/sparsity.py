"""Artificial degradation of the auxiliary geospatial inputs.

Drops a fraction of POI records together with a proportional removal of road
segments, used for the robustness-to-sparsity evaluation. Entities and the
reference layout are never touched.
"""
import copy
from typing import List

import numpy as np

from hgc_layout.data.schema import Scene


def degrade_scene(scene: Scene, dropout: float, rng: np.random.Generator) -> Scene:
    """Return a copy of `scene` with `dropout` of POIs and road edges removed."""
    if dropout <= 0:
        return scene
    s = copy.deepcopy(scene)
    keep_poi = [p for p in s.pois if rng.random() > dropout]
    s.pois = keep_poi
    edges: List = s.roads.get("edges", [])
    s.roads = {"nodes": s.roads.get("nodes", []),
               "edges": [e for e in edges if rng.random() > dropout]}
    kept_ids = {p["id"] for p in keep_poi}
    s.rs_stats = {k: v for k, v in s.rs_stats.items()
                  if not k.startswith("poi_") or k in kept_ids}
    return s
