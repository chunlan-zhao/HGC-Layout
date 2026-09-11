"""Per-scene evaluation across all reported metrics."""
import time
from typing import Dict, Optional, Sequence

from hgc_layout.data.schema import Scene
from hgc_layout.metrics.fidelity import area_ratio, category_match, entity_count, spatial_relationship
from hgc_layout.metrics.plausibility import (boundary_adherence, buffer_compliance, collision_free,
                                             green_accessibility, road_connectivity)
from hgc_layout.metrics.semantic import clip_similarity, semantic_plausibility

METRIC_NAMES = ("CNT", "CAT", "SPR", "ARE", "COL", "RCN", "BUF", "GAC", "BND",
                "CLIPsim", "SP", "Time")

FIDELITY = ("CNT", "CAT", "SPR", "ARE")
PLAUSIBILITY = ("COL", "RCN", "BUF", "GAC", "BND")
HIGHER_IS_BETTER = {m: True for m in METRIC_NAMES}
HIGHER_IS_BETTER["Time"] = False


def evaluate_layout(entities: Sequence[Dict], scene: Scene, elapsed: float = 0.0,
                    sp_model: Optional[str] = None, clip_model: Optional[object] = None,
                    setback: float = 18.0, walkable: float = 300.0) -> Dict[str, float]:
    """All twelve reported quantities for one generated layout."""
    sp = semantic_plausibility(entities, scene, model=sp_model)
    return {
        "CNT": entity_count(entities, scene.entities),
        "CAT": category_match(entities, scene.entities),
        "SPR": spatial_relationship(entities, scene.relations),
        "ARE": area_ratio(entities, scene.area_ratio),
        "COL": collision_free(entities),
        "RCN": road_connectivity(entities, scene.roads),
        "BUF": buffer_compliance(entities, scene.roads, setback),
        "GAC": green_accessibility(entities, walkable),
        "BND": boundary_adherence(entities, scene.boundary),
        "CLIPsim": clip_similarity(entities, scene.description, scene.boundary, clip_model),
        "SP": float(sp["score"]),
        "Time": float(elapsed),
    }


class Timer:
    """Wall-clock timer for the per-scene generation time."""

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self._start
        return False
