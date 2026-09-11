"""Scene schema for the geospatial layout benchmark.

One scene is one JSON file. Coordinates are metres inside a square scene
boundary, orientations are radians in [0, 2*pi). Entity `position` and
`orientation` hold the reference layout: they are the evaluation target and
are never exposed to a generator at inference time.
"""
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

ENTITY_TYPES = ("building", "road", "green", "water", "poi")

FUNCTIONAL_GROUPS = ("commercial core", "residential zone", "industrial zone",
                     "green belt", "transport hub", "waterfront")

CATEGORIES = ("residential area", "commercial center", "industrial park", "waterfront",
              "transport hub", "mixed-use neighborhood", "civic district", "campus",
              "retail cluster", "logistics park")

DIFFICULTIES = ("easy", "medium", "hard")

RELATION_TYPES = ("adjacent_to", "facing", "surrounding", "near", "aligned_with")

SCENE_REQUIRED_KEYS = ("scene_id", "category", "difficulty", "tile_id", "description",
                       "boundary", "entities")


@dataclass
class Entity:
    """One placeable object with its reference pose."""
    id: str
    type: str
    category: str
    description: str
    size: Tuple[float, float, float]
    group: str
    parent: Optional[str] = None
    position: Optional[Tuple[float, float]] = None
    orientation: Optional[float] = None

    def as_dict(self) -> Dict:
        return asdict(self)


@dataclass
class Scene:
    """A benchmark scene: prompt, boundary, entities and auxiliary geospatial data."""
    scene_id: str
    category: str
    difficulty: str
    tile_id: str
    description: str
    boundary: Dict[str, float]
    entities: List[Dict]
    relations: List[Dict] = field(default_factory=list)
    area_ratio: Dict[str, float] = field(default_factory=dict)
    pois: List[Dict] = field(default_factory=list)
    roads: Dict[str, List] = field(default_factory=lambda: {"nodes": [], "edges": []})
    land_cover: List[Dict] = field(default_factory=list)
    rs_stats: Dict[str, List[float]] = field(default_factory=dict)
    split: str = "train"

    def as_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "Scene":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    @property
    def size(self) -> Tuple[float, float]:
        return float(self.boundary["length"]), float(self.boundary["width"])

    def entity_by_id(self, eid: str) -> Dict:
        for e in self.entities:
            if e["id"] == eid:
                return e
        raise KeyError(eid)

    def groups(self) -> List[str]:
        seen: List[str] = []
        for e in self.entities:
            if e["group"] not in seen:
                seen.append(e["group"])
        return seen
