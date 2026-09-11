"""Synthetic benchmark in the GeoSceneBench format.

Each scene is a square region containing a road grid, POIs, land-cover patches
and a set of entities grouped into functional zones. Entity positions are laid
out so that the qualitative structure the framework is meant to exploit is
present: entities respect the boundary, avoid overlaps, keep a setback from
arterial roads, front onto the nearest road, and residential buildings sit
within walking distance of a green patch. Descriptions, relations and area
ratios are derived from the generated layout, so fidelity metrics are
well defined. The data are synthetic; any numbers computed from them are
illustrative only.
"""
from typing import Dict, List, Tuple

import numpy as np

from hgc_layout.data.schema import CATEGORIES, DIFFICULTIES, Scene

GROUPS_BY_CATEGORY = {
    "residential area": ["residential zone", "green belt"],
    "commercial center": ["commercial core", "transport hub"],
    "industrial park": ["industrial zone", "transport hub"],
    "waterfront": ["waterfront", "green belt"],
    "transport hub": ["transport hub", "commercial core"],
    "mixed-use neighborhood": ["commercial core", "residential zone", "green belt"],
    "civic district": ["commercial core", "green belt"],
    "campus": ["residential zone", "green belt"],
    "retail cluster": ["commercial core"],
    "logistics park": ["industrial zone", "transport hub"],
}

ENTITY_MENU = {
    "commercial core": [("building", "retail", "shop"), ("building", "retail", "supermarket"),
                        ("building", "office", "office block"), ("poi", "amenity", "bus stop")],
    "residential zone": [("building", "residential", "apartment block"),
                         ("building", "residential", "detached house"),
                         ("poi", "amenity", "playground")],
    "industrial zone": [("building", "industrial", "warehouse"),
                        ("building", "industrial", "factory hall"),
                        ("poi", "amenity", "loading dock")],
    "green belt": [("green", "park", "park"), ("green", "park", "garden"),
                   ("green", "park", "linear green belt")],
    "transport hub": [("building", "transport", "station building"),
                      ("poi", "amenity", "bus stop"), ("poi", "amenity", "parking access")],
    "waterfront": [("water", "water", "basin"), ("green", "park", "promenade"),
                   ("building", "retail", "boathouse")],
}

SIZE_BY_CATEGORY = {
    "retail": (30.0, 22.0, 8.0), "office": (45.0, 35.0, 24.0),
    "residential": (35.0, 25.0, 15.0), "industrial": (60.0, 40.0, 12.0),
    "transport": (55.0, 35.0, 10.0), "park": (80.0, 60.0, 0.5),
    "water": (120.0, 70.0, 0.0), "amenity": (8.0, 8.0, 3.0),
}

N_ENTITIES = {"easy": (6, 9), "medium": (10, 19), "hard": (20, 28)}
BOUNDARY_M = 1000.0
ROAD_SETBACK_M = 18.0
WALKABLE_M = 300.0


def _road_grid(rng: np.random.Generator, size: float) -> Dict[str, List]:
    """Two to four arterials plus their intersections, as a node/edge graph."""
    xs = sorted(rng.choice(np.linspace(0.18, 0.82, 7), size=int(rng.integers(2, 4)), replace=False) * size)
    ys = sorted(rng.choice(np.linspace(0.18, 0.82, 7), size=int(rng.integers(2, 4)), replace=False) * size)
    nodes, index = [], {}
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            nid = f"r_{i}_{j}"
            index[(i, j)] = nid
            nodes.append({"id": nid, "position": [float(x), float(y)], "kind": "intersection"})
    edges = []
    for i in range(len(xs)):
        for j in range(len(ys)):
            if i + 1 < len(xs):
                edges.append([index[(i, j)], index[(i + 1, j)]])
            if j + 1 < len(ys):
                edges.append([index[(i, j)], index[(i, j + 1)]])
    return {"nodes": nodes, "edges": edges, "axes": {"x": [float(v) for v in xs],
                                                     "y": [float(v) for v in ys]}}


def _road_distance(p: np.ndarray, axes: Dict[str, List[float]]) -> Tuple[float, float]:
    """Distance to the nearest arterial and the bearing of that arterial."""
    dx = min(abs(p[0] - x) for x in axes["x"])
    dy = min(abs(p[1] - y) for y in axes["y"])
    return (dx, 0.0) if dx <= dy else (dy, np.pi / 2)


def _zone_centres(rng: np.random.Generator, groups: List[str], size: float) -> Dict[str, np.ndarray]:
    """One centre per functional zone, spread across the region."""
    angles = rng.permutation(np.linspace(0, 2 * np.pi, len(groups), endpoint=False))
    radius = 0.26 * size
    return {g: np.array([0.5 * size + radius * np.cos(a), 0.5 * size + radius * np.sin(a)])
            for g, a in zip(groups, angles)}


def _place(rng, centre, size, footprint, placed, axes, boundary) -> np.ndarray:
    """Rejection sampling: inside boundary, clear of others, set back from arterials."""
    half = footprint[:2] / 2
    for attempt in range(60):
        p = centre + rng.normal(0, 0.10 * boundary, size=2)
        p = np.clip(p, half + 5.0, boundary - half - 5.0)
        d_road, _ = _road_distance(p, axes)
        if d_road < ROAD_SETBACK_M + half.max() and attempt < 40:
            continue
        if all(np.any(np.abs(p - q) > (half + qh + 6.0)) for q, qh in placed):
            return p
    return p


def _make_scene(rng: np.random.Generator, idx: int, category: str, difficulty: str,
                tile_id: str, split: str) -> Scene:
    size = BOUNDARY_M
    roads = _road_grid(rng, size)
    axes = roads["axes"]
    groups = GROUPS_BY_CATEGORY[category]
    centres = _zone_centres(rng, groups, size)
    lo, hi = N_ENTITIES[difficulty]
    n_entities = int(rng.integers(lo, hi + 1))

    entities, placed = [], []
    for k in range(n_entities):
        group = groups[k % len(groups)]
        etype, ecat, label = ENTITY_MENU[group][int(rng.integers(0, len(ENTITY_MENU[group])))]
        base = np.array(SIZE_BY_CATEGORY[ecat], dtype=float)
        footprint = base * rng.uniform(0.85, 1.2, size=3)
        p = _place(rng, centres[group], size, footprint, placed, axes, size)
        placed.append((p, footprint[:2] / 2))
        _, bearing = _road_distance(p, axes)
        theta = float((bearing + rng.normal(0, 0.12)) % (2 * np.pi))
        entities.append({
            "id": f"e{k:02d}", "type": etype, "category": ecat, "description": label,
            "size": [round(float(v), 2) for v in footprint], "group": group, "parent": None,
            "position": [round(float(p[0]), 2), round(float(p[1]), 2)],
            "orientation": round(theta, 4),
        })

    # green patches guarantee walkable access for residential buildings
    greens = [np.array(e["position"]) for e in entities if e["type"] == "green"]
    if not greens:
        g = np.array([0.5 * size, 0.5 * size])
        entities.append({"id": f"e{len(entities):02d}", "type": "green", "category": "park",
                         "description": "park", "size": [90.0, 70.0, 0.5],
                         "group": groups[-1], "parent": None,
                         "position": [float(g[0]), float(g[1])], "orientation": 0.0})
        greens = [g]
    for e in entities:
        if e["category"] == "residential":
            p = np.array(e["position"])
            nearest = min(greens, key=lambda g: np.linalg.norm(p - g))
            if np.linalg.norm(p - nearest) > WALKABLE_M:
                direction = (nearest - p) / (np.linalg.norm(nearest - p) + 1e-9)
                p = nearest - direction * (WALKABLE_M * 0.8)
                e["position"] = [round(float(p[0]), 2), round(float(p[1]), 2)]

    pois = [{"id": f"p{i:02d}", "category": str(rng.choice(["cafe", "school", "clinic", "bank",
                                                            "market", "pharmacy", "gym"])),
             "position": [float(rng.uniform(0.05, 0.95) * size), float(rng.uniform(0.05, 0.95) * size)]}
            for i in range(int(rng.integers(12, 25)))]
    land_cover = [{"id": f"l{i:02d}", "class": str(rng.choice(["built", "vegetation", "water", "bare"])),
                   "position": [float(rng.uniform(0, 1) * size), float(rng.uniform(0, 1) * size)]}
                  for i in range(10)]

    relations = _derive_relations(entities, rng)
    area_ratio = _area_ratio(entities)
    rs_stats = _rs_stats(entities, pois, land_cover, rng)
    description = _describe(category, difficulty, entities, relations)
    return Scene(scene_id=f"scene_{idx:03d}", category=category, difficulty=difficulty,
                 tile_id=tile_id, description=description,
                 boundary={"length": size, "width": size}, entities=entities,
                 relations=relations, area_ratio=area_ratio, pois=pois,
                 roads={"nodes": roads["nodes"], "edges": roads["edges"], "axes": axes},
                 land_cover=land_cover, rs_stats=rs_stats, split=split)


def _derive_relations(entities: List[Dict], rng: np.random.Generator) -> List[Dict]:
    """Relations read off the reference layout, used as the fidelity target.

    Thresholds sit inside the tolerances the fidelity metric applies, so the
    reference layout satisfies every relation it declares.
    """
    pos = {e["id"]: np.array(e["position"]) for e in entities}
    theta = {e["id"]: float(e["orientation"]) for e in entities}
    label = {e["id"]: e["description"] for e in entities}
    rel: List[Dict] = []
    ids = list(pos)

    def add(a: str, b: str, kind: str) -> None:
        rel.append({"source": a, "target": b, "type": kind,
                    "source_label": label[a], "target_label": label[b]})

    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            d = float(np.linalg.norm(pos[a] - pos[b]))
            if d < 140.0:
                add(a, b, "adjacent_to")
            elif d < 280.0 and rng.random() < 0.35:
                add(a, b, "near")
    for a in ids:                      # facing only where the reference pose agrees
        for b in ids:
            if a == b or len(rel) > 36:
                continue
            delta = pos[b] - pos[a]
            bearing = float(np.arctan2(delta[1], delta[0])) % (2 * np.pi)
            diff = abs((bearing - theta[a] + np.pi) % (2 * np.pi) - np.pi)
            if diff < np.pi / 8 and rng.random() < 0.5:
                add(a, b, "facing")
                break
    return rel[:40]


def _area_ratio(entities: List[Dict]) -> Dict[str, float]:
    """Footprint-area share per functional group."""
    areas: Dict[str, float] = {}
    for e in entities:
        areas[e["group"]] = areas.get(e["group"], 0.0) + e["size"][0] * e["size"][1]
    total = sum(areas.values()) or 1.0
    return {g: round(a / total, 4) for g, a in areas.items()}


def _rs_stats(entities, pois, land_cover, rng) -> Dict[str, List[float]]:
    """Compact remote-sensing summary per node, standing in for image patches.

    Eight channels: built-up, vegetation, water, bare fractions, NDVI mean and
    standard deviation, brightness, and texture energy.
    """
    out: Dict[str, List[float]] = {}
    for e in entities:
        base = {"building": [0.75, 0.10, 0.02, 0.13], "green": [0.08, 0.85, 0.03, 0.04],
                "water": [0.04, 0.08, 0.86, 0.02], "road": [0.55, 0.10, 0.02, 0.33],
                "poi": [0.60, 0.20, 0.05, 0.15]}[e["type"]]
        frac = np.clip(np.array(base) + rng.normal(0, 0.03, 4), 0, 1)
        frac = frac / frac.sum()
        ndvi = float(0.8 * frac[1] - 0.2 * frac[0] + rng.normal(0, 0.02))
        out[e["id"]] = [round(float(v), 4) for v in
                        list(frac) + [ndvi, float(abs(rng.normal(0.08, 0.02))),
                                      float(np.clip(rng.normal(0.5, 0.1), 0, 1)),
                                      float(abs(rng.normal(0.3, 0.05)))]]
    for p in pois:
        out[p["id"]] = [round(float(v), 4) for v in np.clip(rng.normal(0.4, 0.15, 8), 0, 1)]
    for l in land_cover:
        out[l["id"]] = [round(float(v), 4) for v in np.clip(rng.normal(0.4, 0.15, 8), 0, 1)]
    return out


def _describe(category: str, difficulty: str, entities: List[Dict], relations: List[Dict]) -> str:
    """Natural-language prompt consistent with the generated layout."""
    counts: Dict[str, int] = {}
    for e in entities:
        counts[e["description"]] = counts.get(e["description"], 0) + 1
    parts = [f"{n} {label}{'s' if n > 1 else ''}" for label, n in sorted(counts.items())]
    listing = ", ".join(parts[:-1]) + (" and " + parts[-1] if len(parts) > 1 else parts[0])
    adjacency = sum(1 for r in relations if r["type"] == "adjacent_to")
    tail = (" with several adjacent frontages" if adjacency > 3 else
            " with generous separation between blocks")
    return (f"A {difficulty} {category} of roughly one square kilometre containing {listing}, "
            f"arranged along the arterial road grid{tail}.")


def generate_benchmark(n_scenes: int = 100, seed: int = 0,
                       split_ratio: Tuple[float, float, float] = (0.6, 0.2, 0.2)) -> List[Scene]:
    """Generate `n_scenes` scenes with a geographically separated tile split.

    Scenes drawn from the same tile always land in the same split, so spatial
    adjacency cannot leak across splits.
    """
    rng = np.random.default_rng(seed)
    n_tiles = max(3, n_scenes // 4)
    tile_ids = [f"tile_{i:03d}" for i in range(n_tiles)]
    perm = rng.permutation(n_tiles)
    n_train = int(round(split_ratio[0] * n_tiles))
    n_val = int(round(split_ratio[1] * n_tiles))
    # keep every split non-empty, which matters for very small benchmarks
    n_train = min(max(n_train, 1), n_tiles - 2)
    n_val = min(max(n_val, 1), n_tiles - n_train - 1)
    tile_split = {}
    for rank, t in enumerate(perm):
        tile_split[tile_ids[t]] = ("train" if rank < n_train else
                                   "val" if rank < n_train + n_val else "test")
    scenes = []
    for i in range(n_scenes):
        tile = tile_ids[i % n_tiles]
        category = CATEGORIES[i % len(CATEGORIES)]
        difficulty = DIFFICULTIES[i % len(DIFFICULTIES)]
        scenes.append(_make_scene(rng, i, category, difficulty, tile, tile_split[tile]))
    return scenes
