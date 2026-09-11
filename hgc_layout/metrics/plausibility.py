"""Plausibility metrics: whether a layout obeys physical and planning rules.

COL collision avoidance, RCN road connectivity, BUF buffer compliance, GAC
green accessibility and BND boundary adherence, each in [0, 1], higher better.
"""
from typing import Dict, List, Sequence

import numpy as np

SETBACK_M = 18.0
WALKABLE_M = 300.0
SNAP_M = 250.0


def _half(entity: Dict) -> np.ndarray:
    return np.asarray(entity["size"][:2], dtype=float) / 2


def collision_free(entities: Sequence[Dict]) -> float:
    """Share of entity pairs whose footprints do not overlap."""
    n = len(entities)
    if n < 2:
        return 1.0
    ok = total = 0
    for i in range(n):
        for j in range(i + 1, n):
            pi = np.asarray(entities[i]["position"], dtype=float)
            pj = np.asarray(entities[j]["position"], dtype=float)
            reach = _half(entities[i]) + _half(entities[j])
            total += 1
            if np.any(np.abs(pi - pj) >= reach):
                ok += 1
    return float(ok / total)


def boundary_adherence(entities: Sequence[Dict], boundary: Dict[str, float]) -> float:
    """Share of entities whose footprint lies fully inside the boundary."""
    if not entities:
        return 1.0
    extent = np.array([float(boundary["length"]), float(boundary["width"])])
    inside = 0
    for e in entities:
        p, h = np.asarray(e["position"], dtype=float), _half(e)
        inside += int(np.all(p - h >= -1e-6) and np.all(p + h <= extent + 1e-6))
    return float(inside / len(entities))


def _road_axes(roads: Dict) -> Dict[str, List[float]]:
    axes = roads.get("axes")
    if axes:
        return axes
    nodes = roads.get("nodes", [])
    xs = sorted({round(float(n["position"][0]), 3) for n in nodes})
    ys = sorted({round(float(n["position"][1]), 3) for n in nodes})
    return {"x": xs, "y": ys}


def buffer_compliance(entities: Sequence[Dict], roads: Dict, setback: float = SETBACK_M) -> float:
    """Share of buildings that keep the prescribed setback from arterial roads."""
    buildings = [e for e in entities if e["type"] == "building"]
    if not buildings:
        return 1.0
    axes = _road_axes(roads)
    if not axes["x"] and not axes["y"]:
        return 1.0
    ok = 0
    for e in buildings:
        p, h = np.asarray(e["position"], dtype=float), _half(e)
        dx = min((abs(p[0] - x) for x in axes["x"]), default=np.inf) - h[0]
        dy = min((abs(p[1] - y) for y in axes["y"]), default=np.inf) - h[1]
        ok += int(min(dx, dy) >= setback)
    return float(ok / len(buildings))


def green_accessibility(entities: Sequence[Dict], walkable: float = WALKABLE_M) -> float:
    """Share of residential buildings within a walkable distance of green space."""
    residential = [e for e in entities if e["category"] == "residential"]
    greens = [np.asarray(e["position"], dtype=float) for e in entities if e["type"] == "green"]
    if not residential:
        return 1.0
    if not greens:
        return 0.0
    ok = sum(int(min(np.linalg.norm(np.asarray(e["position"], dtype=float) - g) for g in greens)
                 <= walkable) for e in residential)
    return float(ok / len(residential))


def _point_segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance from a point to a line segment."""
    ab = b - a
    denom = float(ab @ ab)
    t = 0.0 if denom == 0 else float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _components(n: int, edges: List[List[int]]) -> List[set]:
    adj: Dict[int, List[int]] = {i: [] for i in range(n)}
    for a, b in edges:
        adj[a].append(b)
        adj[b].append(a)
    seen, comps = set(), []
    for start in range(n):
        if start in seen:
            continue
        stack, comp = [start], set()
        while stack:
            cur = stack.pop()
            if cur in comp:
                continue
            comp.add(cur)
            stack.extend(adj[cur])
        seen |= comp
        comps.append(comp)
    return comps


def road_connectivity(entities: Sequence[Dict], roads: Dict, snap: float = SNAP_M) -> float:
    """Share of entities reachable through the road network.

    Every entity snaps to the nearest road segment within `snap` metres; it
    counts as reachable when both endpoints of that segment lie in the largest
    connected component of the network.
    """
    nodes = roads.get("nodes", [])
    if not entities:
        return 1.0
    if not nodes:
        return 0.0
    index = {node["id"]: i for i, node in enumerate(nodes)}
    pos = np.stack([np.asarray(node["position"], dtype=float) for node in nodes])
    edges = [[index[a], index[b]] for a, b in roads.get("edges", [])
             if a in index and b in index]
    if not edges:
        return 0.0
    comps = _components(len(nodes), edges)
    largest = max(comps, key=len)
    reachable = 0
    for e in entities:
        p = np.asarray(e["position"], dtype=float)
        best, best_edge = np.inf, None
        for a, b in edges:
            d = _point_segment_distance(p, pos[a], pos[b])
            if d < best:
                best, best_edge = d, (a, b)
        if best <= snap and best_edge is not None and best_edge[0] in largest:
            reachable += 1
    return float(reachable / len(entities))
