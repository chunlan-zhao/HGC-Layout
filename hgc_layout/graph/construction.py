"""Multi-source heterogeneous graph construction.

Nodes are the planned target entities plus auxiliary nodes for points of
interest, road intersections and land-cover patches. Three edge types connect
them: spatial proximity (Euclidean distance below delta metres), functional
co-occurrence (POI category pairs that co-occur in the retrieved rules) and
road-network edges (intersection to intersection, and POI or entity to its
nearest intersection).

Target entities have no coordinates when the graph is first built: their
positions are the prediction goal. They enter as position-less nodes that
receive context only through message passing over auxiliary neighbours, and
gain proximity and road edges through `add_placement` as the autoregressive
generator places them using model-predicted coordinates. Reference positions
never enter the graph.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from hgc_layout.data.features import (GRAPH_DIM, POI_DIM, RS_DIM, POIEncoder, RemoteSensingEncoder,
                                      build_node_features, structural_features)
from hgc_layout.data.schema import Scene
from hgc_layout.planning.planner import Plan

EDGE_TYPES = ("proximity", "functional", "road")

CO_OCCURRING = {
    ("cafe", "shop"), ("cafe", "market"), ("shop", "market"), ("school", "playground"),
    ("clinic", "pharmacy"), ("bank", "office block"), ("gym", "park"), ("market", "bus stop"),
}


@dataclass
class GeoGraph:
    """Node table, typed edge list and the feature matrix of one scene."""
    node_ids: List[str]
    node_kind: List[str]                      # entity | poi | road | land_cover
    positions: np.ndarray                     # (N, 2), NaN for unplaced entities
    features: np.ndarray                      # (N, D)
    edges: List[Tuple[int, int, str]] = field(default_factory=list)
    delta: float = 200.0
    _index: Dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        self._index = {nid: i for i, nid in enumerate(self.node_ids)}

    @property
    def n_nodes(self) -> int:
        return len(self.node_ids)

    def index(self, node_id: str) -> int:
        return self._index[node_id]

    def entity_indices(self) -> List[int]:
        return [i for i, k in enumerate(self.node_kind) if k == "entity"]

    def edge_index(self, include_self_loops: bool = True) -> np.ndarray:
        """(2, E) array of undirected edges expanded to both directions."""
        src, dst = [], []
        for i, j, _ in self.edges:
            src += [i, j]
            dst += [j, i]
        if include_self_loops:
            src += list(range(self.n_nodes))
            dst += list(range(self.n_nodes))
        if not src:
            return np.zeros((2, 0), dtype=np.int64)
        return np.stack([np.asarray(src), np.asarray(dst)]).astype(np.int64)

    def add_placement(self, node_id: str, position: np.ndarray,
                      roads: Optional[np.ndarray] = None) -> None:
        """Register a model-predicted position and grow the partial graph."""
        i = self.index(node_id)
        self.positions[i] = np.asarray(position, dtype=float)
        for j in range(self.n_nodes):
            if j == i or not np.all(np.isfinite(self.positions[j])):
                continue
            if np.linalg.norm(self.positions[i] - self.positions[j]) <= self.delta:
                if not self.has_edge(i, j, "proximity"):
                    self.edges.append((i, j, "proximity"))
        if roads is not None and len(roads):
            nearest = int(np.argmin(np.linalg.norm(roads - self.positions[i], axis=1)))
            road_nodes = [k for k, kind in enumerate(self.node_kind) if kind == "road"]
            if road_nodes:
                j = road_nodes[nearest % len(road_nodes)]
                if not self.has_edge(i, j, "road"):
                    self.edges.append((i, j, "road"))

    def has_edge(self, i: int, j: int, etype: str) -> bool:
        return any((a, b, t) in ((i, j, etype), (j, i, etype)) for a, b, t in self.edges)

    def neighbors(self, i: int) -> List[int]:
        out = []
        for a, b, _ in self.edges:
            if a == i:
                out.append(b)
            elif b == i:
                out.append(a)
        return sorted(set(out))

    def degree_by_type(self, i: int) -> Dict[str, int]:
        counts = {t: 0 for t in EDGE_TYPES}
        for a, b, t in self.edges:
            if i in (a, b):
                counts[t] += 1
        return counts


def _clustering(graph_edges: List[Tuple[int, int, str]], i: int, neighbors: List[int]) -> float:
    if len(neighbors) < 2:
        return 0.0
    pairs = {(min(a, b), max(a, b)) for a, b, _ in graph_edges}
    links = sum(1 for a_idx in range(len(neighbors)) for b_idx in range(a_idx + 1, len(neighbors))
                if (min(neighbors[a_idx], neighbors[b_idx]),
                    max(neighbors[a_idx], neighbors[b_idx])) in pairs)
    return 2.0 * links / (len(neighbors) * (len(neighbors) - 1))


FEATURE_BLOCKS = ("rs", "poi", "graph")


def block_mask(blocks: Optional[Sequence[str]]) -> np.ndarray:
    """Channel mask keeping only the requested feature blocks."""
    blocks = tuple(blocks) if blocks else FEATURE_BLOCKS
    parts = [np.full(RS_DIM, float("rs" in blocks), dtype=np.float32),
             np.full(POI_DIM, float("poi" in blocks), dtype=np.float32),
             np.full(GRAPH_DIM, float("graph" in blocks), dtype=np.float32)]
    return np.concatenate(parts)


def build_graph(scene: Scene, plan: Plan, delta: float = 200.0,
                rs_encoder: Optional[RemoteSensingEncoder] = None,
                poi_encoder: Optional[POIEncoder] = None,
                feature_blocks: Optional[Sequence[str]] = None,
                use_poi_nodes: bool = True, use_road_edges: bool = True) -> GeoGraph:
    """Build the heterogeneous graph for one scene from auxiliary data only.

    `feature_blocks` restricts the multi-modal node features to a subset of
    {rs, poi, graph}; `use_poi_nodes` and `use_road_edges` drop the POI
    auxiliary nodes or the road-network edges for the module ablations.
    """
    rs_encoder = rs_encoder or RemoteSensingEncoder()
    poi_encoder = poi_encoder or POIEncoder()

    node_ids: List[str] = []
    node_kind: List[str] = []
    positions: List[np.ndarray] = []
    texts: List[str] = []
    stats: List[np.ndarray] = []
    default_stats = np.full(8, 0.25, dtype=np.float32)

    for e in plan.entities:                       # target entities: no coordinates yet
        node_ids.append(e["id"])
        node_kind.append("entity")
        positions.append(np.array([np.nan, np.nan]))
        texts.append(f'{e["category"]} {e["description"]}')
        stats.append(np.asarray(scene.rs_stats.get(e["id"], default_stats), dtype=np.float32))
    for p in (scene.pois if use_poi_nodes else []):   # auxiliary nodes with known coordinates
        node_ids.append(p["id"])
        node_kind.append("poi")
        positions.append(np.asarray(p["position"], dtype=float))
        texts.append(p["category"])
        stats.append(np.asarray(scene.rs_stats.get(p["id"], default_stats), dtype=np.float32))
    for r in scene.roads.get("nodes", []):
        node_ids.append(r["id"])
        node_kind.append("road")
        positions.append(np.asarray(r["position"], dtype=float))
        texts.append("road intersection")
        stats.append(default_stats)
    for l in scene.land_cover:
        node_ids.append(l["id"])
        node_kind.append("land_cover")
        positions.append(np.asarray(l["position"], dtype=float))
        texts.append(f'land cover {l["class"]}')
        stats.append(np.asarray(scene.rs_stats.get(l["id"], default_stats), dtype=np.float32))

    pos = np.stack(positions) if positions else np.zeros((0, 2))
    edges: List[Tuple[int, int, str]] = []
    known = [i for i in range(len(node_ids)) if np.all(np.isfinite(pos[i]))]
    for a_idx, i in enumerate(known):             # proximity among auxiliary nodes
        for j in known[a_idx + 1:]:
            if np.linalg.norm(pos[i] - pos[j]) <= delta:
                edges.append((i, j, "proximity"))
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    for a, b in (scene.roads.get("edges", []) if use_road_edges else []):   # road network
        if a in id_to_idx and b in id_to_idx:
            edges.append((id_to_idx[a], id_to_idx[b], "road"))
    road_idx = [i for i, k in enumerate(node_kind) if k == "road"] if use_road_edges else []
    for i, k in enumerate(node_kind):             # POI to nearest intersection
        if k == "poi" and road_idx:
            j = min(road_idx, key=lambda r: float(np.linalg.norm(pos[i] - pos[r])))
            edges.append((i, j, "road"))
    for a_idx, i in enumerate(range(len(node_ids))):   # functional co-occurrence
        for j in range(i + 1, len(node_ids)):
            pair = (texts[i].split()[-1], texts[j].split()[-1])
            if pair in CO_OCCURRING or pair[::-1] in CO_OCCURRING:
                edges.append((i, j, "functional"))

    graph = GeoGraph(node_ids=node_ids, node_kind=node_kind, positions=pos,
                     features=np.zeros((len(node_ids), 1), dtype=np.float32),
                     edges=edges, delta=delta)
    feats = []
    for i, nid in enumerate(node_ids):
        neigh = graph.neighbors(i)
        struct = structural_features(len(neigh), graph.degree_by_type(i),
                                     _clustering(edges, i, neigh))
        feats.append(build_node_features(rs_encoder.encode(stats[i], nid),
                                         poi_encoder.encode(texts[i]), struct))
    mask = block_mask(feature_blocks)
    graph.features = (np.stack(feats).astype(np.float32) * mask[None, :]
                      if feats else np.zeros((0, feature_width()), dtype=np.float32))
    return graph


def feature_width() -> int:
    return RS_DIM + POI_DIM + GRAPH_DIM
