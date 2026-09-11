"""End-to-end HGC-Layout pipeline.

Planning, graph construction, contrastive embedding, graph-conditioned
generation and adaptive optimization are chained here, with the switches used
by the ablations exposed as constructor arguments.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from hgc_layout.data.schema import Scene
from hgc_layout.generation.decoder import GraphConditionedDecoder
from hgc_layout.generation.generator import GeneratedLayout, LayoutGenerator
from hgc_layout.graph.construction import GeoGraph, build_graph, feature_width
from hgc_layout.graph.gat import GATEncoder
from hgc_layout.optimization.constraints import CONSTRAINT_TYPES, LAMBDA_DEFAULT, neighbor_sets, pair_relation
from hgc_layout.optimization.optimizer import AdaptiveOptimizer, OptimizerState, SceneTensors
from hgc_layout.optimization.weights import AdaptiveWeights, FixedWeights
from hgc_layout.planning.planner import Plan, TemplatePlanner

log = logging.getLogger(__name__)

DEFAULT_REQUIRED_M = 60.0
DEFAULT_FRONTAGE_M = 160.0
DEFAULT_SETBACK_M = 18.0


@dataclass
class LayoutResult:
    """Generated entities plus the intermediate artefacts the analyses need."""
    entities: List[Dict]
    traces: List = field(default_factory=list)
    residuals: List[float] = field(default_factory=list)
    iterations: int = 0
    deadlocks: int = 0
    graph: Optional[GeoGraph] = None
    plan: Optional[Plan] = None


class HGCLayout:
    """The full framework, with per-component switches for the ablations."""

    def __init__(self, planner: TemplatePlanner, encoder: Optional[GATEncoder] = None,
                 decoder: Optional[GraphConditionedDecoder] = None,
                 weights: Optional[torch.nn.Module] = None,
                 optimizer: Optional[AdaptiveOptimizer] = None,
                 delta: float = 200.0, use_graph_conditioning: bool = True,
                 use_adaptive_weights: bool = True, use_optimizer: bool = True,
                 embed_dim: int = 256, device: Optional[torch.device] = None,
                 feature_blocks: Optional[Sequence[str]] = None, use_poi_nodes: bool = True,
                 use_road_edges: bool = True, use_rule_retrieval: bool = True,
                 use_global_context: bool = True, use_hierarchical_groups: bool = True):
        self.planner = planner
        self.device = device or torch.device("cpu")
        self.encoder = encoder or GATEncoder(feature_width(), out_dim=embed_dim)
        self.decoder = decoder or GraphConditionedDecoder(embed_dim=embed_dim)
        self.use_graph_conditioning = use_graph_conditioning
        self.use_adaptive_weights = use_adaptive_weights
        self.use_optimizer = use_optimizer
        self.weights = weights or (AdaptiveWeights(embed_dim) if use_adaptive_weights else FixedWeights())
        self.optimizer = optimizer or AdaptiveOptimizer(self.weights)
        self.delta = delta
        self.feature_blocks = feature_blocks
        self.use_poi_nodes = use_poi_nodes
        self.use_road_edges = use_road_edges
        self.use_rule_retrieval = use_rule_retrieval
        self.use_global_context = use_global_context
        self.use_hierarchical_groups = use_hierarchical_groups

    def plan_scene(self, scene: Scene) -> Plan:
        plan = self.planner.plan(scene)
        if not self.use_rule_retrieval:
            plan.rules, plan.constraints = [], {}
        if not self.use_global_context:
            plan.global_context = ""
        if not self.use_hierarchical_groups:
            for entity in plan.entities:
                entity["group"] = "scene"
        return plan

    def build_graph(self, scene: Scene, plan: Plan) -> GeoGraph:
        graph = build_graph(scene, plan, delta=self.delta, feature_blocks=self.feature_blocks,
                            use_poi_nodes=self.use_poi_nodes, use_road_edges=self.use_road_edges)
        if not self.use_graph_conditioning:
            graph = _strip_entity_edges(graph)
        return graph

    def generate(self, scene: Scene, seed: int = 0, sample: bool = False) -> LayoutResult:
        """Plan, place and refine one scene."""
        plan = self.plan_scene(scene)
        graph = self.build_graph(scene, plan)
        roads = _road_points(scene)
        generator = LayoutGenerator(self.encoder, self.decoder, self.device, sample=sample)
        layout: GeneratedLayout = generator.generate(graph, plan, roads, seed=seed)
        entities = layout.entities
        residuals: List[float] = []
        iterations = deadlocks = 0
        if self.use_optimizer and entities:
            tensors = scene_tensors(entities, scene, plan, graph, self.encoder, self.device)
            state, _ = self.optimizer.run(tensors)
            entities = apply_state(entities, state, scene.boundary)
            residuals, iterations, deadlocks = state.history, state.iterations, state.deadlocks
        return LayoutResult(entities=entities, traces=layout.traces, residuals=residuals,
                            iterations=iterations, deadlocks=deadlocks, graph=graph, plan=plan)


def _strip_entity_edges(graph: GeoGraph) -> GeoGraph:
    """Ablation: remove edges incident to target entities, leaving no graph conditioning."""
    entity_idx = set(graph.entity_indices())
    graph.edges = [(i, j, t) for i, j, t in graph.edges
                   if i not in entity_idx and j not in entity_idx]
    return graph


def _road_points(scene: Scene) -> np.ndarray:
    nodes = scene.roads.get("nodes", [])
    if not nodes:
        return np.zeros((0, 2))
    return np.stack([np.asarray(n["position"], dtype=float) for n in nodes])


def _nearest_road(position: np.ndarray, roads: np.ndarray, axes: Dict[str, List[float]]
                  ) -> Tuple[np.ndarray, float]:
    """Nearest point on the arterial grid and the bearing of that arterial."""
    if axes and (axes.get("x") or axes.get("y")):
        dx = min(((abs(position[0] - x), np.array([x, position[1]]), 0.0) for x in axes.get("x", [])),
                 default=(np.inf, position, 0.0))
        dy = min(((abs(position[1] - y), np.array([position[0], y]), np.pi / 2)
                  for y in axes.get("y", [])), default=(np.inf, position, 0.0))
        best = dx if dx[0] <= dy[0] else dy
        return best[1], best[2]
    if len(roads):
        j = int(np.argmin(np.linalg.norm(roads - position, axis=1)))
        return roads[j], 0.0
    return position, 0.0


def scene_tensors(entities: Sequence[Dict], scene: Scene, plan: Plan, graph: GeoGraph,
                  encoder: GATEncoder, device: torch.device,
                  required_m: float = DEFAULT_REQUIRED_M,
                  frontage_m: float = DEFAULT_FRONTAGE_M,
                  setback_m: float = DEFAULT_SETBACK_M) -> SceneTensors:
    """Pack one layout into the normalized tensors the optimizer consumes."""
    length = float(scene.boundary["length"])
    width = float(scene.boundary["width"])
    scale = np.array([length, width])
    ids = [e["id"] for e in entities]
    pos = np.stack([np.asarray(e["position"], dtype=float) for e in entities]) / scale
    size = np.stack([np.asarray(e["size"], dtype=float) for e in entities])
    size[:, :2] /= scale
    size[:, 2] /= 50.0
    theta = np.array([float(e.get("orientation") or 0.0) for e in entities])

    with torch.no_grad():
        x = torch.as_tensor(graph.features, dtype=torch.float32, device=device)
        ei = torch.as_tensor(graph.edge_index(), dtype=torch.long, device=device)
        node_emb, attention, edge_index = encoder(x, ei, return_attention=True), None, ei
        if isinstance(node_emb, tuple):
            node_emb, attention = node_emb
    rows = [graph.index(i) for i in ids]
    embeddings = node_emb[rows].detach()
    attn_per_entity = _mean_attention(attention, edge_index, rows, len(ids)) if attention else None

    relations = pair_relation(scene.relations)
    neighbours = neighbor_sets(ids, relations, positions=pos)
    index = {e: i for i, e in enumerate(ids)}
    pairs = [[index[a], index[b]] for a, nb in neighbours.items() for b in nb if a != b]
    related = [[index[a], index[b]] for (a, b) in relations if a in index and b in index]
    parents = [[index[e["id"]], index[e["parent"]]] for e in entities
               if e.get("parent") and e["parent"] in index]

    axes = scene.roads.get("axes", {})
    roads = _road_points(scene)
    road_points, bearings = [], []
    for e in entities:
        p, b = _nearest_road(np.asarray(e["position"], dtype=float), roads, axes)
        road_points.append(p / scale)
        bearings.append(b)
    frontage_mask = np.array([e["type"] in ("building", "poi") for e in entities])

    t = lambda a, dtype=torch.float32: torch.as_tensor(a, dtype=dtype, device=device)
    return SceneTensors(
        pos=t(pos), theta=t(theta), size=t(size), embeddings=embeddings,
        pairs=t(np.array(pairs, dtype=np.int64).reshape(-1, 2), torch.long),
        related_pairs=t(np.array(related, dtype=np.int64).reshape(-1, 2), torch.long),
        parent_pairs=t(np.array(parents, dtype=np.int64).reshape(-1, 2), torch.long),
        road_points=t(np.stack(road_points) if road_points else np.zeros((0, 2))),
        road_bearing=t(np.array(bearings)),
        frontage_mask=t(frontage_mask, torch.bool),
        extent=t(np.array([1.0, 1.0])),
        required=t(plan.constraints.get("proximity_distance_m", required_m) / length),
        admissible=t(frontage_m / length),
        setback=t(plan.constraints.get("buffer_distance_m", setback_m) / length),
        attention=attn_per_entity)


def _mean_attention(attention: Sequence[torch.Tensor], edge_index: torch.Tensor,
                    rows: Sequence[int], n: int) -> torch.Tensor:
    """Average attention received by each entity node, across heads and layers."""
    last = attention[-1].mean(dim=-1)
    dst = edge_index[1]
    totals = torch.zeros(int(dst.max()) + 1 if dst.numel() else 1, dtype=last.dtype)
    counts = torch.zeros_like(totals)
    totals.index_add_(0, dst, last)
    counts.index_add_(0, dst, torch.ones_like(last))
    mean = totals / counts.clamp(min=1)
    return mean[torch.as_tensor(rows, dtype=torch.long)]


def apply_state(entities: Sequence[Dict], state: OptimizerState,
                boundary: Dict[str, float]) -> List[Dict]:
    """Write optimizer output back into entity dictionaries, in metres."""
    scale = np.array([float(boundary["length"]), float(boundary["width"])])
    pos = state.pos.detach().cpu().numpy() * scale
    theta = state.theta.detach().cpu().numpy()
    height = state.size.detach().cpu().numpy()[:, 2] * 50.0
    out = []
    for e, p, th, h in zip(entities, pos, theta, height):
        e = dict(e)
        e["position"] = [float(p[0]), float(p[1])]
        e["orientation"] = float(th % (2 * np.pi))
        e["size"] = [float(e["size"][0]), float(e["size"][1]), float(h)]
        out.append(e)
    return out
