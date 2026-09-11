"""Graph-conditioned hierarchical layout generation.

Generation proceeds group by group, and inside a group entity by entity. Before
each placement the partial scene graph holds the auxiliary nodes and the
entities already placed, with their model-predicted coordinates. The decoder
cross-attends to the node embeddings of that partial graph and emits the
parameters of Eq. (3); the placed coordinate is written back into the graph so
that later placements condition on updated proximity relations. Reference
coordinates are never read at inference.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from hgc_layout.generation.decoder import GraphConditionedDecoder, causal_mask
from hgc_layout.generation.heads import GaussianMixtureHead, VonMisesHead
from hgc_layout.graph.construction import GeoGraph
from hgc_layout.graph.gat import GATEncoder
from hgc_layout.planning.planner import Plan

N_ATTRIBUTES = 8


@dataclass
class PlacementTrace:
    """Cross-attention weights recorded for one placement."""
    entity_id: str
    neighbors: List[str]
    weights: List[float]

    def top_k(self, k: int = 8) -> List[Tuple[str, float]]:
        order = np.argsort(-np.asarray(self.weights))[:k]
        return [(self.neighbors[i], float(self.weights[i])) for i in order]


@dataclass
class GeneratedLayout:
    """Placed entities plus the attention traces that produced them."""
    entities: List[Dict]
    traces: List[PlacementTrace] = field(default_factory=list)

    def positions(self) -> Dict[str, np.ndarray]:
        return {e["id"]: np.asarray(e["position"], dtype=float) for e in self.entities}


def entity_attributes(entity: Dict, boundary: Dict[str, float]) -> np.ndarray:
    """Size, aspect, area share and one-hot-ish type code, all scale free."""
    length, width = float(boundary["length"]), float(boundary["width"])
    sx, sy, sz = (float(v) for v in entity["size"])
    type_code = {"building": 0.2, "road": 0.4, "green": 0.6, "water": 0.8, "poi": 1.0}
    return np.array([sx / length, sy / width, sz / 50.0, sx / max(sy, 1e-6) / 10.0,
                     (sx * sy) / (length * width), type_code.get(entity["type"], 0.0),
                     float(entity.get("parent") is not None), 1.0], dtype=np.float32)


class LayoutGenerator:
    """Autoregressive placement over the partial scene graph."""

    def __init__(self, encoder: GATEncoder, decoder: GraphConditionedDecoder,
                 device: Optional[torch.device] = None, sample: bool = False):
        self.encoder = encoder
        self.decoder = decoder
        self.device = device or torch.device("cpu")
        self.sample = sample

    def _embed(self, graph: GeoGraph) -> torch.Tensor:
        x = torch.as_tensor(graph.features, dtype=torch.float32, device=self.device)
        ei = torch.as_tensor(graph.edge_index(), dtype=torch.long, device=self.device)
        return self.encoder(x, ei)

    @torch.no_grad()
    def generate(self, graph: GeoGraph, plan: Plan, roads: Optional[np.ndarray] = None,
                 seed: int = 0) -> GeneratedLayout:
        """Place every planned entity, group by group, updating the partial graph."""
        self.encoder.eval()
        self.decoder.eval()
        gen = torch.Generator().manual_seed(seed)
        length, width = float(plan.boundary["length"]), float(plan.boundary["width"])
        placed: List[Dict] = []
        traces: List[PlacementTrace] = []
        for group in plan.groups():
            group_entities = plan.entities_of_group(group)
            tokens: List[torch.Tensor] = []
            for entity in group_entities:
                embeddings = self._embed(graph)
                node = embeddings[graph.index(entity["id"])]
                attrs = torch.as_tensor(entity_attributes(entity, plan.boundary), device=self.device)
                tokens.append(torch.cat([node, attrs]))
                query = torch.stack(tokens).unsqueeze(0)
                context = embeddings.mean(dim=0, keepdim=True)
                out = self.decoder(query, embeddings.unsqueeze(0), context,
                                   causal_mask(query.size(1), self.device))
                params = {k: v[-1:].contiguous() for k, v in out["coord"].items()}
                unit = (GaussianMixtureHead.sample(params, gen) if self.sample
                        else GaussianMixtureHead.mode(params))
                unit = unit.clamp(0.0, 1.0)
                hidden = out["hidden"][:, -1]
                orient = self.decoder.orientation(hidden, unit)
                theta = float(VonMisesHead.mode(orient).item()) % (2 * np.pi)
                position = np.array([float(unit[0, 0]) * length, float(unit[0, 1]) * width])
                position = _clip_to_boundary(position, entity["size"], length, width)
                entity = dict(entity)
                entity["position"] = [float(position[0]), float(position[1])]
                entity["orientation"] = theta
                placed.append(entity)
                traces.append(self._trace(graph, entity["id"]))
                graph.add_placement(entity["id"], position, roads)
        return GeneratedLayout(entities=placed, traces=traces)

    def _trace(self, graph: GeoGraph, entity_id: str) -> PlacementTrace:
        """Cross-attention of the last layer for the current query token."""
        layers = self.decoder.cross_attention
        if not layers:
            return PlacementTrace(entity_id, [], [])
        weights = layers[-1][0, -1].cpu().numpy()
        return PlacementTrace(entity_id=entity_id, neighbors=list(graph.node_ids),
                              weights=[float(w) for w in weights])

    def nll(self, graph: GeoGraph, plan: Plan, targets: Dict[str, Tuple[np.ndarray, float]]
            ) -> torch.Tensor:
        """Teacher-forced negative log-likelihood on a training scene.

        Reference coordinates are used here, and only here: training supervises
        the decoder with the reference layout, inference never reads it.
        """
        length, width = float(plan.boundary["length"]), float(plan.boundary["width"])
        embeddings = self._embed(graph)
        context = embeddings.mean(dim=0, keepdim=True)
        losses = []
        for group in plan.groups():
            entities = [e for e in plan.entities_of_group(group) if e["id"] in targets]
            if not entities:
                continue
            tokens = []
            for e in entities:
                node = embeddings[graph.index(e["id"])]
                attrs = torch.as_tensor(entity_attributes(e, plan.boundary), device=self.device)
                tokens.append(torch.cat([node, attrs]))
            query = torch.stack(tokens).unsqueeze(0)
            out = self.decoder(query, embeddings.unsqueeze(0), context,
                               causal_mask(query.size(1), self.device))
            target_xy = torch.as_tensor(
                np.stack([targets[e["id"]][0] for e in entities]) / np.array([length, width]),
                dtype=torch.float32, device=self.device).clamp(0.0, 1.0)
            target_th = torch.as_tensor([targets[e["id"]][1] for e in entities],
                                        dtype=torch.float32, device=self.device)
            coord_ll = GaussianMixtureHead.log_prob(out["coord"], target_xy)
            orient = self.decoder.orientation(out["hidden"][0], target_xy)
            orient_ll = VonMisesHead.log_prob(orient, target_th)
            losses.append(-(coord_ll.mean() + orient_ll.mean()))
        if not losses:
            return torch.zeros((), device=self.device, requires_grad=True)
        return torch.stack(losses).mean()


def _clip_to_boundary(position: np.ndarray, size, length: float, width: float) -> np.ndarray:
    """Keep the footprint fully inside the scene boundary."""
    half = np.array([float(size[0]), float(size[1])]) / 2
    return np.clip(position, half, np.array([length, width]) - half)
