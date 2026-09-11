"""Node feature extraction: remote sensing, POI semantics, graph structure.

Three blocks are concatenated and projected to a common dimension d:
    f_rs    visual features of the remote-sensing patch (ResNet-18 when
            imagery and torchvision are available, otherwise the compact
            patch statistics stored with each scene)
    f_poi   semantic embedding of the category string (a sentence encoder
            when sentence-transformers is available, otherwise a
            deterministic hashed n-gram embedding)
    f_graph local structural properties of the node in the multi-source graph
"""
import hashlib
from typing import Dict, List, Optional, Sequence

import numpy as np

RS_DIM = 512
POI_DIM = 384
GRAPH_DIM = 16


def _hash_embedding(text: str, dim: int, seed: str = "") -> np.ndarray:
    """Deterministic embedding from character trigrams, unit normalised.

    Stands in for a learned sentence encoder when one is not installed; it is
    deterministic, order sensitive and shares no state across processes.
    """
    vec = np.zeros(dim, dtype=np.float32)
    padded = f"  {text.lower().strip()}  "
    for i in range(len(padded) - 2):
        gram = padded[i:i + 3] + seed
        h = int.from_bytes(hashlib.blake2b(gram.encode(), digest_size=8).digest(), "big")
        vec[h % dim] += 1.0 if (h >> 17) % 2 else -1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


class RemoteSensingEncoder:
    """Patch statistics expanded to `dim`; uses ResNet-18 when imagery is given."""

    def __init__(self, dim: int = RS_DIM, backbone: Optional[object] = None):
        self.dim = dim
        self.backbone = backbone

    def encode(self, stats: Sequence[float], key: str = "") -> np.ndarray:
        if self.backbone is not None:  # pragma: no cover - requires torchvision + imagery
            return np.asarray(self.backbone(stats), dtype=np.float32)[: self.dim]
        stats = np.asarray(stats, dtype=np.float32)
        basis = _hash_embedding(key or "patch", self.dim, seed="rs")
        tiled = np.resize(stats, self.dim).astype(np.float32)
        out = 0.7 * tiled + 0.3 * basis
        return out / (np.linalg.norm(out) + 1e-9)


class POIEncoder:
    """Category-string embedding; uses a sentence encoder when available."""

    def __init__(self, dim: int = POI_DIM, model: Optional[object] = None):
        self.dim = dim
        self.model = model

    def encode(self, category: str, description: str = "") -> np.ndarray:
        text = f"{category} {description}".strip()
        if self.model is not None:  # pragma: no cover - requires sentence-transformers
            v = np.asarray(self.model.encode(text), dtype=np.float32)
            return v / (np.linalg.norm(v) + 1e-9)
        return _hash_embedding(text, self.dim, seed="poi")


def structural_features(degree: int, n_neighbors_by_type: Dict[str, int],
                        clustering: float, dim: int = GRAPH_DIM) -> np.ndarray:
    """Degree, per-edge-type degree, clustering coefficient and their transforms."""
    d = float(degree)
    prox = float(n_neighbors_by_type.get("proximity", 0))
    func = float(n_neighbors_by_type.get("functional", 0))
    road = float(n_neighbors_by_type.get("road", 0))
    raw = [d, np.log1p(d), prox, func, road, clustering,
           prox / (d + 1e-6), func / (d + 1e-6), road / (d + 1e-6),
           float(d == 0), np.sqrt(d), clustering ** 2,
           np.log1p(prox), np.log1p(func), np.log1p(road), 1.0]
    return np.asarray(raw[:dim], dtype=np.float32)


def build_node_features(rs: np.ndarray, poi: np.ndarray, struct: np.ndarray) -> np.ndarray:
    """Concatenate the three blocks into one multi-modal feature vector."""
    return np.concatenate([rs, poi, struct]).astype(np.float32)


def feature_dim(rs_dim: int = RS_DIM, poi_dim: int = POI_DIM, graph_dim: int = GRAPH_DIM) -> int:
    return rs_dim + poi_dim + graph_dim
