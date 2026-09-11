"""Asset retrieval score fusing text, visual and size-compatibility similarity.

The score combines a sentence-embedding similarity between the entity
description and the asset caption, a CLIP similarity between the entity
description and the asset image, and a size-compatibility term comparing the
asset footprint with the planned dimensions. When neither sentence encoder nor
CLIP is installed, both similarity terms fall back to the deterministic hashed
text embedding, so that ranking remains reproducible offline.
"""
from typing import Dict, List, Optional, Sequence

import numpy as np

from hgc_layout.data.features import _hash_embedding

TEXT_DIM = 384


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def size_compatibility(planned: Sequence[float], asset: Sequence[float]) -> float:
    """1 when footprints match, decaying with the relative dimension error."""
    p = np.asarray(planned, dtype=float)[:2]
    a = np.asarray(asset, dtype=float)[:2]
    rel = np.abs(p - a) / np.maximum(p, 1e-6)
    return float(np.exp(-rel.mean()))


def retrieval_score(entity: Dict, asset: Dict, weights: Sequence[float] = (0.4, 0.3, 0.3),
                    text_model: Optional[object] = None, clip_model: Optional[object] = None) -> float:
    """Weighted fusion of the three similarity terms."""
    text = f'{entity["category"]} {entity["description"]}'
    caption = asset.get("caption", asset.get("name", ""))
    if text_model is not None:  # pragma: no cover - optional dependency
        sbert = _cos(np.asarray(text_model.encode(text)), np.asarray(text_model.encode(caption)))
    else:
        sbert = _cos(_hash_embedding(text, TEXT_DIM, "sbert"),
                     _hash_embedding(caption, TEXT_DIM, "sbert"))
    if clip_model is not None and asset.get("image") is not None:  # pragma: no cover
        clip = _cos(np.asarray(clip_model.encode_text(text)),
                    np.asarray(clip_model.encode_image(asset["image"])))
    else:
        clip = _cos(_hash_embedding(text, TEXT_DIM, "clip"),
                    _hash_embedding(f'{caption} {asset.get("style", "")}', TEXT_DIM, "clip"))
    size = size_compatibility(entity["size"], asset.get("size", entity["size"]))
    w = np.asarray(weights, dtype=float)
    return float((w @ np.array([sbert, clip, size])) / w.sum())


def rank_assets(entity: Dict, assets: List[Dict], **kwargs) -> List[Dict]:
    """Assets sorted by decreasing retrieval score."""
    scored = [(retrieval_score(entity, a, **kwargs), a) for a in assets]
    return [a for _, a in sorted(scored, key=lambda t: -t[0])]
