"""Semantic alignment metrics: CLIP similarity and the Semantic Plausibility score.

CLIPsim compares a top-down render of the generated layout with the text
prompt. When a CLIP checkpoint is supplied, image and text are encoded with it;
otherwise both sides are embedded with a deterministic offline encoder that
reads the same layout summary, so the metric stays reproducible without
network access.

SP is a 0-100 judgment of functional coherence and realism. With an
OpenAI-compatible endpoint configured, it is produced by a chat model; the
offline fallback is a rule-based score that must not be reported as the
model judgment, and `semantic_plausibility` marks which backend produced it.
"""
import json
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from hgc_layout.data.features import _hash_embedding
from hgc_layout.metrics.plausibility import (boundary_adherence, buffer_compliance, collision_free,
                                             green_accessibility, road_connectivity)

EMB_DIM = 512

SP_PROMPT = (
    "Score the functional coherence and realism of the urban layout below from 0 to 100. "
    "Consider whether the entity mix, their spatial arrangement and their relation to the "
    "road network would be plausible in a real district. Reply with JSON of the form "
    '{"score": <number>} and nothing else.'
)


def layout_summary(entities: Sequence[Dict], boundary: Dict[str, float]) -> str:
    """Compact textual render of a layout, used as the image stand-in."""
    length = float(boundary["length"])
    cells: Dict[Tuple[int, int], List[str]] = {}
    for e in entities:
        gx = int(np.clip(float(e["position"][0]) / length * 4, 0, 3))
        gy = int(np.clip(float(e["position"][1]) / length * 4, 0, 3))
        cells.setdefault((gx, gy), []).append(e["description"])
    parts = [f"({x},{y}): " + ", ".join(sorted(v)) for (x, y), v in sorted(cells.items())]
    return " | ".join(parts)


def clip_similarity(entities: Sequence[Dict], description: str, boundary: Dict[str, float],
                    clip_model: Optional[object] = None) -> float:
    """Cosine similarity between the layout render and the prompt, scaled to [0, 1]."""
    summary = layout_summary(entities, boundary)
    if clip_model is not None:  # pragma: no cover - optional dependency
        img = np.asarray(clip_model.encode_image(summary), dtype=float)
        txt = np.asarray(clip_model.encode_text(description), dtype=float)
    else:
        img = _hash_embedding(summary, EMB_DIM, "clip-image")
        txt = _hash_embedding(description, EMB_DIM, "clip-text")
    cos = float(img @ txt / (np.linalg.norm(img) * np.linalg.norm(txt) + 1e-9))
    return float(np.clip(0.5 * (cos + 1.0), 0.0, 1.0))


def _offline_sp(entities: Sequence[Dict], scene) -> float:
    """Rule-based stand-in for the model judgment, on the same 0-100 scale."""
    terms = [collision_free(entities),
             boundary_adherence(entities, scene.boundary),
             buffer_compliance(entities, scene.roads),
             green_accessibility(entities),
             road_connectivity(entities, scene.roads)]
    spread = _spatial_spread(entities, scene.boundary)
    return float(np.clip(100.0 * (0.82 * float(np.mean(terms)) + 0.18 * spread), 0.0, 100.0))


def _spatial_spread(entities: Sequence[Dict], boundary: Dict[str, float]) -> float:
    """How evenly entities cover the region; degenerate piles score near zero."""
    if len(entities) < 2:
        return 1.0
    pos = np.stack([np.asarray(e["position"], dtype=float) for e in entities])
    extent = np.array([float(boundary["length"]), float(boundary["width"])])
    return float(np.clip(pos.std(axis=0).mean() / (0.28 * extent.mean()), 0.0, 1.0))


def semantic_plausibility(entities: Sequence[Dict], scene, model: Optional[str] = None,
                          temperature: float = 1.0, top_p: float = 1.0,
                          n_runs: int = 1) -> Dict[str, object]:
    """Return the SP score, the backend that produced it and the per-run scores."""
    if model and _openai_available():  # pragma: no cover - requires network
        scores = [_llm_sp(entities, scene, model, temperature, top_p) for _ in range(n_runs)]
        scores = [s for s in scores if s is not None]
        if scores:
            return {"score": float(np.mean(scores)), "backend": model, "runs": scores}
    scores = [_offline_sp(entities, scene)] * max(n_runs, 1)
    return {"score": float(np.mean(scores)), "backend": "offline-rule-based", "runs": scores}


def _openai_available() -> bool:
    try:  # pragma: no cover - optional dependency
        import openai  # noqa: F401
    except ImportError:
        return False
    return bool(os.environ.get("OPENAI_API_KEY"))


def _llm_sp(entities, scene, model, temperature, top_p) -> Optional[float]:  # pragma: no cover
    from openai import OpenAI

    payload = json.dumps({"description": scene.description,
                          "boundary": scene.boundary,
                          "layout": [{"id": e["id"], "label": e["description"],
                                      "position": e["position"],
                                      "orientation": e.get("orientation")} for e in entities]},
                         ensure_ascii=False)
    client = OpenAI()
    reply = client.chat.completions.create(
        model=model, temperature=temperature, top_p=top_p,
        messages=[{"role": "system", "content": SP_PROMPT}, {"role": "user", "content": payload}])
    text = reply.choices[0].message.content
    match = re.search(r'"score"\s*:\s*([0-9.]+)', text or "")
    return float(match.group(1)) if match else None
