"""Fidelity metrics: how well a generated scene matches its description.

CNT entity count, CAT category match, SPR spatial-relationship compliance and
ARE area ratio, each normalized to [0, 1] with higher being better.
"""
from typing import Dict, List, Optional, Sequence

import numpy as np

ADJACENT_M = 160.0
NEAR_M = 320.0
FACING_TOLERANCE = np.pi / 4


def _counts(entities: Sequence[Dict], key: str = "description") -> Dict[str, int]:
    out: Dict[str, int] = {}
    for e in entities:
        out[e[key]] = out.get(e[key], 0) + 1
    return out


def entity_count(generated: Sequence[Dict], reference: Sequence[Dict]) -> float:
    """1 minus the normalized absolute count error over entity labels."""
    gen, ref = _counts(generated), _counts(reference)
    labels = set(gen) | set(ref)
    if not labels:
        return 1.0
    total = sum(ref.get(l, 0) for l in labels) or 1
    error = sum(abs(gen.get(l, 0) - ref.get(l, 0)) for l in labels)
    return float(np.clip(1.0 - error / total, 0.0, 1.0))


def category_match(generated: Sequence[Dict], reference: Sequence[Dict]) -> float:
    """Share of required (label, category) pairs present in the generated set."""
    ref = _counts(reference)
    gen: Dict[str, List[str]] = {}
    for e in generated:
        gen.setdefault(e["description"], []).append(e["category"])
    ref_cat = {e["description"]: e["category"] for e in reference}
    matched = total = 0
    for label, n in ref.items():
        total += n
        produced = gen.get(label, [])
        matched += sum(1 for c in produced[:n] if c == ref_cat[label])
    return float(matched / total) if total else 1.0


def _resolve(rel: Dict, key: str, pos: Dict[str, np.ndarray],
             by_label: Dict[str, List[str]]) -> Optional[str]:
    """Endpoint id in the generated layout, by id when present, else by label.

    Regenerated layouts may carry different identifiers than the reference, so
    an endpoint also resolves through the entity label; labels are consumed in
    order, which keeps the assignment one-to-one when a label repeats.
    """
    eid = rel.get(key)
    if eid in pos:
        return eid
    label = rel.get(f"{key}_label")
    candidates = by_label.get(label, [])
    return candidates.pop(0) if candidates else None


def _satisfied(rel: Dict, pos: Dict[str, np.ndarray], theta: Dict[str, float],
               by_label: Dict[str, List[str]]) -> bool:
    a = _resolve(rel, "source", pos, by_label)
    b = _resolve(rel, "target", pos, by_label)
    if a is None or b is None:
        return False
    delta = pos[b] - pos[a]
    d = float(np.linalg.norm(delta))
    if rel["type"] == "adjacent_to":
        return d <= ADJACENT_M
    if rel["type"] == "near":
        return d <= NEAR_M
    if rel["type"] == "surrounding":
        return d <= NEAR_M
    if rel["type"] == "aligned_with":
        return abs(np.sin(theta.get(a, 0.0) - theta.get(b, 0.0))) < np.sin(FACING_TOLERANCE)
    if rel["type"] == "facing":
        bearing = float(np.arctan2(delta[1], delta[0])) % (2 * np.pi)
        diff = abs((bearing - theta.get(a, 0.0) + np.pi) % (2 * np.pi) - np.pi)
        return diff <= FACING_TOLERANCE
    return False


def spatial_relationship(generated: Sequence[Dict], relations: Sequence[Dict]) -> float:
    """Share of described pairwise relations that the layout satisfies."""
    if not relations:
        return 1.0
    pos = {e["id"]: np.asarray(e["position"], dtype=float) for e in generated}
    theta = {e["id"]: float(e.get("orientation") or 0.0) for e in generated}
    satisfied = []
    for r in relations:
        by_label: Dict[str, List[str]] = {}
        for e in generated:
            by_label.setdefault(e["description"], []).append(e["id"])
        satisfied.append(_satisfied(r, pos, theta, by_label))
    return float(np.mean(satisfied))


def area_ratio(generated: Sequence[Dict], reference_ratio: Dict[str, float]) -> float:
    """1 minus the mean relative deviation of functional-zone area shares."""
    if not reference_ratio:
        return 1.0
    areas: Dict[str, float] = {}
    for e in generated:
        areas[e["group"]] = areas.get(e["group"], 0.0) + float(e["size"][0]) * float(e["size"][1])
    total = sum(areas.values()) or 1.0
    errors = [abs(areas.get(g, 0.0) / total - share) / max(share, 1e-6)
              for g, share in reference_ratio.items()]
    return float(np.clip(1.0 - np.mean(errors), 0.0, 1.0))
