import numpy as np

from hgc_layout.metrics.evaluate import METRIC_NAMES, evaluate_layout
from hgc_layout.metrics.fidelity import area_ratio, category_match, entity_count, spatial_relationship
from hgc_layout.metrics.plausibility import (boundary_adherence, buffer_compliance, collision_free,
                                             green_accessibility, road_connectivity)
from hgc_layout.metrics.reliability import icc, rank_correlations, reliability_report, within_subject_sd
from hgc_layout.metrics.semantic import clip_similarity, layout_summary, semantic_plausibility
from hgc_layout.stats.tests import holm_bonferroni, paired_comparisons, wilcoxon_exact


def test_reference_layout_scores_near_ceiling(scenes):
    rows = [evaluate_layout(s.entities, s, 0.0) for s in scenes]
    means = {m: float(np.mean([r[m] for r in rows])) for m in METRIC_NAMES}
    assert set(means) == set(METRIC_NAMES)
    for metric in ("CNT", "CAT", "SPR", "ARE", "BND", "GAC"):
        assert means[metric] > 0.95
    assert means["COL"] > 0.9 and means["SP"] > 70


def test_fidelity_detects_missing_and_wrong_entities(scene):
    assert entity_count(scene.entities, scene.entities) == 1.0
    assert entity_count(scene.entities[:-2], scene.entities) < 1.0
    wrong = [dict(e, category="industrial") for e in scene.entities]
    assert category_match(wrong, scene.entities) < 1.0
    assert area_ratio(scene.entities, scene.area_ratio) > 0.95
    scattered = [dict(e, position=[10.0, 10.0]) for e in scene.entities]
    if scene.relations:
        assert spatial_relationship(scattered, scene.relations) <= 1.0


def test_spatial_relationship_matches_by_label():
    generated = [{"id": "x", "description": "shop", "position": [0.0, 0.0], "orientation": 0.0,
                  "type": "building", "category": "retail", "size": [10, 10, 5], "group": "g"},
                 {"id": "y", "description": "park", "position": [50.0, 0.0], "orientation": 0.0,
                  "type": "green", "category": "park", "size": [10, 10, 1], "group": "g"}]
    relation = [{"source": "e00", "target": "e01", "type": "adjacent_to",
                 "source_label": "shop", "target_label": "park"}]
    assert spatial_relationship(generated, relation) == 1.0


def test_plausibility_edges():
    boundary = {"length": 100.0, "width": 100.0}
    overlapping = [{"id": "a", "type": "building", "category": "retail", "description": "shop",
                    "size": [20, 20, 5], "group": "g", "position": [50, 50], "orientation": 0.0},
                   {"id": "b", "type": "building", "category": "retail", "description": "shop",
                    "size": [20, 20, 5], "group": "g", "position": [55, 50], "orientation": 0.0}]
    assert collision_free(overlapping) == 0.0
    assert boundary_adherence(overlapping, boundary) == 1.0
    outside = [dict(overlapping[0], position=[98, 98])]
    assert boundary_adherence(outside, boundary) == 0.0
    roads = {"nodes": [{"id": "r0", "position": [0, 0]}, {"id": "r1", "position": [0, 100]}],
             "edges": [["r0", "r1"]], "axes": {"x": [0.0], "y": []}}
    assert road_connectivity(overlapping, roads, snap=100.0) == 1.0
    assert road_connectivity(overlapping, roads, snap=1.0) == 0.0
    assert buffer_compliance(overlapping, roads, setback=18.0) == 1.0
    assert buffer_compliance([dict(overlapping[0], position=[5, 50])], roads, setback=18.0) == 0.0
    residential = [dict(overlapping[0], category="residential")]
    assert green_accessibility(residential) == 0.0


def test_semantic_backends(scene):
    summary = layout_summary(scene.entities, scene.boundary)
    assert summary
    sim = clip_similarity(scene.entities, scene.description, scene.boundary)
    assert 0.0 <= sim <= 1.0
    sp = semantic_plausibility(scene.entities, scene)
    assert sp["backend"] == "offline-rule-based"
    assert 0.0 <= sp["score"] <= 100.0


def test_icc_and_correlations():
    base = np.linspace(40, 90, 12)
    ratings = np.stack([base, base + 0.5, base - 0.5], axis=1)
    report = reliability_report(ratings)
    assert report["ICC(A,k)"] > 0.95 and report["ICC(C,k)"] > 0.95
    assert all(c > 0.95 for c in rank_correlations(ratings))
    assert within_subject_sd(ratings) < 1.0
    noise = np.random.default_rng(0).normal(60, 20, size=(12, 3))
    assert icc(noise)["ICC(C,k)"] < report["ICC(C,k)"]


def test_wilcoxon_and_holm():
    a = list(np.linspace(0.6, 0.9, 20))
    b = [v - 0.1 for v in a]
    w, p = wilcoxon_exact(a, b)
    assert p < 0.001 and w >= 0
    assert wilcoxon_exact(a, a) == (0.0, 1.0)
    adjusted = holm_bonferroni([0.001, 0.02, 0.5])
    assert adjusted[0] <= adjusted[1] <= adjusted[2]
    assert all(0 <= v <= 1 for v in adjusted)


def test_paired_comparisons_shape():
    rng = np.random.default_rng(0)
    scores = {"ours": {"SPR": list(rng.normal(0.85, 0.02, 20))},
              "a": {"SPR": list(rng.normal(0.60, 0.02, 20))},
              "b": {"SPR": list(rng.normal(0.55, 0.02, 20))}}
    rows = paired_comparisons(scores, "ours", ["a", "b"], ["SPR"])
    assert len(rows) == 2
    assert all("p_holm" in r and "significant" in r for r in rows)
    assert all(r["p_holm"] >= r["p_raw"] for r in rows)
