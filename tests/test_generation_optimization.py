import math

import numpy as np
import torch

from hgc_layout.generation.decoder import GraphConditionedDecoder, causal_mask
from hgc_layout.generation.generator import LayoutGenerator, entity_attributes
from hgc_layout.generation.heads import GaussianMixtureHead, VonMisesHead
from hgc_layout.generation.retrieval import rank_assets, retrieval_score, size_compatibility
from hgc_layout.graph.construction import build_graph, feature_width
from hgc_layout.graph.gat import GATEncoder
from hgc_layout.optimization.constraints import CONSTRAINT_TYPES, LAMBDA_DEFAULT, neighbor_sets, pair_relation
from hgc_layout.optimization.forces import constraint_forces, pair_relation_vector, required_separation
from hgc_layout.optimization.optimizer import OptimizerState
from hgc_layout.optimization.violations import alignment_loss, violation_measures
from hgc_layout.optimization.weights import AdaptiveWeights, FixedWeights
from hgc_layout.pipeline import apply_state, scene_tensors


def test_mixture_head_shapes_and_density():
    head = GaussianMixtureHead(16, n_components=5)
    params = head(torch.randn(3, 16))
    assert params["mu"].shape == (3, 5, 2)
    assert torch.allclose(params["pi"].sum(-1), torch.ones(3), atol=1e-5)
    target = GaussianMixtureHead.mode(params)
    assert target.shape == (3, 2)
    near = GaussianMixtureHead.log_prob(params, target)
    far = GaussianMixtureHead.log_prob(params, target + 5.0)
    assert torch.all(near > far)


def test_von_mises_head_peaks_at_mode():
    head = VonMisesHead(16)
    params = head(torch.randn(4, 16))
    mode = VonMisesHead.mode(params)
    assert torch.all((mode >= 0) & (mode < 2 * math.pi))
    assert torch.all(VonMisesHead.log_prob(params, mode) >=
                     VonMisesHead.log_prob(params, mode + math.pi))


def test_decoder_records_cross_attention():
    decoder = GraphConditionedDecoder(embed_dim=32, d_model=32, n_layers=2, n_heads=4, ff_dim=64)
    query = torch.randn(1, 3, 32 + 8)
    memory = torch.randn(1, 7, 32)
    out = decoder(query, memory, torch.randn(1, 32), causal_mask(3))
    assert out["hidden"].shape == (1, 3, 32)
    assert len(decoder.cross_attention) == 2
    assert decoder.cross_attention[-1].shape == (1, 3, 7)
    assert torch.isneginf(causal_mask(3)[0, 1])


def test_generator_places_inside_boundary(planner, scene):
    plan = planner.plan(scene)
    graph = build_graph(scene, plan)
    encoder = GATEncoder(feature_width(), out_dim=32, hidden_dim=8)
    decoder = GraphConditionedDecoder(embed_dim=32, d_model=32, n_layers=2, n_heads=4, ff_dim=64)
    layout = LayoutGenerator(encoder, decoder).generate(graph, plan, seed=0)
    length, width = scene.size
    assert len(layout.entities) == len(plan.entities)
    for e in layout.entities:
        half = np.asarray(e["size"][:2]) / 2
        p = np.asarray(e["position"])
        assert np.all(p - half >= -1e-6) and np.all(p + half <= np.array([length, width]) + 1e-6)
    assert len(layout.traces) == len(plan.entities)
    assert layout.traces[0].top_k(3)


def test_generator_nll_is_differentiable(planner, scene):
    plan = planner.plan(scene)
    graph = build_graph(scene, plan)
    encoder = GATEncoder(feature_width(), out_dim=32, hidden_dim=8)
    decoder = GraphConditionedDecoder(embed_dim=32, d_model=32, n_layers=2, n_heads=4, ff_dim=64)
    targets = {e["id"]: (np.asarray(r["position"]), float(r["orientation"]))
               for e, r in zip(plan.entities, scene.entities)}
    loss = LayoutGenerator(encoder, decoder).nll(graph, plan, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None for p in decoder.parameters())


def test_entity_attributes_are_scale_free(scene):
    attrs = entity_attributes(scene.entities[0], scene.boundary)
    assert attrs.shape == (8,) and np.all(np.abs(attrs) <= 10)


def test_retrieval_prefers_matching_asset():
    entity = {"category": "retail", "description": "supermarket", "size": [30.0, 22.0, 8.0]}
    good = {"caption": "retail supermarket block", "size": [31.0, 21.0, 8.0]}
    bad = {"caption": "industrial chimney stack", "size": [90.0, 90.0, 40.0]}
    assert retrieval_score(entity, good) > retrieval_score(entity, bad)
    assert rank_assets(entity, [bad, good])[0] is good
    assert size_compatibility([30, 20], [30, 20]) == 1.0


def test_forces_push_apart_and_inside():
    pos_i = torch.tensor([[0.50, 0.50]])
    pos_j = torch.tensor([[0.51, 0.50]])
    size = torch.tensor([[0.05, 0.05, 0.2]])
    planar, vertical, _ = constraint_forces("collision", pos_i=pos_i, pos_j=pos_j,
                                            size_i=size, size_j=size)
    assert planar[0, 0] < 0 and vertical[0] > 0
    outside = torch.tensor([[-0.02, 0.50]])
    planar, _, _ = constraint_forces("boundary", pos_i=outside, size_i=size,
                                     extent=torch.tensor([1.0, 1.0]))
    assert planar[0, 0] > 0
    r = pair_relation_vector(pos_i, pos_j, size, size)
    assert r.shape == (1, 4)
    assert float(required_separation(size, size, torch.tensor(0.06))) > 0.06


def test_frontage_band_attracts_and_repels():
    pos = torch.tensor([[0.50, 0.50]])
    size = torch.tensor([[0.04, 0.03, 0.2]])
    far_road = torch.tensor([[0.90, 0.50]])
    near_road = torch.tensor([[0.505, 0.50]])
    theta = torch.tensor([0.0])
    admissible, setback = torch.tensor(0.16), torch.tensor(0.018)
    pull, _, _ = constraint_forces("frontage", pos_i=pos, theta_i=theta, road_point=far_road,
                                   road_bearing=torch.tensor([0.0]), admissible=admissible,
                                   setback=setback, size_i=size)
    push, _, _ = constraint_forces("frontage", pos_i=pos, theta_i=theta, road_point=near_road,
                                   road_bearing=torch.tensor([0.0]), admissible=admissible,
                                   setback=setback, size_i=size)
    assert pull[0, 0] > 0 and push[0, 0] < 0


def test_adaptive_weights_start_at_unit():
    weights = AdaptiveWeights(embed_dim=8)
    h = torch.randn(4, 8)
    r = torch.randn(4, 4)
    for c in CONSTRAINT_TYPES:
        w = weights(c, h, h, r)
        assert torch.allclose(w, torch.ones(4), atol=1e-5)
    distance_only = AdaptiveWeights(embed_dim=8, use_graph_embeddings=False)
    assert distance_only("collision", None, None, r).shape == (4,)
    assert torch.allclose(FixedWeights()("collision", h, h, r), torch.ones(4))


def test_optimizer_reduces_violations(model, scene):
    result = model.generate(scene, seed=0)
    tensors = scene_tensors(result.entities, scene, result.plan, result.graph,
                            model.encoder, model.device)
    state = OptimizerState(tensors.pos.clone(), tensors.theta.clone(), tensors.size.clone())
    before = violation_measures(state.pos, state.size, tensors.extent, tensors.pairs,
                                tensors.related_pairs, tensors.parent_pairs, tensors.road_points,
                                tensors.admissible, tensors.frontage_mask, tensors.required,
                                tensors.setback)
    final, weights = model.optimizer.run(tensors, state)
    after = violation_measures(final.pos, final.size, tensors.extent, tensors.pairs,
                               tensors.related_pairs, tensors.parent_pairs, tensors.road_points,
                               tensors.admissible, tensors.frontage_mask, tensors.required,
                               tensors.setback)
    assert sum(float(v) for v in after.values()) <= sum(float(v) for v in before.values()) + 1e-6
    loss = alignment_loss(after, weights, LAMBDA_DEFAULT, beta=0.01)
    assert torch.isfinite(loss)
    assert final.iterations > 0 and len(final.history) == final.iterations


def test_apply_state_returns_metres(model, scene):
    result = model.generate(scene, seed=0)
    tensors = scene_tensors(result.entities, scene, result.plan, result.graph,
                            model.encoder, model.device)
    state = OptimizerState(tensors.pos, tensors.theta, tensors.size)
    entities = apply_state(result.entities, state, scene.boundary)
    for original, updated in zip(result.entities, entities):
        assert np.allclose(original["position"], updated["position"], atol=1e-3)


def test_neighbor_sets_and_relations(scene):
    relations = pair_relation(scene.relations)
    ids = [e["id"] for e in scene.entities]
    positions = np.stack([np.asarray(e["position"]) for e in scene.entities])
    neighbours = neighbor_sets(ids, relations, positions=positions)
    assert set(neighbours) == set(ids)
    assert all(i not in neighbours[i] for i in ids)
