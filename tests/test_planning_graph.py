import numpy as np
import torch

from hgc_layout.graph.augment import augment_view
from hgc_layout.graph.construction import EDGE_TYPES, block_mask, build_graph, feature_width
from hgc_layout.graph.contrastive import embed_graph, nt_xent_loss, pretrain_encoder
from hgc_layout.graph.gat import GATEncoder, _edge_softmax
from hgc_layout.planning.rules import RuleStore, merged_constraints


def test_rule_retrieval_respects_category(rules_path):
    store = RuleStore.from_yaml(rules_path)
    rules = store.retrieve("residential blocks near a park", "residential area", k=5)
    assert len(rules) == 5
    assert all("all" in r["applies_to"] or "residential area" in r["applies_to"] for r in rules)
    assert "buffer_distance_m" in merged_constraints(rules)


def test_planner_recovers_counted_entities(planner, scenes):
    for scene in scenes[:4]:
        plan = planner.plan(scene)
        assert len(plan.entities) == len(scene.entities)
        assert plan.boundary == scene.boundary


def test_graph_hides_target_positions(planner, scene):
    plan = planner.plan(scene)
    graph = build_graph(scene, plan)
    entity_rows = graph.entity_indices()
    assert len(entity_rows) == len(plan.entities)
    assert np.isnan(graph.positions[entity_rows]).all()
    assert np.isfinite(graph.positions[[i for i in range(graph.n_nodes)
                                        if i not in entity_rows]]).all()
    assert graph.features.shape == (graph.n_nodes, feature_width())
    assert {t for _, _, t in graph.edges} <= set(EDGE_TYPES)


def test_add_placement_grows_partial_graph(planner, scene):
    plan = planner.plan(scene)
    graph = build_graph(scene, plan)
    before = len(graph.edges)
    graph.add_placement(plan.entities[0]["id"], np.array([500.0, 500.0]))
    assert len(graph.edges) >= before
    assert np.isfinite(graph.positions[graph.index(plan.entities[0]["id"])]).all()


def test_feature_blocks_and_node_switches(planner, scene):
    plan = planner.plan(scene)
    rs_only = build_graph(scene, plan, feature_blocks=["rs"])
    assert np.allclose(rs_only.features[:, 512:], 0.0)
    assert not np.allclose(rs_only.features[:, :512], 0.0)
    no_poi = build_graph(scene, plan, use_poi_nodes=False)
    assert "poi" not in no_poi.node_kind
    no_roads = build_graph(scene, plan, use_road_edges=False)
    assert all(t != "road" for _, _, t in no_roads.edges)
    assert block_mask(None).sum() == feature_width()


def test_augmentation_changes_views(planner, scene):
    plan = planner.plan(scene)
    graph = build_graph(scene, plan)
    rng = np.random.default_rng(0)
    view = augment_view(graph, rng)
    assert view.features.shape == graph.features.shape
    assert not np.allclose(view.features, graph.features)


def test_edge_softmax_normalises():
    logits = torch.randn(6, 2)
    dst = torch.tensor([0, 0, 1, 1, 1, 2])
    alpha = _edge_softmax(logits, dst, 3)
    for node in range(3):
        assert torch.allclose(alpha[dst == node].sum(0), torch.ones(2), atol=1e-5)


def test_contrastive_loss_and_pretraining(planner, scenes):
    plans = [(s, planner.plan(s)) for s in scenes[:3]]
    graphs = [build_graph(s, p) for s, p in plans]
    encoder = GATEncoder(feature_width(), out_dim=32, hidden_dim=8)
    z = torch.randn(5, 32)
    assert float(nt_xent_loss(z, z, 0.5)) < float(nt_xent_loss(z, torch.randn(5, 32), 0.5))
    history = pretrain_encoder(graphs, encoder, epochs=2, batch_size=4, log_every=0)
    assert len(history["loss"]) == 2
    h, attention, edge_index = embed_graph(graphs[0], encoder, return_attention=True)
    assert h.shape == (graphs[0].n_nodes, 32)
    assert len(attention) == 2 and attention[0].shape[0] == edge_index.shape[1]
