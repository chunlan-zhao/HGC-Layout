"""Construction of the framework and its variants from a configuration."""
from typing import Dict, Optional, Sequence

import torch

from hgc_layout.generation.decoder import GraphConditionedDecoder
from hgc_layout.graph.construction import feature_width
from hgc_layout.graph.gat import GATEncoder
from hgc_layout.optimization.optimizer import AdaptiveOptimizer
from hgc_layout.optimization.weights import AdaptiveWeights, FixedWeights
from hgc_layout.pipeline import HGCLayout
from hgc_layout.planning.planner import build_planner


def build_planner_from_config(cfg) -> object:
    planner_cfg = cfg.get("planner", {})
    return build_planner(planner_cfg.get("kind", "template"), cfg["data"]["rules"],
                         k=planner_cfg.get("top_k_rules", 5),
                         model=planner_cfg.get("model", "gpt-4o-mini"))


def build_model(cfg, planner=None, device: Optional[torch.device] = None,
                use_adaptive_weights: bool = True, **overrides) -> HGCLayout:
    """Instantiate the framework with the hyper-parameters given in `cfg`."""
    device = device or torch.device("cpu")
    planner = planner or build_planner_from_config(cfg)
    g, gen, opt = cfg["graph"], cfg["generator"], cfg["optimizer"]
    encoder = GATEncoder(feature_width(), hidden_dim=g["hidden_dim"], out_dim=g["embed_dim"],
                         heads=g["heads"], n_layers=g["layers"])
    decoder = GraphConditionedDecoder(embed_dim=g["embed_dim"], d_model=gen["d_model"],
                                      n_layers=gen["layers"], n_heads=gen["heads"],
                                      ff_dim=gen["ff_dim"], dropout=gen["dropout"],
                                      n_components=gen["mixture_components"])
    weights = AdaptiveWeights(g["embed_dim"]) if use_adaptive_weights else FixedWeights()
    optimizer = AdaptiveOptimizer(weights, dict(opt["lambdas"]), step_size=opt["step_size"],
                                  max_iter=opt["max_iter"], tol=opt["tol"],
                                  deadlock_cumulative=opt["deadlock_cumulative"],
                                  deadlock_net=opt["deadlock_net"], seed=cfg["seed"])
    return HGCLayout(planner, encoder=encoder, decoder=decoder, weights=weights,
                     optimizer=optimizer, delta=g["delta_m"], embed_dim=g["embed_dim"],
                     use_adaptive_weights=use_adaptive_weights, device=device, **overrides)


def core_variants(cfg, model: HGCLayout) -> Dict[str, HGCLayout]:
    """The three core-component ablations, sharing the trained modules."""
    from hgc_layout.experiments import _clone

    no_graph = _clone(model)
    no_graph.use_graph_conditioning = False

    fixed = _clone(model)
    fixed.weights = FixedWeights()
    fixed.optimizer = AdaptiveOptimizer(fixed.weights, model.optimizer.lambdas,
                                        step_size=model.optimizer.step_size,
                                        max_iter=model.optimizer.max_iter)
    return {"w/o graph-conditioned gen.": no_graph, "w/o adaptive force weights": fixed}


def module_variants(cfg, model: HGCLayout) -> Dict[str, HGCLayout]:
    """Individual-module ablations of the planning and graph stages."""
    from hgc_layout.experiments import _clone

    variants = {}
    for label, field in (("w/o rule retrieval", "use_rule_retrieval"),
                         ("w/o global context", "use_global_context"),
                         ("w/o hierarchical grouping", "use_hierarchical_groups"),
                         ("w/o POI nodes", "use_poi_nodes"),
                         ("w/o road edges", "use_road_edges")):
        variant = _clone(model)
        setattr(variant, field, False)
        variants[label] = variant
    return variants
