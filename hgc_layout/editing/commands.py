"""Scene editing by command routing.

An add or move command updates the graph by inserting or relocating the
corresponding node; the graph-conditioned generator re-predicts the affected
coordinates and the adaptive optimizer re-balances the forces, so that global
coherence is preserved without regenerating the whole scene. A delete command
removes the node and only re-runs the optimizer.
"""
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

ADD = re.compile(r"^add\s+(?:a\s+|an\s+)?(.+)$", re.I)
DELETE = re.compile(r"^(?:delete|remove)\s+(.+)$", re.I)
MOVE = re.compile(r"^move\s+(.+?)\s+(?:to|toward|towards)\s+(.+)$", re.I)

DIRECTIONS = {"north": (0.0, 1.0), "south": (0.0, -1.0), "east": (1.0, 0.0), "west": (-1.0, 0.0),
              "centre": (0.0, 0.0), "center": (0.0, 0.0)}


@dataclass
class EditCommand:
    """Parsed editing instruction."""
    operation: str                # add | delete | move
    target: str
    argument: Optional[str] = None


def parse_command(text: str) -> EditCommand:
    """Parse one natural-language editing instruction."""
    text = text.strip()
    m = MOVE.match(text)
    if m:
        return EditCommand("move", m.group(1).strip(), m.group(2).strip())
    m = DELETE.match(text)
    if m:
        return EditCommand("delete", m.group(1).strip())
    m = ADD.match(text)
    if m:
        return EditCommand("add", m.group(1).strip())
    raise ValueError(f"Unrecognised editing command: {text!r}")


def _match(entities: Sequence[Dict], target: str) -> Optional[int]:
    target = target.lower()
    for i, e in enumerate(entities):
        if target in e["description"].lower() or target == e["id"].lower():
            return i
    return None


def apply_edit(model, scene, entities: Sequence[Dict], command: EditCommand,
               seed: int = 0) -> Tuple[List[Dict], float]:
    """Apply one command and return the updated layout and the elapsed seconds."""
    from hgc_layout.pipeline import apply_state, scene_tensors
    from hgc_layout.graph.construction import build_graph

    start = time.perf_counter()
    entities = [dict(e) for e in entities]
    plan = model.plan_scene(scene)
    extent = np.array([float(scene.boundary["length"]), float(scene.boundary["width"])])

    if command.operation == "delete":
        idx = _match(entities, command.target)
        if idx is not None:
            removed = entities.pop(idx)
            plan.entities = [e for e in plan.entities if e["id"] != removed["id"]]
    elif command.operation == "move":
        idx = _match(entities, command.target)
        if idx is not None:
            direction = np.array(DIRECTIONS.get(command.argument.lower(), (0.0, 0.0)))
            target = np.asarray(entities[idx]["position"], dtype=float) + direction * 0.2 * extent
            if not direction.any():
                target = extent / 2
            entities[idx]["position"] = [float(v) for v in np.clip(target, 0, extent)]
    elif command.operation == "add":
        template = plan.entities[0] if plan.entities else None
        if template is not None:
            new = dict(template)
            new["id"] = f"e{len(entities):02d}_new"
            new["description"] = command.target
            rng = np.random.default_rng(seed)
            new["position"] = [float(v) for v in rng.uniform(0.2, 0.8, 2) * extent]
            new["orientation"] = 0.0
            entities.append(new)
            plan.entities.append({k: v for k, v in new.items()
                                  if k not in ("position", "orientation")})

    graph = build_graph(scene, plan, delta=model.delta)
    for e in entities:                                  # re-insert placed nodes into the graph
        if e["id"] in graph.node_ids:
            graph.add_placement(e["id"], np.asarray(e["position"], dtype=float))
    if model.use_optimizer and entities:
        tensors = scene_tensors(entities, scene, plan, graph, model.encoder, model.device)
        state, _ = model.optimizer.run(tensors)
        entities = apply_state(entities, state, scene.boundary)
    return entities, time.perf_counter() - start
