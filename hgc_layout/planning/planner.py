"""Geospatial scene planning.

The planner turns a free-form description plus the K retrieved layout rules
into a structured plan: an entity set with types, categories, approximate
dimensions and functional groups; the scene boundary; and a global context
description. Two backends are provided. `LLMPlanner` queries a chat model
through an OpenAI-compatible endpoint and parses its JSON reply.
`TemplatePlanner` derives the same structure from the description and the
retrieved rules without any network access, and is the default so that the
pipeline is runnable offline.
"""
import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from hgc_layout.data.schema import Scene
from hgc_layout.data.synthetic import ENTITY_MENU, GROUPS_BY_CATEGORY, SIZE_BY_CATEGORY
from hgc_layout.planning.rules import RuleStore, merged_constraints

PLURAL = re.compile(r"(\d+)\s+([a-z][a-z \-]*?)(?:s)?(?=,| and | with |$)")

SYSTEM_PROMPT = (
    "You are a geospatial layout planner. Given a scene description and a list of "
    "layout rules, return JSON with keys 'entities' (list of objects with id, type, "
    "category, description, size [l,w,h] in metres, group), 'boundary' "
    "(object with length and width in metres) and 'global_context' (one sentence). "
    "Return JSON only, with no prose and no code fences."
)


@dataclass
class Plan:
    """Structured plan handed to graph construction and generation."""
    entities: List[Dict]
    boundary: Dict[str, float]
    global_context: str
    rules: List[Dict] = field(default_factory=list)
    constraints: Dict[str, float] = field(default_factory=dict)

    def groups(self) -> List[str]:
        seen: List[str] = []
        for e in self.entities:
            if e["group"] not in seen:
                seen.append(e["group"])
        return seen

    def entities_of_group(self, group: str) -> List[Dict]:
        return [e for e in self.entities if e["group"] == group]


class TemplatePlanner:
    """Offline planner: parses counted noun phrases and applies retrieved rules."""

    def __init__(self, store: RuleStore, k: int = 5):
        self.store = store
        self.k = k

    def plan(self, scene: Scene) -> Plan:
        rules = self.store.retrieve(scene.description, scene.category, self.k)
        constraints = merged_constraints(rules)
        counts = self._parse_counts(scene.description)
        groups = GROUPS_BY_CATEGORY.get(scene.category, ["commercial core"])
        entities: List[Dict] = []
        for label, n in counts.items():
            group, etype, ecat = self._lookup(label, groups)
            for _ in range(n):
                base = SIZE_BY_CATEGORY.get(ecat, (30.0, 25.0, 10.0))
                entities.append({"id": f"e{len(entities):02d}", "type": etype, "category": ecat,
                                 "description": label, "size": [float(v) for v in base],
                                 "group": group, "parent": None})
        if not entities:  # description without explicit counts
            for k, group in enumerate(groups):
                etype, ecat, label = ENTITY_MENU[group][0]
                entities.append({"id": f"e{k:02d}", "type": etype, "category": ecat,
                                 "description": label,
                                 "size": [float(v) for v in SIZE_BY_CATEGORY[ecat]],
                                 "group": group, "parent": None})
        context = f"{scene.category} ({scene.difficulty}) organised into {', '.join(groups)}"
        return Plan(entities=entities, boundary=dict(scene.boundary), global_context=context,
                    rules=rules, constraints=constraints)

    @staticmethod
    def _parse_counts(description: str) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for n, label in PLURAL.findall(description.lower()):
            label = label.strip()
            if label:
                counts[label] = counts.get(label, 0) + int(n)
        return counts

    @staticmethod
    def _lookup(label: str, groups: List[str]):
        for group in groups:
            for etype, ecat, name in ENTITY_MENU[group]:
                if name.startswith(label) or label.startswith(name):
                    return group, etype, ecat
        for group, menu in ENTITY_MENU.items():
            for etype, ecat, name in menu:
                if name.startswith(label) or label.startswith(name):
                    return (group if group in groups else groups[0]), etype, ecat
        return groups[0], "building", "retail"


class LLMPlanner:
    """Chat-model planner using an OpenAI-compatible endpoint.

    Requires the `openai` package and an API key in the environment. Falls back
    to the template planner when either is unavailable, so that callers do not
    have to branch.
    """

    def __init__(self, store: RuleStore, model: str = "gpt-4o-mini", k: int = 5,
                 temperature: float = 0.7, top_p: float = 1.0,
                 fallback: Optional[TemplatePlanner] = None):
        self.store = store
        self.model = model
        self.k = k
        self.temperature = temperature
        self.top_p = top_p
        self.fallback = fallback or TemplatePlanner(store, k)

    def available(self) -> bool:
        try:  # pragma: no cover - depends on optional dependency
            import openai  # noqa: F401
        except ImportError:
            return False
        return bool(os.environ.get("OPENAI_API_KEY"))

    def plan(self, scene: Scene) -> Plan:  # pragma: no cover - requires network
        if not self.available():
            return self.fallback.plan(scene)
        from openai import OpenAI

        rules = self.store.retrieve(scene.description, scene.category, self.k)
        user = json.dumps({"description": scene.description,
                           "boundary": scene.boundary,
                           "rules": [r["text"] for r in rules]}, ensure_ascii=False)
        client = OpenAI()
        reply = client.chat.completions.create(
            model=self.model, temperature=self.temperature, top_p=self.top_p,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": user}])
        text = reply.choices[0].message.content.strip()
        text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return self.fallback.plan(scene)
        entities = parsed.get("entities") or []
        for i, e in enumerate(entities):
            e.setdefault("id", f"e{i:02d}")
            e.setdefault("parent", None)
        return Plan(entities=entities,
                    boundary=parsed.get("boundary", dict(scene.boundary)),
                    global_context=parsed.get("global_context", scene.category),
                    rules=rules, constraints=merged_constraints(rules))


def build_planner(kind: str, rules_path, k: int = 5, model: str = "gpt-4o-mini"):
    """Factory: 'template' (offline, default) or 'llm'."""
    store = RuleStore.from_yaml(rules_path)
    if kind == "llm":
        return LLMPlanner(store, model=model, k=k)
    return TemplatePlanner(store, k)
