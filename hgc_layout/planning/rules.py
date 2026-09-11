"""Vector store over geospatial layout rules with top-K retrieval."""
from pathlib import Path
from typing import Dict, List

import numpy as np
import yaml

from hgc_layout.data.features import _hash_embedding

RULE_DIM = 128


class RuleStore:
    """Embeds rule keywords once and retrieves the K nearest rules per query."""

    def __init__(self, rules: List[Dict]):
        self.rules = rules
        self._emb = np.stack([self._embed(" ".join(r["keywords"]) + " " + r["text"])
                              for r in rules])

    @staticmethod
    def _embed(text: str) -> np.ndarray:
        return _hash_embedding(text, RULE_DIM, seed="rule")

    @classmethod
    def from_yaml(cls, path) -> "RuleStore":
        with Path(path).open("r", encoding="utf-8") as fh:
            return cls(yaml.safe_load(fh))

    def retrieve(self, query: str, category: str = None, k: int = 5) -> List[Dict]:
        """Cosine-similarity retrieval, restricted to rules valid for `category`."""
        mask = np.array([("all" in r["applies_to"]) or (category in r["applies_to"])
                         for r in self.rules]) if category else np.ones(len(self.rules), bool)
        sims = self._emb @ self._embed(query)
        sims = np.where(mask, sims, -np.inf)
        order = np.argsort(-sims)[:k]
        return [self.rules[i] for i in order if np.isfinite(sims[i])]


def retrieve_rules(store: RuleStore, description: str, category: str, k: int = 5) -> List[Dict]:
    return store.retrieve(description, category, k)


def merged_constraints(rules: List[Dict]) -> Dict[str, float]:
    """Collapse the constraint dictionaries of the retrieved rules."""
    out: Dict[str, float] = {}
    for r in rules:
        for key, value in (r.get("constraints") or {}).items():
            out[key] = float(value) if key not in out else max(out[key], float(value))
    return out
