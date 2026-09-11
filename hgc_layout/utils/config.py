"""YAML configuration loading with attribute access and light validation."""
from pathlib import Path
from typing import Any

import yaml

REQUIRED_SECTIONS = ("data", "graph", "generator", "optimizer", "evaluation", "output")


class Config(dict):
    """Dict with attribute access on nested mappings."""

    def __getattr__(self, item: str) -> Any:
        try:
            value = self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc
        return Config(value) if isinstance(value, dict) else value


def load_config(path) -> Config:
    """Load a YAML config file and check that the top-level sections exist."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    missing = [s for s in REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise KeyError(f"Config {path} is missing sections: {missing}")
    cfg.setdefault("seed", 0)
    cfg.setdefault("seeds", [0, 1, 2])
    return Config(cfg)
