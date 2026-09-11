"""Load and validate benchmark scenes from disk."""
from pathlib import Path
from typing import Dict, List

from hgc_layout.data.schema import SCENE_REQUIRED_KEYS, Scene
from hgc_layout.utils.io import read_json, write_json

SPLITS = ("train", "val", "test")


def _validate(d: Dict, path: Path) -> None:
    missing = [k for k in SCENE_REQUIRED_KEYS if k not in d]
    if missing:
        raise ValueError(f"{path} is missing keys {missing}; see data/README.md for the schema.")


def load_scene(path) -> Scene:
    """Read one scene JSON file."""
    path = Path(path)
    d = read_json(path)
    _validate(d, path)
    return Scene.from_dict(d)


def save_scene(scene: Scene, path) -> Path:
    return write_json(scene.as_dict(), path)


def load_benchmark(root, split: str = None) -> List[Scene]:
    """Read every scene under `root` (optionally one split subdirectory)."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"Expected benchmark scenes under {root}. The reference benchmark is not "
            "distributed with this repository; see data/README.md for the schema and "
            "sources, or generate a synthetic benchmark with scripts/prepare_data.py."
        )
    subdirs = [root / split] if split else [root / s for s in SPLITS if (root / s).is_dir()]
    if not subdirs or not any(p.is_dir() for p in subdirs):
        subdirs = [root]
    scenes: List[Scene] = []
    for d in subdirs:
        for f in sorted(Path(d).glob("*.json")):
            scenes.append(load_scene(f))
    if not scenes:
        raise FileNotFoundError(f"No scene JSON files found under {root}.")
    return scenes


def split_scenes(scenes: List[Scene]) -> Dict[str, List[Scene]]:
    """Group scenes by their `split` field."""
    out: Dict[str, List[Scene]] = {s: [] for s in SPLITS}
    for sc in scenes:
        out.setdefault(sc.split, []).append(sc)
    return out
