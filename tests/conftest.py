from pathlib import Path

import pytest

from hgc_layout.data.synthetic import generate_benchmark
from hgc_layout.factory import build_model
from hgc_layout.planning.planner import build_planner
from hgc_layout.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def rules_path():
    return ROOT / "data" / "rules" / "layout_rules.yaml"


@pytest.fixture(scope="session")
def planner(rules_path):
    return build_planner("template", rules_path)


@pytest.fixture(scope="session")
def scenes():
    return generate_benchmark(8, seed=3)


@pytest.fixture(scope="session")
def scene(scenes):
    return scenes[0]


@pytest.fixture(scope="session")
def demo_config():
    cfg = load_config(ROOT / "configs" / "demo.yaml")
    cfg["data"]["rules"] = str(ROOT / "data" / "rules" / "layout_rules.yaml")
    cfg["optimizer"]["max_iter"] = 30
    cfg["graph"]["pretrain_epochs"] = 2
    cfg["generator"]["epochs"] = 2
    cfg["optimizer"]["epochs"] = 1
    cfg["baselines"]["epochs"] = 2
    return cfg


@pytest.fixture(scope="session")
def model(demo_config, planner):
    return build_model(demo_config, planner=planner)
