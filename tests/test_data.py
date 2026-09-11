import numpy as np
import pytest

from hgc_layout.data.features import POIEncoder, RemoteSensingEncoder, structural_features
from hgc_layout.data.loaders import load_benchmark, load_scene, save_scene, split_scenes
from hgc_layout.data.schema import SCENE_REQUIRED_KEYS, Scene
from hgc_layout.data.sparsity import degrade_scene
from hgc_layout.data.synthetic import generate_benchmark


def test_scene_schema(scene):
    d = scene.as_dict()
    for key in SCENE_REQUIRED_KEYS:
        assert key in d
    assert Scene.from_dict(d).scene_id == scene.scene_id
    assert scene.size == (1000.0, 1000.0)


def test_entities_inside_boundary(scenes):
    for s in scenes:
        length, width = s.size
        for e in s.entities:
            p = np.asarray(e["position"])
            half = np.asarray(e["size"][:2]) / 2
            assert np.all(p - half >= -1e-6) and np.all(p + half <= np.array([length, width]) + 1e-6)
            assert 0 <= e["orientation"] < 2 * np.pi


def test_geographic_split_is_tile_disjoint(scenes):
    by_tile = {}
    for s in scenes:
        by_tile.setdefault(s.tile_id, set()).add(s.split)
    assert all(len(v) == 1 for v in by_tile.values())


def test_relations_carry_labels(scenes):
    for s in scenes:
        for r in s.relations:
            assert {"source", "target", "type", "source_label", "target_label"} <= set(r)


def test_round_trip(tmp_path, scene):
    path = save_scene(scene, tmp_path / "train" / f"{scene.scene_id}.json")
    assert load_scene(path).description == scene.description
    loaded = load_benchmark(tmp_path)
    assert len(loaded) == 1
    assert list(split_scenes(loaded)[scene.split])[0].scene_id == scene.scene_id


def test_missing_benchmark_message(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        load_benchmark(tmp_path / "absent")
    assert "data/README.md" in str(exc.value)


def test_degrade_removes_inputs(scene):
    rng = np.random.default_rng(0)
    degraded = degrade_scene(scene, 0.75, rng)
    assert len(degraded.pois) <= len(scene.pois)
    assert len(degraded.roads["edges"]) <= len(scene.roads["edges"])
    assert len(degraded.entities) == len(scene.entities)
    assert degrade_scene(scene, 0.0, rng) is scene


def test_feature_encoders_are_deterministic():
    poi = POIEncoder()
    a, b = poi.encode("cafe"), poi.encode("cafe")
    assert np.allclose(a, b)
    assert not np.allclose(a, poi.encode("warehouse"))
    rs = RemoteSensingEncoder()
    v = rs.encode(np.full(8, 0.3), "e00")
    assert v.shape == (512,) and np.isfinite(v).all()
    s = structural_features(3, {"proximity": 2, "road": 1}, 0.5)
    assert s.shape == (16,)


def test_generate_benchmark_sizes():
    scenes = generate_benchmark(12, seed=1)
    assert len(scenes) == 12
    assert {s.split for s in scenes} <= {"train", "val", "test"}
    assert all(len(s.entities) >= 6 for s in scenes)
