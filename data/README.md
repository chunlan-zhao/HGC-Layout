# Data

The evaluation benchmark is a set of text-conditioned geospatial scenes, each covering a
1 km² region: a natural-language description, a difficulty label, auxiliary geospatial
inputs (POIs, road network, land-cover patches, remote-sensing patch statistics) and a
curated reference layout. The reference benchmark is under embargo and is **not**
distributed with this repository. A synthetic benchmark with the same schema is generated
by `scripts/prepare_data.py` so that every script runs end to end; numbers computed from
it are illustrative only.

## Third-party sources

| Input | Original source | License / access |
|---|---|---|
| Building footprints, road networks, POI records | OpenStreetMap (https://www.openstreetmap.org) | Open Database License (ODbL) 1.0, © OpenStreetMap contributors |
| Satellite imagery patches | Copernicus Sentinel-2 (https://dataspace.copernicus.eu) | Free, full and open Copernicus data policy |
| Benchmark scale and organization (100 descriptions, 10 categories, 3 difficulty levels) | SceneEval-100, the indoor benchmark this one mirrors | See the original release for its terms |
| Remote-sensing patch encoder | ResNet-18, `IMAGENET1K_V1` weights, torchvision (https://pytorch.org/vision) | BSD-3-Clause (weights subject to the ImageNet terms) |
| POI category and retrieval text embeddings | `sentence-transformers/all-MiniLM-L6-v2` (https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | Apache-2.0 |
| Asset retrieval image-text similarity | OpenAI CLIP ViT-B/32 (https://github.com/openai/CLIP) | MIT |
| CLIP similarity metric | Long-CLIP-L checkpoint (https://github.com/beichenzbc/Long-CLIP) | See the original release for its terms |
| Scene planner and semantic-plausibility judge | A chat model reached through an OpenAI-compatible endpoint | Commercial API terms |

All third-party models are optional. Install them with `pip install -e .[full]` and set
`OPENAI_API_KEY` for the language-model backends. Without them the pipeline runs with
deterministic offline substitutes, described under "Offline substitutes" below.

## Expected layout for the real benchmark

```
data/benchmark/train/scene_000.json
data/benchmark/val/scene_012.json
data/benchmark/test/scene_037.json
```

One JSON file per scene, grouped by split. `scripts/prepare_data.py --data real
--benchmark-dir data/benchmark` validates the schema and names the first missing key.
Required keys (see `hgc_layout/data/schema.py`):

| Key | Content |
|---|---|
| `scene_id`, `category`, `difficulty`, `tile_id`, `split` | identifiers; `tile_id` drives the geographic split |
| `description` | the natural-language prompt |
| `boundary` | `{"length": metres, "width": metres}` |
| `entities` | list of objects with `id`, `type`, `category`, `description`, `size` `[l, w, h]`, `group`, `parent`, and the reference `position` and `orientation` |
| `relations` | described pairwise relations, each with `source`, `target`, `type`, and the `source_label` / `target_label` used to match re-planned entities |
| `area_ratio` | reference footprint-area share per functional group |
| `pois`, `roads`, `land_cover` | auxiliary inputs; `roads` holds `nodes`, `edges` and optional arterial `axes` |
| `rs_stats` | eight-channel remote-sensing summary per node, replaced by imagery features when a backbone is supplied |

Coordinates are metres inside the scene boundary; orientations are radians in [0, 2π).
Reference positions and orientations are the evaluation target: no code path reads them
during inference, and the graph never receives them (see
`hgc_layout/graph/construction.py`).

## Splits

Scenes are partitioned 60/20/20 by `tile_id`, so two regions drawn from the same tile
always land in the same split and spatial adjacency cannot leak across splits. The
synthetic generator applies the same rule.

## Offline substitutes

| Component | With optional dependencies | Offline substitute |
|---|---|---|
| Remote-sensing features | ResNet-18 on the imagery patch | the stored patch statistics expanded to 512 channels |
| POI and retrieval text embeddings | sentence encoder | deterministic hashed character-trigram embedding |
| CLIPsim | Long-CLIP on a top-down render | the same hashed encoder applied to a textual layout render |
| Scene planner | chat model | template planner parsing counted noun phrases plus the retrieved rules |
| Semantic plausibility (SP) | chat-model judgment | rule-based score over the plausibility terms and spatial spread |

The substitutes are deterministic and reproducible, but they are not the models they
stand in for. `semantic_plausibility` returns the backend that produced each score, and
an offline SP score must not be reported as a model judgment.

## Layout rules

`data/rules/layout_rules.yaml` holds the retrievable geospatial rules: setbacks, green
accessibility, retail clustering, industrial separation, transport access, waterfront
edges, zone shares, boundary containment, facade orientation and zone hierarchy. Each
rule carries the scene categories it applies to, retrieval keywords and the numeric
constraints it contributes to the plan. Edit this file to adapt the planner to another
regulatory context.
