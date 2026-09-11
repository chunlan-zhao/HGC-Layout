"""Aggregation of per-scene records into the reported tables."""
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from hgc_layout.metrics.evaluate import HIGHER_IS_BETTER, METRIC_NAMES
from hgc_layout.utils.io import ensure_dir


def aggregate(rows: Sequence[Dict[str, object]], group_key: str = "method",
              metrics: Sequence[str] = METRIC_NAMES, with_std: bool = False,
              extra_keys: Sequence[str] = ()) -> List[Dict[str, object]]:
    """Mean (and optionally seed-level standard deviation) per group."""
    groups: Dict[tuple, List[Dict[str, object]]] = {}
    keys = (group_key,) + tuple(extra_keys)
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    out = []
    for key, items in groups.items():
        record: Dict[str, object] = dict(zip(keys, key))
        for metric in metrics:
            values = [float(i[metric]) for i in items if metric in i]
            if not values:
                continue
            record[metric] = float(np.mean(values))
            if with_std:
                per_seed = {}
                for i in items:
                    per_seed.setdefault(i["seed"], []).append(float(i[metric]))
                means = [float(np.mean(v)) for v in per_seed.values()]
                record[f"{metric}_std"] = float(np.std(means, ddof=1)) if len(means) > 1 else 0.0
        out.append(record)
    return out


def per_scene_series(rows: Sequence[Dict[str, object]], metrics: Sequence[str]
                     ) -> Dict[str, Dict[str, List[float]]]:
    """Per-method, per-metric vectors averaged over seeds and aligned by scene."""
    scenes = sorted({r["scene_id"] for r in rows})
    out: Dict[str, Dict[str, List[float]]] = {}
    for method in sorted({r["method"] for r in rows}):
        out[method] = {}
        for metric in metrics:
            series = []
            for scene in scenes:
                values = [float(r[metric]) for r in rows
                          if r["method"] == method and r["scene_id"] == scene and metric in r]
                series.append(float(np.mean(values)) if values else float("nan"))
            out[method][metric] = series
    return out


def _numeric(value) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _fmt(value: float, metric: str, decimals: Optional[int] = None) -> str:
    if decimals is not None:
        return f"{value:.{decimals}f}"
    if metric in ("SP", "Time"):
        return f"{value:.1f}"
    return f"{value:.3f}"


def markdown_table(records: Sequence[Dict[str, object]], columns: Sequence[str],
                   key_columns: Sequence[str] = ("method",), title: str = "",
                   with_std: bool = False, bold_best: bool = True) -> str:
    """Render aggregated records as a markdown table, marking the best value."""
    header = list(key_columns) + list(columns)
    lines = [f"**{title}**", ""] if title else []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    best: Dict[str, float] = {}
    for metric in columns:
        values = [float(r[metric]) for r in records if metric in r and _numeric(r[metric])]
        if values:
            best[metric] = max(values) if HIGHER_IS_BETTER.get(metric, True) else min(values)
    for record in records:
        cells = [str(record.get(k, "")) for k in key_columns]
        for metric in columns:
            if metric not in record:
                cells.append("")
                continue
            if not _numeric(record[metric]):
                cells.append(str(record[metric]))
                continue
            value = float(record[metric])
            text = _fmt(value, metric)
            if with_std and f"{metric}_std" in record:
                text += f" ± {_fmt(float(record[f'{metric}_std']), metric)}"
            if bold_best and metric in best and np.isclose(value, best[metric]):
                text = f"**{text}**"
            cells.append(text)
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def save_table(records: Sequence[Dict[str, object]], out_dir, name: str,
               columns: Sequence[str], key_columns: Sequence[str] = ("method",),
               title: str = "", with_std: bool = False) -> Path:
    """Write `<name>.md` and `<name>.csv`."""
    out_dir = ensure_dir(out_dir)
    text = markdown_table(records, columns, key_columns, title, with_std)
    (out_dir / f"{name}.md").write_text(text, encoding="utf-8")
    header = list(key_columns) + [c for c in columns] + \
             ([f"{c}_std" for c in columns] if with_std else [])
    lines = [",".join(header)]
    for record in records:
        lines.append(",".join(str(record.get(k, "")) for k in header))
    (out_dir / f"{name}.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_dir / f"{name}.md"


def significance_table(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """Format paired-test rows for reporting."""
    out = []
    for r in rows:
        out.append({"metric": r["metric"], "baseline": r["baseline"], "W": f"{r['W']:.1f}",
                    "p_raw": f"{r['p_raw']:.3e}", "p_holm": f"{r['p_holm']:.3e}",
                    "n": r["n"], "significant": "yes" if r["significant"] else "no"})
    return out
