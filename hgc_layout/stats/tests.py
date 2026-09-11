"""Paired significance testing with family-wise error control.

Comparisons use the two-sided Wilcoxon signed-rank test with exact P values,
appropriate at the sample size of a 20-scene test split, and the family-wise
error rate is controlled with the Holm-Bonferroni procedure across the whole
family of tests.
"""
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats


def wilcoxon_exact(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    """W statistic and exact two-sided P value for paired samples."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diff = a - b
    if np.allclose(diff, 0):
        return 0.0, 1.0
    mode = "exact" if len(diff) <= 25 and len(np.unique(np.abs(diff[diff != 0]))) == np.sum(diff != 0) else "auto"
    res = stats.wilcoxon(a, b, alternative="two-sided", method=mode, zero_method="wilcox")
    return float(res.statistic), float(res.pvalue)


def holm_bonferroni(pvalues: Sequence[float]) -> List[float]:
    """Holm-adjusted P values, monotone and clipped to 1."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adjusted[idx] = min(running, 1.0)
    return [float(v) for v in adjusted]


def paired_comparisons(scores: Dict[str, Dict[str, List[float]]], method: str,
                       baselines: Sequence[str], metrics: Sequence[str],
                       alpha: float = 0.05) -> List[Dict[str, object]]:
    """One row per (metric, baseline) comparison, with Holm-adjusted P values.

    `scores[method][metric]` holds the per-scene values, aligned across methods.
    """
    rows: List[Dict[str, object]] = []
    for metric in metrics:
        for baseline in baselines:
            w, p = wilcoxon_exact(scores[method][metric], scores[baseline][metric])
            rows.append({"metric": metric, "baseline": baseline, "W": w, "p_raw": p,
                         "n": len(scores[method][metric])})
    for row, adj in zip(rows, holm_bonferroni([r["p_raw"] for r in rows])):
        row["p_holm"] = adj
        row["significant"] = bool(adj < alpha)
    return rows
