"""Reliability of a repeated-rating metric: ICC and rank correlations.

Implements the two-way random-effects intraclass correlation coefficients for
the mean of k raters: ICC(A,k) for absolute agreement and ICC(C,k) for
consistency, together with pairwise Spearman correlations and the mean
within-subject standard deviation.
"""
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats


def icc(ratings: np.ndarray) -> Dict[str, float]:
    """ICC(A,k) and ICC(C,k) for a subjects-by-raters matrix."""
    x = np.asarray(ratings, dtype=float)
    n, k = x.shape
    grand = x.mean()
    ms_rows = k * ((x.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    ms_cols = n * ((x.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    residual = x - x.mean(axis=1, keepdims=True) - x.mean(axis=0, keepdims=True) + grand
    ms_err = (residual ** 2).sum() / ((n - 1) * (k - 1))
    icc_ck = (ms_rows - ms_err) / ms_rows if ms_rows > 0 else float("nan")
    denom = ms_rows + (ms_cols - ms_err) / n
    icc_ak = (ms_rows - ms_err) / denom if denom > 0 else float("nan")
    return {"ICC(A,k)": float(icc_ak), "ICC(C,k)": float(icc_ck)}


def rank_correlations(ratings: np.ndarray) -> List[float]:
    """Pairwise Spearman correlations between raters."""
    x = np.asarray(ratings, dtype=float)
    out = []
    for a in range(x.shape[1]):
        for b in range(a + 1, x.shape[1]):
            out.append(float(stats.spearmanr(x[:, a], x[:, b]).statistic))
    return out


def within_subject_sd(ratings: np.ndarray) -> float:
    """Mean standard deviation across raters within each subject."""
    return float(np.asarray(ratings, dtype=float).std(axis=1, ddof=1).mean())


def reliability_report(ratings: np.ndarray) -> Dict[str, object]:
    """ICCs, Spearman correlations and the mean within-subject standard deviation."""
    out: Dict[str, object] = dict(icc(ratings))
    out["spearman"] = rank_correlations(ratings)
    out["within_sd"] = within_subject_sd(ratings)
    return out
