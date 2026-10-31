"""Generalized Cochran-Mantel-Haenszel statistics (Landis, Heyman & Koch, 1978).

Notation follows the SAS PROC FREQ documentation.  For stratum h with an
R x C table n_h and n_h observations:

    G   = sum_h B_h (vec(n_h) - vec(m_h))
    V_G = sum_h B_h Var(vec(n_h) | H0) B_h'
    Q   = G' V_G^{-1} G          ~ chi-square(rank B_h) under H0

with B_h = C_h (x) R_h and the multiple hypergeometric covariance

    Var(vec(n_h) | H0) = n_h^2 / (n_h - 1) * (D_pc - pc pc') (x) (D_pr - pr pr')

Rather than building the RC x RC Kronecker products we use the identities
(C (x) R) vec(D) = vec(R D C')  and  (C (x) R)(A (x) B)(C (x) R)' = (CAC') (x) (RBR'),
which give the same numbers with far less memory.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy import linalg, stats

SCORE_TYPES = ("table", "rank", "ridit", "modridit")

# rank tolerance used to decide that V_G is singular (SAS then reports missing)
_SINGULAR_RTOL = 1e-10


def stratum_scores(margin: np.ndarray, kind: str, table_scores: np.ndarray) -> np.ndarray:
    """Scores for one margin of one stratum (SAS SCORES= option).

    Rank-type scores are midranks computed from the stratum's own margin, so
    tied observations share the midpoint of the ranks they occupy.  With
    ``modridit`` and two rows, the row mean scores statistic is van Elteren's
    stratified Wilcoxon test.
    """
    if kind == "table":
        return table_scores
    n = margin.sum()
    before = np.cumsum(margin) - margin
    rank = before + (margin + 1.0) / 2.0
    if kind == "rank":
        return rank
    if kind == "ridit":
        return rank / n
    if kind == "modridit":
        return rank / (n + 1.0)
    raise ValueError(f"scores must be one of {SCORE_TYPES}, got {kind!r}")


def _contrast(m: int) -> np.ndarray:
    """SAS contrast matrix [I_{m-1}, -J_{m-1}]."""
    return np.hstack([np.eye(m - 1), -np.ones((m - 1, 1))])


def _centered_quadratic(M: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Compute M (D_p - p p') M' without the cancellation of the naive formula.

    (D_p - p p') annihilates constant vectors, so each row of M can be centered
    at its p-weighted mean first; the result is then an explicit weighted
    cross-product and stays positive semi-definite in floating point.
    """
    Mc = M - (M @ p)[:, None]
    return (Mc * p) @ Mc.T


def _quadratic_form(G: np.ndarray, V: np.ndarray) -> float:
    """Return G' V^{-1} G, or NaN if V is singular (SAS sets the statistic to missing)."""
    V = (V + V.T) / 2.0
    eig = np.linalg.eigvalsh(V)
    top = eig[-1] if eig.size else 0.0
    if top <= 0 or eig[0] <= _SINGULAR_RTOL * top:
        return float("nan")
    try:
        cho = linalg.cho_factor(V)
        return float(G @ linalg.cho_solve(cho, G))
    except linalg.LinAlgError:  # pragma: no cover - guarded by the eigenvalue check
        return float("nan")


STAT_ORDER = ("correlation", "row_mean_scores", "general_association")
SAS_LABELS = {
    "correlation": "Nonzero Correlation",
    "row_mean_scores": "Row Mean Scores Differ",
    "general_association": "General Association",
}


def generalized_cmh(
    counts: np.ndarray,
    row_table_scores: np.ndarray,
    col_table_scores: np.ndarray,
    scores: str = "table",
) -> dict:
    """Compute the three CMH statistics for an R x C x K table of counts.

    Strata with fewer than two observations are skipped: with a single
    observation the observed table equals its expectation and the covariance
    is zero, so the stratum carries no information (its covariance formula is
    0 * n^2/(n-1), undefined at n = 1).
    """
    if scores not in SCORE_TYPES:
        raise ValueError(f"scores must be one of {SCORE_TYPES}, got {scores!r}")
    R, C, K = counts.shape
    dfs = {"correlation": 1, "row_mean_scores": R - 1, "general_association": (R - 1) * (C - 1)}
    G = {k: np.zeros(d) for k, d in dfs.items()}
    V = {k: np.zeros((d, d)) for k, d in dfs.items()}
    Rc, Cc = _contrast(R), _contrast(C)
    used, skipped = [], []

    for h in range(K):
        n = counts[:, :, h]
        N = n.sum()
        if N < 2:
            skipped.append(h)
            continue
        used.append(h)
        rs, cs = n.sum(axis=1), n.sum(axis=0)
        pr, pc = rs / N, cs / N
        D = n - np.outer(rs, cs) / N
        c = N * N / (N - 1.0)
        a = stratum_scores(rs, scores, row_table_scores)[None, :]
        b = stratum_scores(cs, scores, col_table_scores)[None, :]
        mats = {
            "correlation": (a, b),
            "row_mean_scores": (Rc, b),
            "general_association": (Rc, Cc),
        }
        for key, (Rh, Ch) in mats.items():
            G[key] += (Rh @ D @ Ch.T).ravel(order="F")
            V[key] += c * np.kron(_centered_quadratic(Ch, pc), _centered_quadratic(Rh, pr))

    out = {}
    singular = []
    for key in STAT_ORDER:
        q = _quadratic_form(G[key], V[key]) if used else float("nan")
        if np.isnan(q):
            singular.append(SAS_LABELS[key])
        p = float(stats.chi2.sf(q, dfs[key])) if not np.isnan(q) else float("nan")
        out[key] = (q, dfs[key], p)
    if singular:
        warnings.warn(
            "covariance matrix is singular for: " + ", ".join(singular)
            + "; statistic set to NaN (SAS reports it as missing)",
            stacklevel=3,
        )
    out["_used"] = used
    out["_skipped"] = skipped
    return out
