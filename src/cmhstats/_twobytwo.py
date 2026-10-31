"""Estimates and tests that only apply to 2 x 2 x K tables.

These reproduce what SAS PROC FREQ prints with the CMH (and BDT) options:

* Mantel-Haenszel common odds ratio, with Robins-Breslow-Greenland limits
* logit (Woolf) common odds ratio
* Mantel-Haenszel common relative risks for column 1 and column 2, with
  Greenland-Robins limits
* logit common relative risks for column 1 and column 2
* Breslow-Day test for homogeneity of odds ratios, optionally with Tarone's
  adjustment
* the Mantel-Fleiss criterion for the chi-square approximation

Row 1 is compared with row 2 throughout, as in SAS ("Row1/Row2").
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class Estimate:
    """A point estimate with confidence limits."""

    measure: str
    method: str
    value: float
    lower: float
    upper: float
    conf_level: float

    def __post_init__(self):
        for name in ("value", "lower", "upper", "conf_level"):
            object.__setattr__(self, name, float(getattr(self, name)))


@dataclass(frozen=True)
class HomogeneityTest:
    """A chi-square test for homogeneity of odds ratios across strata."""

    name: str
    statistic: float
    df: int
    pvalue: float


def _z(conf_level: float) -> float:
    if not 0 < conf_level < 1:
        raise ValueError("conf_level must be between 0 and 1")
    return float(stats.norm.ppf(1 - (1 - conf_level) / 2))


def _cells(counts: np.ndarray):
    counts = np.asarray(counts, dtype=float)
    counts = counts[:, :, counts.sum(axis=(0, 1)) > 0]  # empty strata carry nothing
    a, b = counts[0, 0, :], counts[0, 1, :]
    c, d = counts[1, 0, :], counts[1, 1, :]
    return a, b, c, d, a + b + c + d


def _safe_div(num: float, den: float) -> float:
    if den == 0:
        return float("nan") if num == 0 else float("inf")
    return num / den


def mh_odds_ratio(counts: np.ndarray, conf_level: float = 0.95) -> Estimate:
    """Mantel-Haenszel common odds ratio with Robins-Breslow-Greenland limits."""
    a, b, c, d, n = _cells(counts)
    R = a * d / n
    S = b * c / n
    P = (a + d) / n
    Q = (b + c) / n
    sR, sS = R.sum(), S.sum()
    est = _safe_div(sR, sS)
    if sR > 0 and sS > 0:
        var = (
            (P * R).sum() / (2 * sR**2)
            + (P * S + Q * R).sum() / (2 * sR * sS)
            + (Q * S).sum() / (2 * sS**2)
        )
        z = _z(conf_level)
        se = math.sqrt(var)
        lo, hi = est * math.exp(-z * se), est * math.exp(z * se)
    else:
        lo = hi = float("nan")
    return Estimate("Odds Ratio", "Mantel-Haenszel", est, lo, hi, conf_level)


def _mh_relative_risk(counts: np.ndarray, column: int, conf_level: float) -> Estimate:
    a, b, c, d, n = _cells(counts)
    if column == 1:
        x1, x2 = a, c  # events in column 1 for rows 1 and 2
    else:
        x1, x2 = b, d
    n1, n2 = a + b, c + d
    m = x1 + x2  # column total
    num = (x1 * n2 / n).sum()
    den = (x2 * n1 / n).sum()
    est = _safe_div(num, den)
    if num > 0 and den > 0:
        var = ((n1 * n2 * m - x1 * x2 * n) / n**2).sum() / (num * den)
        z = _z(conf_level)
        se = math.sqrt(var)
        lo, hi = est * math.exp(-z * se), est * math.exp(z * se)
    else:
        lo = hi = float("nan")
    return Estimate(f"Relative Risk (Column {column})", "Mantel-Haenszel", est, lo, hi, conf_level)


def _logit_pooled(log_est: np.ndarray, var: np.ndarray, measure: str, conf_level: float) -> Estimate:
    z = _z(conf_level)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = 1.0 / var
        sw = float(w.sum())
        mean = float((w * log_est).sum() / sw)
    if not (np.isfinite(mean) and np.isfinite(sw) and sw > 0):
        # a stratum with zero variance (e.g. every observation in one outcome) makes
        # the inverse-variance weights undefined
        nan = float("nan")
        return Estimate(measure, "Logit", nan, nan, nan, conf_level)
    est = math.exp(mean)
    half = z / math.sqrt(sw)
    return Estimate(measure, "Logit", est, est * math.exp(-half), est * math.exp(half), conf_level)


def logit_odds_ratio(counts: np.ndarray, conf_level: float = 0.95) -> Estimate:
    """Logit (Woolf) common odds ratio.

    As in SAS, 0.5 is added to every cell of a stratum that has a zero cell.
    """
    a, b, c, d, _ = (x.astype(float).copy() for x in _cells(counts))
    zero = (a == 0) | (b == 0) | (c == 0) | (d == 0)
    for arr in (a, b, c, d):
        arr[zero] += 0.5
    with np.errstate(divide="ignore", invalid="ignore"):
        log_or = np.log(a * d / (b * c))
        var = 1 / a + 1 / b + 1 / c + 1 / d
    return _logit_pooled(log_or, var, "Odds Ratio", conf_level)


def _logit_relative_risk(counts: np.ndarray, column: int, conf_level: float) -> Estimate:
    a, b, c, d, _ = (x.astype(float).copy() for x in _cells(counts))
    x1, x2 = (a, c) if column == 1 else (b, d)
    zero = (x1 == 0) | (x2 == 0)
    for arr in (a, b, c, d):
        arr[zero] += 0.5
    x1, x2 = (a, c) if column == 1 else (b, d)
    n1, n2 = a + b, c + d
    with np.errstate(divide="ignore", invalid="ignore"):
        p1, p2 = x1 / n1, x2 / n2
        log_rr = np.log(p1 / p2)
        var = (1 - p1) / x1 + (1 - p2) / x2
    return _logit_pooled(log_rr, var, f"Relative Risk (Column {column})", conf_level)


def _expected_a(or_mh: float, n1: float, n2: float, m1: float) -> float:
    """E(n11 | OR) for one stratum: the root of a(n2 - m1 + a) = OR (n1 - a)(m1 - a)
    that lies inside the support of n11."""
    lo, hi = max(0.0, m1 - n2), min(n1, m1)
    A = 1.0 - or_mh
    B = n2 - m1 + or_mh * (n1 + m1)
    Cq = -or_mh * n1 * m1
    if abs(A) < 1e-12:
        return -Cq / B
    disc = math.sqrt(max(B * B - 4 * A * Cq, 0.0))
    # numerically stable pair of roots
    q = -0.5 * (B + math.copysign(disc, B))
    roots = [q / A, Cq / q] if q != 0 else [(-B + disc) / (2 * A), (-B - disc) / (2 * A)]
    tol = 1e-9 * max(1.0, hi)
    for r in roots:
        if lo - tol <= r <= hi + tol:
            return min(max(r, lo), hi)
    raise RuntimeError("no admissible root for the Breslow-Day expected count")  # pragma: no cover


def breslow_day(counts: np.ndarray, tarone: bool = False) -> HomogeneityTest | None:
    """Breslow-Day test (optionally Breslow-Day-Tarone) for a 2 x 2 x K table.

    Strata with a zero row or column are left out of the sum, as in SAS.
    Returns None when the test is not defined (fewer than two usable strata,
    or a zero / infinite Mantel-Haenszel odds ratio).
    """
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 3 or counts.shape[:2] != (2, 2):
        raise ValueError("Breslow-Day test needs a 2 x 2 x K table")
    or_mh = mh_odds_ratio(counts).value
    if not np.isfinite(or_mh) or or_mh <= 0:
        return None
    a, b, c, d, _ = _cells(counts)
    n1, n2, m1, m2 = a + b, c + d, a + c, b + d
    ok = (n1 > 0) & (n2 > 0) & (m1 > 0) & (m2 > 0)
    if ok.sum() < 2:
        return None
    q = 0.0
    dev_sum = 0.0
    var_sum = 0.0
    for h in np.flatnonzero(ok):
        e = _expected_a(or_mh, n1[h], n2[h], m1[h])
        cells = (e, n1[h] - e, m1[h] - e, n2[h] - m1[h] + e)
        v = 1.0 / sum(1.0 / x for x in cells)
        dev = a[h] - e
        q += dev * dev / v
        dev_sum += dev
        var_sum += v
    if tarone:
        q -= dev_sum * dev_sum / var_sum
    df = int(ok.sum()) - 1
    name = "Breslow-Day-Tarone" if tarone else "Breslow-Day"
    return HomogeneityTest(name, float(q), df, float(stats.chi2.sf(q, df)))


def mantel_fleiss(counts: np.ndarray) -> float:
    """Mantel-Fleiss criterion; SAS warns when it is below 5."""
    a, b, c, d, n = _cells(counts)
    n1, n2, m1 = a + b, c + d, a + c
    m11 = n1 * m1 / n
    lower = np.maximum(0.0, n1 - (b + d))
    upper = np.minimum(m1, n1)
    return float(min(m11.sum() - lower.sum(), upper.sum() - m11.sum()))


def two_by_two_measures(counts: np.ndarray, conf_level: float = 0.95) -> dict:
    """All of the 2 x 2 x K output of SAS's CMH option in one dictionary."""
    return {
        "common_odds_ratio": (mh_odds_ratio(counts, conf_level), logit_odds_ratio(counts, conf_level)),
        "relative_risk_col1": (
            _mh_relative_risk(counts, 1, conf_level),
            _logit_relative_risk(counts, 1, conf_level),
        ),
        "relative_risk_col2": (
            _mh_relative_risk(counts, 2, conf_level),
            _logit_relative_risk(counts, 2, conf_level),
        ),
        "breslow_day": breslow_day(counts, tarone=False),
        "breslow_day_tarone": breslow_day(counts, tarone=True),
        "mantel_fleiss": mantel_fleiss(counts),
    }
