"""Public entry points.

* :func:`cmh_test` mirrors SAS ``PROC FREQ ... / CMH``.
* :func:`mantelhaen_test` mirrors R's ``stats::mantelhaen.test`` (including its
  default continuity correction for 2 x 2 x K tables), so results can be
  checked against either reference without guessing at option differences.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from scipy import stats

from ._core import SAS_LABELS, STAT_ORDER, SCORE_TYPES, generalized_cmh
from ._input import StratifiedTable, as_stratified_table
from ._twobytwo import Estimate, HomogeneityTest, two_by_two_measures


@dataclass(frozen=True)
class CMHStatistic:
    """One of the three CMH statistics."""

    name: str
    alternative: str
    statistic: float
    df: int
    pvalue: float

    def __post_init__(self):
        object.__setattr__(self, "statistic", float(self.statistic))
        object.__setattr__(self, "pvalue", float(self.pvalue))
        object.__setattr__(self, "df", int(self.df))


@dataclass(frozen=True)
class CMHResult:
    """Everything SAS PROC FREQ prints for the CMH option.

    The 2 x 2 x K fields (odds ratios, relative risks, Breslow-Day) are None
    unless the table has exactly two rows and two columns.
    """

    correlation: CMHStatistic
    row_mean_scores: CMHStatistic
    general_association: CMHStatistic
    scores: str
    n_total: float
    row_levels: tuple
    col_levels: tuple
    strata_levels: tuple
    strata_excluded: tuple
    row_scores: np.ndarray
    col_scores: np.ndarray
    common_odds_ratio: tuple[Estimate, Estimate] | None = None
    relative_risk_col1: tuple[Estimate, Estimate] | None = None
    relative_risk_col2: tuple[Estimate, Estimate] | None = None
    breslow_day: HomogeneityTest | None = None
    breslow_day_tarone: HomogeneityTest | None = None
    mantel_fleiss: float | None = None
    notes: tuple[str, ...] = field(default=())

    @property
    def statistics(self) -> tuple[CMHStatistic, CMHStatistic, CMHStatistic]:
        return (self.correlation, self.row_mean_scores, self.general_association)

    def to_frame(self):
        """The CMH statistics as a pandas DataFrame (SAS layout)."""
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "Statistic": i + 1,
                    "Alternative Hypothesis": s.alternative,
                    "DF": s.df,
                    "Value": s.statistic,
                    "Prob": s.pvalue,
                }
                for i, s in enumerate(self.statistics)
            ]
        ).set_index("Statistic")

    def estimates_frame(self):
        """Common odds ratio and relative risks as a DataFrame (2 x 2 x K only)."""
        import pandas as pd

        if self.common_odds_ratio is None:
            raise ValueError("common odds ratios and relative risks need a 2 x 2 x K table")
        rows = []
        for group in (self.common_odds_ratio, self.relative_risk_col1, self.relative_risk_col2):
            for e in group:  # type: ignore[union-attr]
                rows.append(
                    {"Statistic": e.measure, "Method": e.method, "Value": e.value,
                     f"{e.conf_level:.0%} Lower CL": e.lower, f"{e.conf_level:.0%} Upper CL": e.upper}
                )
        return pd.DataFrame(rows)

    def summary(self, digits: int = 4) -> str:
        """Plain-text output in the layout of SAS PROC FREQ."""
        f = lambda v: "." if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.{digits}f}"
        p = lambda v: "." if math.isnan(v) else ("<.0001" if v < 1e-4 else f"{v:.4f}")
        score_name = {"table": "Table", "rank": "Rank", "ridit": "Ridit", "modridit": "Modified Ridit"}
        lines = [f"Cochran-Mantel-Haenszel Statistics (Based on {score_name[self.scores]} Scores)", ""]
        lines.append(f"{'Statistic':<10}{'Alternative Hypothesis':<26}{'DF':>4}{'Value':>12}{'Prob':>10}")
        lines.append("-" * 62)
        for i, s in enumerate(self.statistics, 1):
            lines.append(f"{i:<10}{s.alternative:<26}{s.df:>4}{f(s.statistic):>12}{p(s.pvalue):>10}")
        if self.common_odds_ratio is not None:
            cl = int(round(self.common_odds_ratio[0].conf_level * 100))
            lines += ["", "Common Odds Ratio and Relative Risks", ""]
            lines.append(f"{'Statistic':<26}{'Method':<17}{'Value':>9}{cl:>9}% CL")
            lines.append("-" * 70)
            for group in (self.common_odds_ratio, self.relative_risk_col1, self.relative_risk_col2):
                for e in group:  # type: ignore[union-attr]
                    lines.append(
                        f"{e.measure:<26}{e.method:<17}{f(e.value):>9}{f(e.lower):>10}{f(e.upper):>10}"
                    )
            for test in (self.breslow_day, self.breslow_day_tarone):
                if test is not None:
                    lines += ["", f"{test.name} Test for Homogeneity of Odds Ratios",
                              f"  Chi-Square  {f(test.statistic)}", f"  DF          {test.df}",
                              f"  Pr > ChiSq  {p(test.pvalue)}"]
        lines += ["", f"Total Sample Size = {self.n_total:g}"]
        if self.strata_excluded:
            lines.append(
                f"Strata with fewer than 2 observations (no information): {len(self.strata_excluded)}"
            )
        if self.mantel_fleiss is not None and self.mantel_fleiss < 5:
            lines.append(
                f"WARNING: Mantel-Fleiss criterion = {self.mantel_fleiss:.4g} < 5; "
                "the chi-square approximation may be poor."
            )
        for note in self.notes:
            lines.append(f"Note: {note}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def cmh_test(
    table: Any = None,
    *,
    data: Any = None,
    x: str | None = None,
    y: str | None = None,
    strata: str | Sequence[str] | None = None,
    weight: str | None = None,
    scores: str = "table",
    row_scores: Sequence[float] | None = None,
    col_scores: Sequence[float] | None = None,
    order: str = "internal",
    conf_level: float = 0.95,
) -> CMHResult:
    """Cochran-Mantel-Haenszel statistics, matching SAS ``PROC FREQ / CMH``.

    Parameters
    ----------
    table
        Array of counts with shape ``(R, C, K)``: rows = X, columns = Y,
        last axis = strata.  A 2-D array is treated as a single stratum.
    data, x, y, strata, weight
        Alternatively, a long-format DataFrame and column names.  ``strata``
        may be a list of columns, whose combinations form the strata (as in
        SAS ``tables A*B*X*Y``).  ``weight`` is SAS's WEIGHT variable.
    scores
        ``"table"`` (default), ``"rank"``, ``"ridit"`` or ``"modridit"``, as
        the SAS ``SCORES=`` option.  Rank-type scores are computed within each
        stratum; ``scores="modridit"`` with two rows gives van Elteren's test.
    row_scores, col_scores
        Custom table scores, one per level in level order.
    order
        ``"internal"`` (sorted values, SAS default) or ``"data"`` (order of
        first appearance) for DataFrame input.
    conf_level
        Confidence level for the 2 x 2 x K estimates.

    Returns
    -------
    CMHResult
        ``.correlation``, ``.row_mean_scores`` and ``.general_association``,
        plus odds ratios, relative risks and the Breslow-Day test for
        2 x 2 x K tables.  ``print(result)`` gives SAS-style output.

    Notes
    -----
    * Strata with a single observation carry no information and are skipped.
    * If the covariance matrix of a statistic is singular, the statistic is
      NaN and a warning is issued (SAS prints it as missing).
    * No continuity correction is applied, as in SAS.  Use
      :func:`mantelhaen_test` to reproduce R's corrected default.
    """
    if scores not in SCORE_TYPES:
        raise ValueError(f"scores must be one of {SCORE_TYPES}, got {scores!r}")
    if scores != "table" and (row_scores is not None or col_scores is not None):
        raise ValueError("custom row_scores/col_scores can only be used with scores='table'")

    st: StratifiedTable = as_stratified_table(
        table, data=data, x=x, y=y, strata=strata, weight=weight, order=order,
        row_scores=row_scores, col_scores=col_scores,
    )
    core = generalized_cmh(st.counts, st.row_table_scores, st.col_table_scores, scores)
    stats_out = {
        key: CMHStatistic(key, SAS_LABELS[key], *core[key]) for key in STAT_ORDER
    }
    extra: dict[str, Any] = {}
    if st.counts.shape[:2] == (2, 2):
        extra = two_by_two_measures(st.counts, conf_level)
        if extra["mantel_fleiss"] < 5:
            warnings.warn(
                f"Mantel-Fleiss criterion is {extra['mantel_fleiss']:.4g} (< 5): "
                "the chi-square approximation may not be valid",
                stacklevel=2,
            )

    return CMHResult(
        correlation=stats_out["correlation"],
        row_mean_scores=stats_out["row_mean_scores"],
        general_association=stats_out["general_association"],
        scores=scores,
        n_total=float(st.counts.sum()),
        row_levels=st.row_levels,
        col_levels=st.col_levels,
        strata_levels=st.strata_levels,
        strata_excluded=tuple(st.strata_levels[h] for h in core["_skipped"]),
        row_scores=st.row_table_scores,
        col_scores=st.col_table_scores,
        notes=st.notes,
        **extra,
    )


@dataclass(frozen=True)
class MantelHaenszelResult:
    """Result in the shape of R's ``htest`` object from ``mantelhaen.test``."""

    statistic: float
    df: int
    pvalue: float
    method: str
    alternative: str
    estimate: float | None = None
    conf_int: tuple[float, float] | None = None
    conf_level: float | None = None

    def __str__(self) -> str:
        lines = [self.method, "",
                 f"Mantel-Haenszel X-squared = {self.statistic:.6g}, df = {self.df}, "
                 f"p-value = {self.pvalue:.4g}"]
        if self.estimate is not None and self.conf_int is not None:
            lines.append(
                f"{self.conf_level:.0%} confidence interval: "
                f"{self.conf_int[0]:.6g} {self.conf_int[1]:.6g}"
            )
            lines.append(f"common odds ratio: {self.estimate:.6g}")
        return "\n".join(lines)


def mantelhaen_test(
    table: Any,
    *,
    alternative: str = "two.sided",
    correct: bool = True,
    conf_level: float = 0.95,
) -> MantelHaenszelResult:
    """Port of R's ``stats::mantelhaen.test`` (asymptotic version).

    Defaults match R: for 2 x 2 x K tables a continuity correction is applied
    (``correct=True``) and the Mantel-Haenszel common odds ratio is returned
    with Robins-Breslow-Greenland limits.  For larger tables the generalized
    statistic (general association, no correction) is returned.

    Like R, this raises an error when a stratum has fewer than two
    observations; :func:`cmh_test` skips such strata instead.  ``exact=TRUE``
    is not implemented.
    """
    arr = np.asarray(table, dtype=float)
    if arr.ndim != 3:
        raise ValueError("'table' must be a 3-dimensional array")
    if np.any(arr < 0) or not np.all(np.isfinite(arr)):
        raise ValueError("'table' must contain finite, non-negative counts")
    I, J, K = arr.shape
    if K < 2:
        raise ValueError("'table' must be a 3-dimensional array with at least 2 strata (R requirement)")
    if I < 2 or J < 2:
        raise ValueError("each dimension in table must be >= 2")
    n = arr.sum(axis=(0, 1))
    if np.any(n < 2):
        raise ValueError("sample size in each stratum must be > 1")
    if alternative not in ("two.sided", "less", "greater"):
        raise ValueError("alternative must be 'two.sided', 'less' or 'greater'")

    if (I, J) == (2, 2):
        s_x = arr.sum(axis=1)  # row totals, shape (2, K)
        s_y = arr.sum(axis=0)  # column totals, shape (2, K)
        delta = float((arr[0, 0, :] - s_x[0] * s_y[0] / n).sum())
        yates = 0.5 if (correct and abs(delta) >= 0.5) else 0.0
        var = (s_x[0] * s_x[1] * s_y[0] * s_y[1] / (n * n * (n - 1))).sum()
        stat = (abs(delta) - yates) ** 2 / var if var > 0 else float("nan")  # R gives NaN
        if math.isnan(stat):
            pval = float("nan")
        elif alternative == "two.sided":
            pval = float(stats.chi2.sf(stat, 1))
        else:
            z = math.copysign(math.sqrt(stat), delta)
            pval = float(stats.norm.cdf(z) if alternative == "less" else stats.norm.sf(z))
        a, b, c, d = arr[0, 0, :], arr[0, 1, :], arr[1, 0, :], arr[1, 1, :]
        s_diag = (a * d / n).sum()
        s_offd = (b * c / n).sum()
        est = s_diag / s_offd if s_offd > 0 else (float("inf") if s_diag > 0 else float("nan"))
        sd = math.sqrt(
            ((a + d) * a * d / n**2).sum() / (2 * s_diag**2)
            + (((a + d) * b * c + (b + c) * a * d) / n**2).sum() / (2 * s_diag * s_offd)
            + ((b + c) * b * c / n**2).sum() / (2 * s_offd**2)
        ) if s_diag > 0 and s_offd > 0 else float("nan")
        if alternative == "less":
            ci = (0.0, est * math.exp(stats.norm.ppf(conf_level) * sd))
        elif alternative == "greater":
            ci = (est * math.exp(stats.norm.ppf(1 - conf_level) * sd), float("inf"))
        else:
            q = stats.norm.ppf((1 - conf_level) / 2)
            ci = (est * math.exp(q * sd), est * math.exp(-q * sd))
        method = "Mantel-Haenszel chi-squared test " + (
            "with continuity correction" if yates else "without continuity correction"
        )
        return MantelHaenszelResult(float(stat), 1, pval, method, alternative, float(est),
                                    (float(ci[0]), float(ci[1])), conf_level)

    core = generalized_cmh(arr, np.arange(1, I + 1.0), np.arange(1, J + 1.0), "table")
    stat, df, pval = core["general_association"]
    if math.isnan(stat):
        raise np.linalg.LinAlgError("covariance matrix is singular (R's qr.solve fails here too)")
    return MantelHaenszelResult(stat, df, pval, "Cochran-Mantel-Haenszel test", "two.sided")
