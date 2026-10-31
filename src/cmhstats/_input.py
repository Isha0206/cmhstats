"""Turn user input (an array or a long-format DataFrame) into an R x C x K table.

The conventions here follow SAS PROC FREQ, because SAS output is the usual
reference point in clinical reporting:

* ``tables K * X * Y`` in SAS means X defines the rows, Y the columns and K the
  strata.  An array input therefore has shape ``(R, C, K)``, which is also the
  layout R's ``mantelhaen.test`` expects.
* Levels are ordered by their unformatted values (SAS ``ORDER=INTERNAL``):
  numbers ascending, strings in code-point order.  ``order="data"`` keeps the
  order of first appearance (SAS ``ORDER=DATA``).  An ordered pandas
  ``Categorical`` keeps its category order.
* Table scores are the numeric level values for numeric variables and the
  positions 1, 2, 3, ... for character variables.
* Observations with a missing value or a zero weight are excluded, as SAS does
  by default.  Levels that never occur are dropped.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class StratifiedTable:
    """An R x C x K table of counts plus the labels and table scores of its levels."""

    counts: np.ndarray
    row_levels: tuple
    col_levels: tuple
    strata_levels: tuple
    row_table_scores: np.ndarray
    col_table_scores: np.ndarray
    notes: tuple[str, ...] = field(default=())

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.counts.shape  # type: ignore[return-value]


def _is_numeric_kind(values: np.ndarray) -> bool:
    return values.dtype.kind in "biuf"


def _ordered_levels(series: Any, order: str) -> tuple[list, np.ndarray | None]:
    """Return the observed levels of a pandas Series in the requested order,
    and numeric table scores if the variable is numeric."""
    import pandas as pd

    if isinstance(series.dtype, pd.CategoricalDtype):
        observed = set(series.dropna().unique().tolist())
        levels = [c for c in series.cat.categories.tolist() if c in observed]
        cats = np.asarray(series.cat.categories)
        numeric = _is_numeric_kind(cats)
    else:
        uniques = pd.unique(series.dropna())
        numeric = _is_numeric_kind(np.asarray(uniques))
        if order == "data":
            levels = list(uniques)
        elif numeric:
            levels = sorted(uniques.tolist())
        else:
            try:
                levels = sorted(uniques.tolist())
            except TypeError:  # mixed types: fall back to string order
                levels = sorted(uniques.tolist(), key=str)
    scores = np.asarray(levels, dtype=float) if numeric else None
    return levels, scores


def _from_dataframe(
    data: Any,
    x: str,
    y: str,
    strata: str | Sequence[str] | None,
    weight: str | None,
    order: str,
) -> StratifiedTable:
    import pandas as pd

    if order not in ("internal", "data"):
        raise ValueError("order must be 'internal' or 'data'")
    strata_cols: list[str]
    if strata is None:
        strata_cols = []
    elif isinstance(strata, str):
        strata_cols = [strata]
    else:
        strata_cols = list(strata)
    cols = [x, y, *strata_cols] + ([weight] if weight is not None else [])
    missing_cols = [c for c in cols if c not in data.columns]
    if missing_cols:
        raise KeyError(f"columns not found in data: {missing_cols}")

    df = data[cols]
    notes: list[str] = []
    keep = df.notna().all(axis=1)
    n_missing = int((~keep).sum())
    if n_missing:
        notes.append(f"{n_missing} observation(s) with missing values excluded")
    df = df[keep]

    if weight is not None:
        w = pd.to_numeric(df[weight], errors="raise").to_numpy(dtype=float)
        if np.any(w < 0):
            raise ValueError("weights must be non-negative")
        if not np.all(np.isfinite(w)):
            raise ValueError("weights must be finite")
        nonzero = w > 0
        if (~nonzero).any():
            notes.append(f"{int((~nonzero).sum())} observation(s) with zero weight excluded")
        df = df[nonzero]
        w = w[nonzero]
    else:
        w = np.ones(len(df), dtype=float)

    if len(df) == 0:
        raise ValueError("no observations left after excluding missing values and zero weights")

    row_levels, row_scores = _ordered_levels(df[x], order)
    col_levels, col_scores = _ordered_levels(df[y], order)

    row_index = {lvl: i for i, lvl in enumerate(row_levels)}
    col_index = {lvl: j for j, lvl in enumerate(col_levels)}
    ri = df[x].map(row_index).to_numpy(dtype=int)
    ci = df[y].map(col_index).to_numpy(dtype=int)

    if strata_cols:
        per_col = [_ordered_levels(df[s], order)[0] for s in strata_cols]
        codes = []
        for s, lv in zip(strata_cols, per_col):
            idx = {v: i for i, v in enumerate(lv)}
            codes.append(df[s].map(idx).to_numpy(dtype=int))
        keys = list(zip(*codes))
        unique_keys = sorted(set(keys))
        key_index = {k: h for h, k in enumerate(unique_keys)}
        si = np.array([key_index[k] for k in keys], dtype=int)
        if len(strata_cols) == 1:
            strata_levels = tuple(per_col[0][k[0]] for k in unique_keys)
        else:
            strata_levels = tuple(
                tuple(per_col[c][k[c]] for c in range(len(strata_cols))) for k in unique_keys
            )
    else:
        si = np.zeros(len(df), dtype=int)
        strata_levels = ("(all)",)

    counts = np.zeros((len(row_levels), len(col_levels), len(strata_levels)), dtype=float)
    np.add.at(counts, (ri, ci, si), w)

    if row_scores is None:
        row_scores = np.arange(1, len(row_levels) + 1, dtype=float)
    if col_scores is None:
        col_scores = np.arange(1, len(col_levels) + 1, dtype=float)

    return StratifiedTable(
        counts=counts,
        row_levels=tuple(row_levels),
        col_levels=tuple(col_levels),
        strata_levels=strata_levels,
        row_table_scores=row_scores,
        col_table_scores=col_scores,
        notes=tuple(notes),
    )


def _from_array(table: Any) -> StratifiedTable:
    arr = np.asarray(table, dtype=float)
    if arr.ndim == 2:
        arr = arr[:, :, np.newaxis]
    if arr.ndim != 3:
        raise ValueError("table must be a 2-D (R x C) or 3-D (R x C x K) array of counts")
    if not np.all(np.isfinite(arr)):
        raise ValueError("table must contain only finite counts")
    if np.any(arr < 0):
        raise ValueError("table must not contain negative counts")
    R, C, K = arr.shape
    return StratifiedTable(
        counts=arr.copy(),
        row_levels=tuple(range(1, R + 1)),
        col_levels=tuple(range(1, C + 1)),
        strata_levels=tuple(range(1, K + 1)),
        row_table_scores=np.arange(1, R + 1, dtype=float),
        col_table_scores=np.arange(1, C + 1, dtype=float),
    )


def _drop_unused(st: StratifiedTable, warn: bool) -> StratifiedTable:
    """Drop empty strata and levels with a zero total over the whole table.

    SAS never sees such levels because it builds the table from the data.
    Keeping them would make the covariance matrix singular.
    """
    counts = st.counts
    notes = list(st.notes)
    keep_k = counts.sum(axis=(0, 1)) > 0
    keep_r = counts.sum(axis=(1, 2)) > 0
    keep_c = counts.sum(axis=(0, 2)) > 0
    if not keep_k.any():
        raise ValueError("the table contains no observations")
    dropped = []
    if (~keep_k).any():
        dropped.append(f"{int((~keep_k).sum())} empty stratum/strata")
    if (~keep_r).any():
        dropped.append(f"{int((~keep_r).sum())} row level(s) with zero total")
    if (~keep_c).any():
        dropped.append(f"{int((~keep_c).sum())} column level(s) with zero total")
    if dropped:
        msg = "dropped " + ", ".join(dropped)
        notes.append(msg)
        if warn and ((~keep_r).any() or (~keep_c).any()):
            warnings.warn(msg + " (SAS never sees unobserved levels)", stacklevel=4)
    counts = counts[np.ix_(keep_r, keep_c, keep_k)]
    return StratifiedTable(
        counts=counts,
        row_levels=tuple(v for v, k in zip(st.row_levels, keep_r) if k),
        col_levels=tuple(v for v, k in zip(st.col_levels, keep_c) if k),
        strata_levels=tuple(v for v, k in zip(st.strata_levels, keep_k) if k),
        row_table_scores=st.row_table_scores[keep_r],
        col_table_scores=st.col_table_scores[keep_c],
        notes=tuple(notes),
    )


def as_stratified_table(
    table: Any = None,
    *,
    data: Any = None,
    x: str | None = None,
    y: str | None = None,
    strata: str | Sequence[str] | None = None,
    weight: str | None = None,
    order: str = "internal",
    row_scores: Sequence[float] | None = None,
    col_scores: Sequence[float] | None = None,
) -> StratifiedTable:
    """Build a :class:`StratifiedTable` from an array or from a DataFrame."""
    if (table is None) == (data is None):
        raise ValueError("pass either `table` (array of counts) or `data` (DataFrame), not both")
    if data is not None:
        if x is None or y is None:
            raise ValueError("`x` and `y` column names are required with `data`")
        st = _from_dataframe(data, x, y, strata, weight, order)
        from_array = False
    else:
        if any(v is not None for v in (x, y, strata, weight)):
            raise ValueError("`x`, `y`, `strata` and `weight` only apply with `data`")
        st = _from_array(table)
        from_array = True

    if row_scores is not None or col_scores is not None:
        rs = st.row_table_scores if row_scores is None else np.asarray(row_scores, dtype=float)
        cs = st.col_table_scores if col_scores is None else np.asarray(col_scores, dtype=float)
        if rs.shape != (len(st.row_levels),):
            raise ValueError(f"row_scores must have length {len(st.row_levels)}")
        if cs.shape != (len(st.col_levels),):
            raise ValueError(f"col_scores must have length {len(st.col_levels)}")
        if not (np.all(np.isfinite(rs)) and np.all(np.isfinite(cs))):
            raise ValueError("scores must be finite")
        st = StratifiedTable(
            counts=st.counts,
            row_levels=st.row_levels,
            col_levels=st.col_levels,
            strata_levels=st.strata_levels,
            row_table_scores=rs,
            col_table_scores=cs,
            notes=st.notes,
        )

    st = _drop_unused(st, warn=from_array)
    R, C, _ = st.counts.shape
    if R < 2 or C < 2:
        raise ValueError(
            f"need at least 2 observed row levels and 2 observed column levels, got {R} x {C}"
        )
    return st
