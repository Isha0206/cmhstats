import json
import math
from pathlib import Path

import numpy as np
import pytest

FIXTURES = Path(__file__).parent / "fixtures"

# R writes 15 significant digits, so differences beyond ~1e-12 are noise.
REL_TOL = 1e-8
ABS_TOL = 1e-10


def _num(v):
    if isinstance(v, str):  # jsonlite writes non-finite numbers as strings
        return {"NaN": math.nan, "NA": math.nan, "Inf": math.inf, "-Inf": -math.inf}[v]
    return v


def close(a, b, rel=REL_TOL, abs_=ABS_TOL):
    if a is None or b is None:
        return a is b
    a, b = _num(a), _num(b)
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    if isinstance(a, float) and math.isinf(a) and isinstance(b, float) and math.isinf(b):
        return (a > 0) == (b > 0)
    return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_)


def r_array(counts, dim):
    """Rebuild an R array (column-major) in NumPy."""
    return np.asarray(counts, dtype=float).reshape(dim, order="F")


@pytest.fixture(scope="session")
def r_reference():
    with open(FIXTURES / "r_reference.json") as fh:
        return json.load(fh)


@pytest.fixture(scope="session")
def r_camis():
    with open(FIXTURES / "r_camis.json") as fh:
        return json.load(fh)


@pytest.fixture(scope="session")
def adcibc():
    pd = pytest.importorskip("pandas")
    return pd.read_csv(FIXTURES / "adcibc.csv")
