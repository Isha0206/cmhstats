"""cmhstats: Cochran-Mantel-Haenszel statistics for Python, validated against SAS and R.

>>> import numpy as np, cmhstats
>>> migraine = np.array([[[16, 12], [11, 16]], [[5, 7], [20, 19]]])  # Treatment x Response x Gender
>>> res = cmhstats.cmh_test(migraine)
>>> round(res.general_association.statistic, 4)
8.3052
"""

from ._api import CMHResult, CMHStatistic, MantelHaenszelResult, cmh_test, mantelhaen_test
from ._input import StratifiedTable, as_stratified_table
from ._twobytwo import Estimate, HomogeneityTest, breslow_day

__version__ = "0.1.0"

__all__ = [
    "CMHResult",
    "CMHStatistic",
    "Estimate",
    "HomogeneityTest",
    "MantelHaenszelResult",
    "StratifiedTable",
    "as_stratified_table",
    "breslow_day",
    "cmh_test",
    "mantelhaen_test",
    "__version__",
]
