"""Agreement with published SAS PROC FREQ output, at the precision SAS prints.

Sources
-------
* Migraine: SAS/STAT User's Guide, PROC FREQ, Example "Cochran-Mantel-Haenszel
  Statistics" (Output 40.7.2-40.7.4 in SAS/STAT 14.1).
* SexualFun: SAS output posted by Michael Friendly to R-help, 9 Feb 2009
  ("R equivalent of SAS Cochran-Mantel-Haenszel tests?").
* CDISC pilot ADCIBC scenarios: PHUSE CAMIS, "R vs SAS CMH"
  (psiaims.github.io/CAMIS/Comp/r-sas_cmh.html), SAS 9.04.01M7.
"""

import warnings

import numpy as np
import pytest

import cmhstats


def r4(v):
    return round(float(v), 4)


def sas_p(v):
    return "<.0001" if v < 1e-4 else f"{v:.4f}"


# ---------------------------------------------------------------- Migraine
MIGRAINE = np.zeros((2, 2, 2))
MIGRAINE[:, :, 0] = [[16, 11], [5, 20]]  # female: Active / Placebo x Better / Same
MIGRAINE[:, :, 1] = [[12, 16], [7, 19]]  # male


def test_migraine_cmh_statistics():
    res = cmhstats.cmh_test(MIGRAINE)
    for s in res.statistics:
        assert (s.df, r4(s.statistic), sas_p(s.pvalue)) == (1, 8.3052, "0.0040")


@pytest.mark.parametrize(
    "attr, idx, expected",
    [
        ("common_odds_ratio", 0, (3.3132, 1.4456, 7.5934)),
        ("common_odds_ratio", 1, (3.2941, 1.4182, 7.6515)),
        ("relative_risk_col1", 0, (2.1636, 1.2336, 3.7948)),
        ("relative_risk_col1", 1, (2.1059, 1.1951, 3.7108)),
        ("relative_risk_col2", 0, (0.6420, 0.4705, 0.8761)),
        ("relative_risk_col2", 1, (0.6613, 0.4852, 0.9013)),
    ],
)
def test_migraine_common_estimates(attr, idx, expected):
    est = getattr(cmhstats.cmh_test(MIGRAINE), attr)[idx]
    assert (r4(est.value), r4(est.lower), r4(est.upper)) == expected


def test_migraine_breslow_day():
    bd = cmhstats.cmh_test(MIGRAINE).breslow_day
    assert (r4(bd.statistic), bd.df, sas_p(bd.pvalue)) == (1.4929, 1, "0.2218")


def test_migraine_dataframe_input_with_weight():
    pd = pytest.importorskip("pandas")
    rows = [
        ("female", "Active", "Better", 16), ("female", "Active", "Same", 11),
        ("female", "Placebo", "Better", 5), ("female", "Placebo", "Same", 20),
        ("male", "Active", "Better", 12), ("male", "Active", "Same", 16),
        ("male", "Placebo", "Better", 7), ("male", "Placebo", "Same", 19),
    ]
    df = pd.DataFrame(rows, columns=["Gender", "Treatment", "Response", "Count"])
    res = cmhstats.cmh_test(data=df, x="Treatment", y="Response", strata="Gender", weight="Count")
    assert r4(res.general_association.statistic) == 8.3052
    assert r4(res.common_odds_ratio[0].value) == 3.3132
    assert res.row_levels == ("Active", "Placebo")


# ---------------------------------------------------------------- SexualFun
# Husband (rows) x Wife (columns), levels Never Fun, Fairly Often, Very Often, Always Fun
SEXUALFUN = np.array([7, 2, 1, 2, 7, 8, 5, 8, 2, 3, 4, 9, 3, 7, 9, 14], dtype=float).reshape(
    (4, 4), order="F"
)


def test_sexualfun_single_stratum_table_scores():
    res = cmhstats.cmh_test(SEXUALFUN)
    got = [(s.df, r4(s.statistic), sas_p(s.pvalue)) for s in res.statistics]
    assert got == [(1, 10.0142, "0.0016"), (3, 12.5681, "0.0057"), (9, 16.7689, "0.0525")]


# ---------------------------------------------------------------- CAMIS / ADCIBC
# (scenario, X, Y, K, subset) -> SAS (value, df, prob) for correlation, row means, general
CAMIS = {
    1: ("TRTP", "SEX", "AGEGR1", True,
        [(0.2166, 1, 0.6417), (0.2166, 1, 0.6417), (0.2166, 1, 0.6417)]),
    # CAMIS lists df = 1 for the last two rows of scenario 2; the p-value 0.2891 is the
    # chi-square(2) tail of 2.4820 (the chi-square(1) tail would be 0.1152), so df = 2.
    2: ("TRTP", "SEX", "AGEGR1", False,
        [(0.0009, 1, 0.9765), (2.4820, 2, 0.2891), (2.4820, 2, 0.2891)]),
    3: ("TRTP", "SEX", "RACE", False,
        [(0.0028, 1, 0.9579), (2.3861, 2, 0.3033), (2.3861, 2, 0.3033)]),
    6: ("TRTP", "AVAL", "SEX", True,
        [(1.7487, 1, 0.1860), (1.7487, 1, 0.1860), (8.0534, 4, 0.0896)]),
    9: ("TRTP", "AVAL", "SITEID", False,
        [(0.0854, 1, 0.7701), (2.4763, 2, 0.2899), (7.0339, 8, 0.5330)]),
    10: ("AVAL", "AGEGR1N", "TRTP", False,
         [(1.6621, 1, 0.1973), (2.2980, 4, 0.6811), (5.7305, 8, 0.6774)]),
}


@pytest.mark.parametrize("scenario", sorted(CAMIS))
def test_camis_scenarios_match_sas(scenario, adcibc):
    x, y, k, subset, expected = CAMIS[scenario]
    df = adcibc[(adcibc.TRTPN != 54) & (adcibc.AGEGR1 != ">80")] if subset else adcibc
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = cmhstats.cmh_test(data=df, x=x, y=y, strata=k)
    got = [(r4(s.statistic), s.df, r4(s.pvalue)) for s in res.statistics]
    assert got == expected


def test_camis_scenario2_df_typo_is_really_a_typo():
    from scipy import stats

    assert r4(stats.chi2.sf(2.4820, 2)) == 0.2891
    assert r4(stats.chi2.sf(2.4820, 1)) != 0.2891


def test_camis_scenario9_explained(adcibc, r_camis):
    """vcdExtra cannot run scenario 9; the R reference shows why, and that the
    pooled formula in R gives the SAS numbers once the failing per-stratum
    step is skipped."""
    assert "singular" in r_camis["scenario9_vcdExtra_full_table"]
    assert "singular" in r_camis["scenario9_vcdExtra_without_single_obs_sites"]
    pooled = r_camis["scenario9_pooled_formula_in_R"]
    assert [r4(pooled[k]["chisq"]) for k in ("cor", "rmeans", "general")] == [0.0854, 2.4763, 7.0339]
    res = cmhstats.cmh_test(data=adcibc, x="TRTP", y="AVAL", strata="SITEID")
    assert res.strata_excluded == (702,)
    assert np.isclose(res.row_mean_scores.statistic, pooled["rmeans"]["chisq"], rtol=1e-9)
