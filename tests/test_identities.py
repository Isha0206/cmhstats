"""Checks against independent implementations and known mathematical identities."""

import warnings

import numpy as np
import pytest
from scipy import stats

import cmhstats

pytestmark = pytest.mark.filterwarnings("ignore:Mantel-Fleiss criterion")

rng = np.random.default_rng(20261031)


def random_table(shape, lam=4.0):
    while True:
        t = rng.poisson(lam, size=shape).astype(float)
        if (t.sum(axis=(1, 2)) > 0).all() and (t.sum(axis=(0, 2)) > 0).all() and (t.sum(axis=(0, 1)) >= 2).all():
            return t


def expand(table2d):
    """Individual-level (row, col) pairs from a 2-D table of counts."""
    R, C = table2d.shape
    rr, cc = np.meshgrid(np.arange(R), np.arange(C), indexing="ij")
    reps = table2d.ravel().astype(int)
    return np.repeat(rr.ravel(), reps), np.repeat(cc.ravel(), reps)


# ------------------------------------------------- single-stratum identities (SAS docs)
@pytest.mark.parametrize("shape", [(2, 2), (3, 4), (5, 3), (4, 6)])
def test_single_stratum_general_association_is_scaled_pearson(shape):
    t = random_table(shape + (1,))[:, :, 0]
    n = t.sum()
    pearson = stats.chi2_contingency(t, correction=False)[0]
    res = cmhstats.cmh_test(t)
    assert np.isclose(res.general_association.statistic, pearson * (n - 1) / n, rtol=1e-12)


@pytest.mark.parametrize("shape", [(3, 4), (5, 3), (4, 6)])
def test_single_stratum_correlation_is_n_minus_1_r_squared(shape):
    t = random_table(shape + (1,))[:, :, 0]
    n = t.sum()
    r, c = expand(t)
    assert np.isclose(cmhstats.cmh_test(t).correlation.statistic,
                      (n - 1) * stats.pearsonr(r, c).statistic ** 2, rtol=1e-10)
    for scores in ("rank", "ridit", "modridit"):
        assert np.isclose(cmhstats.cmh_test(t, scores=scores).correlation.statistic,
                          (n - 1) * stats.spearmanr(r, c).statistic ** 2, rtol=1e-10)


@pytest.mark.parametrize("shape", [(2, 5), (3, 4), (4, 6)])
def test_single_stratum_rank_row_means_is_kruskal_wallis(shape):
    t = random_table(shape + (1,))[:, :, 0]
    r, c = expand(t)
    kw = stats.kruskal(*[c[r == i] for i in range(shape[0])]).statistic
    assert np.isclose(cmhstats.cmh_test(t, scores="rank").row_mean_scores.statistic, kw, rtol=1e-10)


def van_elteren(table):
    """Van Elteren's stratified Wilcoxon test written from its definition."""
    num, var = 0.0, 0.0
    for h in range(table.shape[2]):
        t = table[:, :, h]
        N = t.sum()
        if N < 2:
            continue
        g, y = expand(t)
        ranks = stats.rankdata(y)
        n1 = (g == 0).sum()
        n2 = N - n1
        W = ranks[g == 0].sum()
        ties = np.unique(y, return_counts=True)[1]
        v = n1 * n2 / 12.0 * ((N + 1) - (ties**3 - ties).sum() / (N * (N - 1)))
        num += (W - n1 * (N + 1) / 2) / (N + 1)
        var += v / (N + 1) ** 2
    return num**2 / var


@pytest.mark.parametrize("K", [2, 5, 9])
def test_modridit_row_means_is_van_elteren(K):
    t = random_table((2, 5, K), lam=3)
    res = cmhstats.cmh_test(t, scores="modridit")
    assert np.isclose(res.row_mean_scores.statistic, van_elteren(t), rtol=1e-10)


# ------------------------------------------------------------------ statsmodels
@pytest.mark.parametrize("K", [2, 4, 8])
def test_two_by_two_matches_statsmodels(K):
    ct = pytest.importorskip("statsmodels.stats.contingency_tables")
    t = random_table((2, 2, K), lam=8)
    st = ct.StratifiedTable([t[:, :, h] for h in range(K)])
    res = cmhstats.cmh_test(t)
    assert np.isclose(res.general_association.statistic, st.test_null_odds(correction=False).statistic)
    mh = res.common_odds_ratio[0]
    assert np.isclose(mh.value, st.oddsratio_pooled)
    lo, hi = st.oddsratio_pooled_confint()
    assert np.isclose(mh.lower, lo) and np.isclose(mh.upper, hi)
    assert np.isclose(res.relative_risk_col1[0].value, st.riskratio_pooled)
    assert np.isclose(res.breslow_day.statistic, st.test_equal_odds(adjust=False).statistic)
    assert np.isclose(res.breslow_day_tarone.statistic, st.test_equal_odds(adjust=True).statistic)
    corrected = cmhstats.mantelhaen_test(t, correct=True)
    assert np.isclose(corrected.statistic, st.test_null_odds(correction=True).statistic)


# ------------------------------------------------------------------ invariances
def test_order_of_strata_does_not_matter():
    t = random_table((3, 4, 6))
    perm = rng.permutation(6)
    a, b = cmhstats.cmh_test(t), cmhstats.cmh_test(t[:, :, perm])
    for s1, s2 in zip(a.statistics, b.statistics):
        assert np.isclose(s1.statistic, s2.statistic, rtol=1e-12)


def test_general_association_is_symmetric_in_rows_and_columns():
    t = random_table((3, 5, 4))
    a = cmhstats.cmh_test(t).general_association.statistic
    b = cmhstats.cmh_test(t.transpose(1, 0, 2)).general_association.statistic
    assert np.isclose(a, b, rtol=1e-12)


def test_affine_change_of_table_scores_changes_nothing():
    t = random_table((3, 4, 5))
    a = cmhstats.cmh_test(t)
    b = cmhstats.cmh_test(t, row_scores=[10, 20, 30], col_scores=[-1, 1.5, 4, 6.5])
    for s1, s2 in zip(a.statistics, b.statistics):
        assert np.isclose(s1.statistic, s2.statistic, rtol=1e-10)


def test_nominal_row_order_does_not_change_row_means_or_general():
    t = random_table((4, 3, 3))
    a = cmhstats.cmh_test(t)
    b = cmhstats.cmh_test(t[[2, 0, 3, 1], :, :])
    assert np.isclose(a.row_mean_scores.statistic, b.row_mean_scores.statistic, rtol=1e-10)
    assert np.isclose(a.general_association.statistic, b.general_association.statistic, rtol=1e-10)


def test_pvalues_keep_precision_in_the_far_tail():
    t = np.zeros((2, 2, 3))
    for h in range(3):
        t[:, :, h] = [[60, 20], [20, 60]]
    res = cmhstats.cmh_test(t).general_association
    assert 0 < res.pvalue < 1e-20                      # survival function keeps the tail
    assert 1 - stats.chi2.cdf(res.statistic, 1) == 0.0  # what 1 - cdf() would report


def test_mantel_fleiss_warning_for_small_tables():
    t = np.zeros((2, 2, 2))
    t[:, :, 0] = [[1, 0], [0, 1]]
    t[:, :, 1] = [[1, 1], [0, 1]]
    with pytest.warns(UserWarning, match="Mantel-Fleiss"):
        res = cmhstats.cmh_test(t)
    assert res.mantel_fleiss < 5


def test_singular_covariance_gives_nan_and_warning():
    # column 2 only occurs in a stratum whose second row is empty -> no information
    t = np.zeros((2, 3, 2))
    t[:, :, 0] = [[3, 0, 2], [1, 0, 4]]
    t[:, :, 1] = [[2, 1, 1], [0, 0, 0]]
    with pytest.warns(UserWarning, match="singular"):
        res = cmhstats.cmh_test(t)
    assert np.isnan(res.general_association.statistic)
    assert np.isfinite(res.correlation.statistic)
