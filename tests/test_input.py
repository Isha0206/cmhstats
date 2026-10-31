"""DataFrame handling, SAS conventions for levels and scores, and input errors."""

import numpy as np
import pytest

import cmhstats

pytestmark = pytest.mark.filterwarnings("ignore:Mantel-Fleiss criterion")

pd = pytest.importorskip("pandas")


@pytest.fixture
def long_df():
    rng = np.random.default_rng(7)
    n = 300
    return pd.DataFrame({
        "trt": rng.choice(["Placebo", "Low", "High"], n),
        "resp": rng.choice([0, 1, 2, 4], n),          # numeric, not evenly spaced
        "site": rng.choice(["S1", "S2", "S3", "S4"], n),
        "sex": rng.choice(["F", "M"], n),
    })


def crosstab3(df, x, y, k):
    rows = sorted(df[x].unique())
    cols = sorted(df[y].unique())
    strata = sorted(df[k].unique())
    t = np.zeros((len(rows), len(cols), len(strata)))
    for (a, b, c), n in df.groupby([x, y, k]).size().items():
        t[rows.index(a), cols.index(b), strata.index(c)] = n
    return t, cols


def test_dataframe_matches_array_with_numeric_table_scores(long_df):
    t, cols = crosstab3(long_df, "trt", "resp", "site")
    a = cmhstats.cmh_test(data=long_df, x="trt", y="resp", strata="site")
    b = cmhstats.cmh_test(t, col_scores=cols)
    for s1, s2 in zip(a.statistics, b.statistics):
        assert np.isclose(s1.statistic, s2.statistic, rtol=1e-12)
    assert list(a.col_scores) == [0, 1, 2, 4]          # SAS: numeric values are the scores
    assert a.row_levels == ("High", "Low", "Placebo")  # SAS ORDER=INTERNAL
    # positions 1..4 give a different correlation statistic when values are unevenly spaced
    c = cmhstats.cmh_test(t)
    assert not np.isclose(a.correlation.statistic, c.correlation.statistic)


def test_weights_equal_expanded_rows(long_df):
    agg = long_df.groupby(["trt", "resp", "site"]).size().rename("n").reset_index()
    a = cmhstats.cmh_test(data=long_df, x="trt", y="resp", strata="site")
    b = cmhstats.cmh_test(data=agg, x="trt", y="resp", strata="site", weight="n")
    for s1, s2 in zip(a.statistics, b.statistics):
        assert np.isclose(s1.statistic, s2.statistic, rtol=1e-12)


def test_multiple_strata_columns_form_combinations(long_df):
    combo = long_df.assign(k=long_df.site + "_" + long_df.sex)
    a = cmhstats.cmh_test(data=long_df, x="trt", y="resp", strata=["site", "sex"])
    b = cmhstats.cmh_test(data=combo, x="trt", y="resp", strata="k")
    assert len(a.strata_levels) == 8 and a.strata_levels[0] == ("S1", "F")
    assert np.isclose(a.general_association.statistic, b.general_association.statistic)


def test_missing_values_and_zero_weights_are_excluded(long_df):
    df = long_df.assign(w=1.0)
    df.loc[:9, "trt"] = None
    df.loc[10:14, "w"] = 0.0
    res = cmhstats.cmh_test(data=df, x="trt", y="resp", strata="site", weight="w")
    assert res.n_total == len(df) - 15
    assert any("missing" in n for n in res.notes) and any("zero weight" in n for n in res.notes)


def test_ordered_categorical_keeps_category_order(long_df):
    df = long_df.assign(trt=pd.Categorical(long_df.trt, ["Placebo", "Low", "High"], ordered=True))
    res = cmhstats.cmh_test(data=df, x="trt", y="resp", strata="site")
    assert res.row_levels == ("Placebo", "Low", "High")


def test_order_data_uses_first_appearance():
    df = pd.DataFrame({"x": ["b", "a", "b", "a"], "y": ["z", "y", "y", "z"]})
    res = cmhstats.cmh_test(data=df, x="x", y="y", order="data")
    assert res.row_levels == ("b", "a") and res.col_levels == ("z", "y")


def test_string_levels_sort_like_sas_not_like_r_locale():
    # SAS sorts by code point: "65-80" < "<65" < ">80". R's default locale sort
    # ignores punctuation and puts "<65" first, which changes the table scores.
    df = pd.DataFrame({"x": ["A", "B"] * 3, "age": ["<65", "65-80", ">80"] * 2})
    res = cmhstats.cmh_test(data=df, x="x", y="age")
    assert res.col_levels == ("65-80", "<65", ">80")


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(table=np.ones((2, 2, 2, 2))), "2-D"),
        (dict(table=-np.ones((2, 2, 2))), "negative"),
        (dict(table=np.ones((1, 3, 2))), "at least 2"),
        (dict(table=np.ones((2, 2, 2)), scores="wilcoxon"), "scores must be"),
        (dict(table=np.ones((2, 2, 2)), scores="rank", row_scores=[1, 2]), "custom"),
        (dict(table=np.ones((3, 2, 2)), row_scores=[1, 2]), "length 3"),
    ],
)
def test_bad_input_is_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        cmhstats.cmh_test(**kwargs)


def test_unobserved_levels_in_arrays_are_dropped_with_warning():
    t = np.ones((3, 2, 2))
    t[1] = 0
    with pytest.warns(UserWarning, match="row level"):
        res = cmhstats.cmh_test(t)
    assert res.row_mean_scores.df == 1


def test_summary_and_frames(long_df):
    res = cmhstats.cmh_test(data=long_df, x="trt", y="resp", strata="site", scores="modridit")
    text = res.summary()
    assert "Based on Modified Ridit Scores" in text and "General Association" in text
    frame = res.to_frame()
    assert list(frame["DF"]) == [1, 2, 6]
    two = cmhstats.cmh_test(np.array([[[16, 12], [11, 16]], [[5, 7], [20, 19]]]))
    assert two.estimates_frame().shape == (6, 5)
    with pytest.raises(ValueError):
        res.estimates_frame()


def test_mantelhaen_test_mirrors_r_errors():
    with pytest.raises(ValueError, match="at least 2 strata"):
        cmhstats.mantelhaen_test(np.ones((2, 2, 1)))
    t = np.ones((2, 2, 3))
    t[:, :, 0] = [[1, 0], [0, 0]]
    with pytest.raises(ValueError, match="sample size in each stratum"):
        cmhstats.mantelhaen_test(t)
