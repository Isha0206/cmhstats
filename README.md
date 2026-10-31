# cmhstats

[![tests](https://github.com/Isha0206/cmhstats/actions/workflows/tests.yml/badge.svg)](https://github.com/Isha0206/cmhstats/actions/workflows/tests.yml)

Cochran-Mantel-Haenszel (CMH) statistics for Python that reproduce SAS `PROC FREQ / CMH` and R's `mantelhaen.test`, with the equivalence tests to show it.

Python's existing options cover parts of this. `statsmodels.stats.contingency_tables.StratifiedTable` handles 2 x 2 x K tables well. What was missing is the generalized R x C x K analysis that clinical efficacy tables use: the three SAS statistics (nonzero correlation, row mean scores, general association) with table, rank, ridit or modified ridit scores, plus the full set of 2 x 2 x K estimates in one place. The [PHUSE CAMIS](https://psiaims.github.io/CAMIS/) comparison project documents CMH for R and SAS but has no Python entry.

## What it does

```python
import cmhstats

# long-format data, as you would give PROC FREQ: tables SITEID * TRTP * AVAL / cmh;
res = cmhstats.cmh_test(data=adcibc, x="TRTP", y="AVAL", strata="SITEID")
print(res)
```

```
Cochran-Mantel-Haenszel Statistics (Based on Table Scores)

Statistic Alternative Hypothesis      DF       Value      Prob
--------------------------------------------------------------
1         Nonzero Correlation          1      0.0854    0.7701
2         Row Mean Scores Differ       2      2.4763    0.2899
3         General Association          8      7.0339    0.5330

Total Sample Size = 231
Strata with fewer than 2 observations (no information): 1
```

These are the SAS numbers for CAMIS scenario 9, the sparse 17-site case that `vcdExtra::CMHtest` cannot compute.

| Feature | SAS `PROC FREQ` option | `cmhstats` |
|---|---|---|
| Correlation, row mean scores, general association statistics | `CMH` | `cmh_test()` |
| Table, rank, ridit, modified ridit scores (within strata) | `SCORES=` | `scores=` |
| Van Elteren test | `CMH2 SCORES=MODRIDIT` | `scores="modridit"`, `.row_mean_scores` |
| Mantel-Haenszel and logit common odds ratio, with CIs | `CMH` (2 x 2 x K) | `.common_odds_ratio` |
| Mantel-Haenszel and logit relative risks, columns 1 and 2 | `CMH` (2 x 2 x K) | `.relative_risk_col1`, `.relative_risk_col2` |
| Breslow-Day test, Tarone adjustment | `CMH`, `BDT` | `.breslow_day`, `.breslow_day_tarone` |
| Mantel-Fleiss criterion warning | `CMH` | warning + `.mantel_fleiss` |
| WEIGHT variable, multiple strata variables | `WEIGHT`, `tables A*B*X*Y` | `weight=`, `strata=[...]` |
| R's `mantelhaen.test`, including its default continuity correction | — | `mantelhaen_test()` |

## Install

```bash
pip install cmhstats
pip install "cmhstats[pandas]"   # with DataFrame support
```

Requires Python 3.9+ (tested on 3.9 and 3.13), NumPy and SciPy. pandas is needed only for DataFrame input.

## Usage

**Array input.** Shape `(R, C, K)`: rows are X, columns are Y, the last axis is the strata. This is SAS's `tables K*X*Y` and R's `mantelhaen.test(x)` layout.

```python
import numpy as np, cmhstats

# SAS documentation example: Treatment x Response, stratified by Gender
migraine = np.zeros((2, 2, 2))
migraine[:, :, 0] = [[16, 11], [5, 20]]   # female
migraine[:, :, 1] = [[12, 16], [7, 19]]   # male

res = cmhstats.cmh_test(migraine)
res.general_association        # CMHStatistic(statistic=8.3052..., df=1, pvalue=0.00395...)
res.common_odds_ratio[0]       # Mantel-Haenszel 3.3132 (1.4456, 7.5934)
res.breslow_day                # 1.4929, df 1, p 0.2218
res.to_frame()                 # SAS-layout DataFrame
```

**DataFrame input.** Level ordering and table scores follow SAS: levels are sorted by value (`ORDER=INTERNAL`), numeric variables use their values as table scores, and character variables use positions 1, 2, 3, ... Observations with missing values or zero weight are excluded.

```python
res = cmhstats.cmh_test(data=df, x="TRTP", y="AVAL", strata=["SITEGR1", "SEX"],
                        weight="COUNT", scores="modridit")
```

**Matching R instead of SAS.** `mantelhaen_test()` is a port of `stats::mantelhaen.test`. For 2 x 2 x K tables it applies R's continuity correction by default, which SAS does not.

```python
cmhstats.mantelhaen_test(migraine)                  # R default: corrected, 7.1983
cmhstats.mantelhaen_test(migraine, correct=False)   # = SAS: 8.3052
```

## Validation

All 1,178 tests run without R or SAS. The R reference results are committed under `tests/fixtures/`, and `validation/generate_r_reference.R` regenerates them. To run them from a clone:

```bash
pip install -e ".[test]"
pytest
```

* **SAS:** every number in the published output for three sources agrees at the 4 decimals SAS prints: the SAS documentation's Migraine example (CMH statistics, all six common odds ratio and relative risk estimates with CIs, Breslow-Day), the SexualFun 4 x 4 example, and all six CAMIS scenarios on the CDISC pilot ADCIBC data.
* **R:** 142 tables in 25 shapes, from 2 x 2 x 2 to 6 x 2 x 10, including sparse ones and ones with single-observation strata. All three statistics agree with vcdExtra's formulas for all four score types, and with `mantelhaen.test`, `vcdExtra::CMHtest` and `coin::cmh_test` wherever those run, to a relative tolerance of 1e-8. The 2 x 2 x K estimates and Breslow-Day test agree with `epiR::epi.2by2` to 1e-7.
* **Independent identities:** Pearson chi-square, Spearman correlation and Kruskal-Wallis (single stratum), van Elteren's test written from its definition, and `statsmodels` for 2 x 2 x K.

See [docs/validation.md](https://github.com/Isha0206/cmhstats/blob/main/docs/validation.md) for the full report, including what is *not* yet confirmed against SAS directly.

## Findings along the way

* **`vcdExtra::CMHtest` fails on sparse data because of per-stratum tests, not because the pooled statistic is singular.** It inverts each stratum's own covariance before pooling, so one site that lacks a treatment arm stops the analysis. In CAMIS scenario 9, sites 706 and 707 trigger this, and removing the single-subject site 702 does not help. Using vcdExtra's own formulas without the per-stratum step reproduces SAS exactly. vcdExtra failed on 46 of the 142 reference tables.
* **The `cmh` package on PyPI (1.0.1) returns wrong values for tables larger than 2 x 2.** It flattens the observed-minus-expected vector in row-major order but builds the covariance for column-major order. On the `Satisfaction` example from R's documentation it reports 12.53 instead of 10.20.
* **CAMIS lists df = 1 for the row mean scores and general association statistics in scenario 2.** The p-value printed alongside (0.2891) is the chi-square(2) tail of 2.4820, so df = 2 is correct.
* **SAS and R sort character levels differently.** SAS puts `"65-80"` before `"<65"`, while R's locale-aware sort does the opposite. With table scores, this changes the correlation and row mean scores statistics.

## Scope and limitations

Not implemented yet: exact tests (R's `exact=TRUE`, SAS's Zelen and exact common odds ratio limits), common risk differences (`COMMONRISKDIFF`), and SAS's formatted-value ordering (`ORDER=FORMATTED`). This is an independent open-source implementation that has not been qualified for regulated use. A risk-based assessment would still be needed before using it in a submission.

## License

MIT
