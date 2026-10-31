# Validation report — cmhstats 0.1.0

Generated 31 October 2026. Python 3.13, NumPy 2.5, SciPy 1.18. R 4.3.3 with vcdExtra 0.8.5, coin 1.4.3, epiR 2.0.67.

## Summary

| Reference | What is compared | Cases | Agreement |
|---|---|---|---|
| Published SAS output | CMH statistics, odds ratios, relative risks, Breslow-Day | 3 sources, 8 analyses | All printed digits (4 decimals) |
| `vcdExtra` formulas in R | 3 statistics x 4 score types | 142 tables | Relative 1e-8 |
| `vcdExtra::CMHtest` | 3 statistics, table and modridit scores | 96 tables it can compute | Relative 1e-8 |
| `stats::mantelhaen.test` | General association; 2x2xK MH OR + CI; correction; one-sided | 131 tables | Relative 1e-8 |
| `coin::cmh_test` | General association | 139 tables | Relative 1e-8 |
| `epiR::epi.2by2` | MH OR, MH RR (both columns) + CIs, Breslow-Day | 21 tables | Relative 1e-7 |
| Hand-written R (from SAS formulas) | Logit OR and RR + CIs, Breslow-Day, Tarone | 30 tables | Relative 1e-7 |
| SciPy / statsmodels / definitions | Pearson, Spearman, Kruskal-Wallis, van Elteren, `StratifiedTable` | Random tables | Relative 1e-10 |

The test suite has 1,178 tests: 1,175 pass and 3 are skipped by design (see "Known differences").

## 1. SAS

Each value is compared with SAS's printed output after rounding to 4 decimals. p-values are compared the same way.

**Migraine** (SAS/STAT documentation, "Cochran-Mantel-Haenszel Statistics" example). All three CMH statistics (8.3052, p = 0.0040), the Mantel-Haenszel and logit odds ratios, the column 1 and column 2 relative risks with 95% limits, and the Breslow-Day test (1.4929, p = 0.2218) agree.

**SexualFun**, a 4 x 4 single-stratum table (SAS output posted to R-help by M. Friendly, 2009). Correlation 10.0142, row mean scores 12.5681, general association 16.7689, with their p-values, agree.

**CAMIS scenarios 1, 2, 3, 6, 9 and 10** on CDISC pilot ADCIBC (SAS 9.04.01M7, from the CAMIS repository). All 18 statistics agree, with their df and p-values. This includes scenario 9 (3 x 5 x 17, by site), for which CAMIS reports no R result.

## 2. R

`validation/generate_r_reference.R` creates the reference tables with a fixed seed:

* 126 random tables: 14 shapes x 3 sparsity levels (Poisson means 0.8, 3, 15) x 3 replicates. Each stratum has at least 2 observations, so every R function can run.
* 12 tables with one single-observation stratum. R refuses these (`mantelhaen.test`) or returns NaN (`vcdExtra`). The reference is computed on the table without that stratum. `cmhstats` receives the full table and must give the same answer.
* 4 textbook tables: `UCBAdmissions`, `Satisfaction` and the penicillin `Rabbits` data from R's documentation, and `Arthritis` from vcd.

`vcdExtra::CMHtest` returned an error for 46 of the 142 tables, including 38 of the 126 random ones. To cover all of them, the script also evaluates vcdExtra's own formulas pooled across strata, without the per-stratum step that fails (`pooled_cmh_r`). Rank and ridit scores, which vcdExtra does not offer, are computed through the same code. Section 4 explains the failure.

## 3. Independent checks

* One stratum: general association = Pearson X² · (n−1)/n, correlation = (n−1)·r² (Pearson for table scores, Spearman for rank-type scores), and rank row mean scores = Kruskal-Wallis H with tie correction. These are the identities stated in the SAS documentation, checked against `scipy.stats`.
* Van Elteren's test, coded from its definition (within-stratum midranks, weights 1/(n_h+1), tie-corrected variance), equals the row mean scores statistic with modified ridit scores.
* For 2 x 2 x K tables, `statsmodels` `StratifiedTable` agrees on the CMH statistic (with and without correction), MH odds ratio and limits, MH relative risk, and Breslow-Day with and without Tarone's adjustment.
* Invariances: strata order, row/column transposition (general association), affine changes of table scores, and permutation of nominal rows all leave the relevant statistics unchanged.

## 4. Findings

**Why `vcdExtra` fails on sparse tables.** `CMHtest` computes a test for every stratum before pooling, and each one calls `solve()` on that stratum's covariance matrix. A stratum missing a row or column level makes its own matrix singular, and the error ends the whole call even though the pooled matrix is fine. In CAMIS scenario 9 this happens at sites 706 and 707. Site 702 has a single subject, but dropping it does not fix the error. The CAMIS page suggests replacing `solve()` with `MASS::ginv`. Skipping the per-stratum tests is enough, and the pooled result then equals SAS. `mantelhaen.test` also gives the SAS general association value (7.0339) once site 702 is removed.

**PyPI `cmh` 1.0.1.** Its `CMH_numpy` flattens the observed-minus-expected vector in row-major order (`reshape(-1)`) but builds the covariance with `kron(columns, rows)`, which assumes column-major order. The two agree only when both dimensions are 2. On R's `Satisfaction` example (4 x 4 x 2) it returns 12.531 where R, Stata and SAS give 10.200. With column-major flattening it gives 10.200.

**CAMIS scenario 2 degrees of freedom.** The SAS table on the CAMIS page lists df = 1 for row mean scores and general association. The p-value 0.2891 is the chi-square(2) tail of 2.4820 (the chi-square(1) tail is 0.1152), so this is a transcription error.

## 5. Known differences and open questions

* **Singular covariance.** `cmhstats` returns NaN with a warning, as SAS reports a missing value. `coin` uses a generalized inverse with reduced df, and `mantelhaen.test` raises an error. The 3 skipped tests are these cases.
* **Single-observation strata.** These are skipped for the CMH statistics, where they contribute exactly zero. This agrees with SAS on scenarios 3 and 9. For the logit estimators, `cmhstats` follows the SAS rule as documented (add 0.5 to each cell of a stratum with a zero cell), so such strata are included. This edge case has not been confirmed against SAS output.
* **Rank-type scores are computed within each stratum.** This follows SAS's stratum-indexed score matrices and the van Elteren equivalence, and it agrees with vcdExtra's `"midrank"` scores and the van Elteren definition. It has not been confirmed against SAS output with `SCORES=RANK` or `RIDIT` on multi-stratum data. That would be the most useful next check for anyone with SAS access.
* **Breslow-Day with zero margins.** Strata with a zero row or column are excluded, as the SAS documentation states. Cases where the MH odds ratio is 0 or infinite return `None` (SAS does not compute the test).

## Reproducing

```bash
pip install -e ".[test]"
pytest                                   # uses committed fixtures, no R needed
Rscript validation/generate_r_reference.R  # optional: needs R with vcdExtra, coin, epiR, jsonlite
```
