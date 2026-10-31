# Changelog

## 0.1.0 — 2026-10-31

First release.

- `cmh_test()`: SAS PROC FREQ CMH statistics (nonzero correlation, row mean scores, general association) for R x C x K tables, with table, rank, ridit and modified ridit scores.
- 2 x 2 x K output: Mantel-Haenszel and logit common odds ratio and relative risks (columns 1 and 2) with confidence limits, Breslow-Day and Breslow-Day-Tarone tests, Mantel-Fleiss criterion.
- `mantelhaen_test()`: port of R's `stats::mantelhaen.test` (asymptotic), including the continuity correction and one-sided alternatives.
- DataFrame input with SAS conventions for level order, table scores, weights, missing values and multiple strata variables.
- Validation suite against published SAS output, R (`mantelhaen.test`, `vcdExtra`, `coin`, `epiR`), SciPy and statsmodels.
