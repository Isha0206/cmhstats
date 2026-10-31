# Generate R reference results for the cmhstats equivalence tests.
#
# Run from the repository root:
#   Rscript validation/generate_r_reference.R
#
# Writes tests/fixtures/r_reference.json and tests/fixtures/r_camis.json.
# The fixtures are committed so the Python test suite does not need R.
#
# References used (all independent of cmhstats):
#   stats::mantelhaen.test   general association, 2x2xK CMH (+/- correction), MH OR + CI
#   vcdExtra::CMHtest         correlation / row mean / general association, table and
#                             "midrank" (= SAS MODRIDIT) scores, computed per stratum
#   vcdExtra internals        per-stratum RANK and RIDIT scores (CMHtest2 + CMHtest3)
#   coin::cmh_test            general association (permutation framework)
#   epiR::epi.2by2            MH odds ratio / relative risk + CI, Breslow-Day test
#   hand-written R            logit (Woolf) estimators and Tarone's adjustment, written
#                             from the SAS documentation formulas

suppressPackageStartupMessages({
  library(vcdExtra)
  library(coin)
  library(epiR)
  library(jsonlite)
})

safe <- function(expr) tryCatch(suppressWarnings(expr), error = function(e) NULL)

info <- list(
  R = R.version.string,
  vcdExtra = as.character(packageVersion("vcdExtra")),
  coin = as.character(packageVersion("coin")),
  epiR = as.character(packageVersion("epiR")),
  generated = format(Sys.time(), "%Y-%m-%d %H:%M:%S %Z")
)

cmh_rows <- function(tab) {
  if (is.null(tab)) return(NULL)
  out <- list()
  for (nm in c("cor", "rmeans", "general")) {
    if (nm %in% rownames(tab)) {
      out[[nm]] <- list(chisq = unname(unlist(tab[nm, "Chisq"])),
                        df = unname(unlist(tab[nm, "Df"])),
                        p = unname(unlist(tab[nm, "Prob"])))
    }
  }
  out
}

midrank_scores <- function(margin, kind) {
  cs <- cumsum(margin)
  rank <- cs - margin + (margin + 1) / 2
  n <- sum(margin)
  switch(kind, rank = rank, ridit = rank / n, modridit = rank / (n + 1))
}

# Pooled CMH statistics using vcdExtra's own formulas (CMHtest2 / CMHtest3), but
# without the per-stratum test that vcdExtra computes first. vcdExtra calls
# solve() on every stratum's own covariance matrix, so a single stratum with a
# missing treatment arm or outcome level aborts the whole analysis even though
# only the pooled matrix needs to be inverted. Strata with n < 2 are skipped.
pooled_cmh_r <- function(x, kind = "table", rtab = seq_len(dim(x)[1]), ctab = seq_len(dim(x)[2])) {
  lkronecker <- function(a, b) kronecker(b, a)
  R <- dim(x)[1]; C <- dim(x)[2]; K <- dim(x)[3]
  A_of <- function(rsc, csc) list(
    cor = lkronecker(t(rsc), t(csc)),
    rmeans = lkronecker(cbind(diag(R - 1), rep(0, R - 1)), t(csc)),
    general = lkronecker(cbind(diag(R - 1), rep(0, R - 1)), cbind(diag(C - 1), rep(0, C - 1))))
  df <- c(cor = 1, rmeans = R - 1, general = (R - 1) * (C - 1))
  AVA <- list(); Anm <- list()
  for (h in seq_len(K)) {
    xh <- x[, , h]; nt <- sum(xh)
    if (nt < 2) next
    pr <- rowSums(xh) / nt; pc <- colSums(xh) / nt
    m <- as.vector(nt * outer(pr, pc)); n <- as.vector(xh)
    V <- (nt^2 / (nt - 1)) * lkronecker(diag(pr) - pr %*% t(pr), diag(pc) - pc %*% t(pc))
    rsc <- if (kind == "table") rtab else midrank_scores(rowSums(xh), kind)
    csc <- if (kind == "table") ctab else midrank_scores(colSums(xh), kind)
    A <- A_of(rsc, csc)
    for (nm in names(A)) {
      AVA[[nm]] <- (if (is.null(AVA[[nm]])) 0 else AVA[[nm]]) + A[[nm]] %*% V %*% t(A[[nm]])
      Anm[[nm]] <- (if (is.null(Anm[[nm]])) 0 else Anm[[nm]]) + A[[nm]] %*% (n - m)
    }
  }
  out <- list()
  for (nm in names(df)) {
    # a singular pooled matrix means the statistic is undefined (SAS: missing)
    Q <- tryCatch(as.numeric(t(Anm[[nm]]) %*% solve(AVA[[nm]]) %*% Anm[[nm]]),
                  error = function(e) NaN)
    out[[nm]] <- list(chisq = Q, df = unname(df[nm]), p = pchisq(Q, df[nm], lower.tail = FALSE))
  }
  out
}

# Woolf / logit estimators, written from the SAS PROC FREQ formulas
logit_pooled <- function(logest, v, conf = 0.95) {
  w <- 1 / v
  est <- exp(sum(w * logest) / sum(w))
  z <- qnorm(1 - (1 - conf) / 2)
  c(est, est * exp(-z / sqrt(sum(w))), est * exp(z / sqrt(sum(w))))
}
logit_or <- function(x) {
  a <- x[1, 1, ]; b <- x[1, 2, ]; c <- x[2, 1, ]; d <- x[2, 2, ]
  z <- (a == 0) | (b == 0) | (c == 0) | (d == 0)
  a[z] <- a[z] + .5; b[z] <- b[z] + .5; c[z] <- c[z] + .5; d[z] <- d[z] + .5
  logit_pooled(log(a * d / (b * c)), 1 / a + 1 / b + 1 / c + 1 / d)
}
logit_rr <- function(x, col) {
  a <- x[1, 1, ]; b <- x[1, 2, ]; c <- x[2, 1, ]; d <- x[2, 2, ]
  ev1 <- if (col == 1) a else b; ev2 <- if (col == 1) c else d
  z <- (ev1 == 0) | (ev2 == 0)
  a[z] <- a[z] + .5; b[z] <- b[z] + .5; c[z] <- c[z] + .5; d[z] <- d[z] + .5
  ev1 <- if (col == 1) a else b; ev2 <- if (col == 1) c else d
  p1 <- ev1 / (a + b); p2 <- ev2 / (c + d)
  logit_pooled(log(p1 / p2), (1 - p1) / ev1 + (1 - p2) / ev2)
}

# Breslow-Day with optional Tarone adjustment (follows DescTools::BreslowDayTest)
bd_test <- function(x, tarone) {
  or_mh <- mantelhaen.test(x, correct = FALSE)$estimate
  K <- dim(x)[3]
  X2 <- 0; dev <- 0; vsum <- 0; used <- 0
  for (h in seq_len(K)) {
    t <- x[, , h]
    mj <- colSums(t); nj <- rowSums(t)
    if (any(mj == 0) || any(nj == 0)) next
    coef <- c(-mj[1] * nj[1] * or_mh, nj[2] - mj[1] + or_mh * (nj[1] + mj[1]), 1 - or_mh)
    sols <- Re(polyroot(coef))
    e <- sols[(0 < sols) & (sols <= min(nj[1], mj[1]))][1]
    cells <- c(e, nj[1] - e, mj[1] - e, nj[2] - mj[1] + e)
    v <- 1 / sum(1 / cells)
    X2 <- X2 + (t[1, 1] - e)^2 / v
    dev <- dev + (t[1, 1] - e); vsum <- vsum + v; used <- used + 1
  }
  if (tarone) X2 <- X2 - dev^2 / vsum
  list(chisq = unname(X2), df = used - 1, p = unname(pchisq(X2, used - 1, lower.tail = FALSE)))
}

reference_for <- function(x, name, family) {
  d <- dim(x)
  out <- list(name = name, family = family, dim = d, counts = as.vector(x))

  mh <- safe(mantelhaen.test(x, correct = FALSE))
  if (!is.null(mh)) {
    out$mantelhaen_nocorrect <- list(statistic = unname(mh$statistic),
                                     df = unname(mh$parameter), p = mh$p.value)
    if (all(d[1:2] == 2)) {
      out$mantelhaen_nocorrect$estimate <- unname(mh$estimate)
      out$mantelhaen_nocorrect$conf_int <- as.numeric(mh$conf.int)
    }
  }
  if (all(d[1:2] == 2)) {
    mhc <- safe(mantelhaen.test(x, correct = TRUE))
    if (!is.null(mhc)) out$mantelhaen_correct <- list(statistic = unname(mhc$statistic), p = mhc$p.value)
    for (alt in c("less", "greater")) {
      m <- safe(mantelhaen.test(x, correct = TRUE, alternative = alt))
      if (!is.null(m)) out[[paste0("mantelhaen_", alt)]] <- list(p = m$p.value, conf_int = as.numeric(m$conf.int))
    }
  }

  if (d[3] == 1) {  # vcdExtra needs a 2-D table for a single stratum
    out$vcd_table <- cmh_rows(safe(CMHtest(x[, , 1])$table))
    out$vcd_modridit <- cmh_rows(safe(CMHtest(x[, , 1], rscores = "midrank", cscores = "midrank")$table))
  } else {
    out$vcd_table <- cmh_rows(safe(CMHtest(x, overall = TRUE)$ALL$table))
    out$vcd_modridit <- cmh_rows(safe(CMHtest(x, rscores = "midrank", cscores = "midrank", overall = TRUE)$ALL$table))
  }
  out$vcd_failed <- is.null(out$vcd_table)
  out$r_pooled <- list(table = safe(pooled_cmh_r(x, "table")), rank = safe(pooled_cmh_r(x, "rank")),
                       ridit = safe(pooled_cmh_r(x, "ridit")), modridit = safe(pooled_cmh_r(x, "modridit")))

  ct <- safe(coin::cmh_test(as.table(x)))
  if (!is.null(ct)) out$coin_general <- list(statistic = unname(statistic(ct)), p = unname(pvalue(ct)))

  if (all(d[1:2] == 2)) {
    margins_ok <- all(apply(x, 3, function(t) all(rowSums(t) > 0) && all(colSums(t) > 0)))
    ep <- if (margins_ok) safe(epi.2by2(as.table(x), method = "cohort.count")) else NULL
    if (!is.null(ep)) {
      md <- ep$massoc.detail
      out$epir <- list(
        or_mh = as.numeric(md$OR.mh.wald[1, c("est", "lower", "upper")]),
        rr1_mh = as.numeric(md$RR.mh.wald[1, c("est", "lower", "upper")]),
        breslow_day = list(chisq = md$bOR.homog$test.statistic, df = md$bOR.homog$df, p = md$bOR.homog$p.value)
      )
      x2 <- x[, 2:1, , drop = FALSE]  # swap columns: column-2 relative risk
      ep2 <- safe(epi.2by2(as.table(x2), method = "cohort.count"))
      if (!is.null(ep2)) out$epir$rr2_mh <- as.numeric(ep2$massoc.detail$RR.mh.wald[1, c("est", "lower", "upper")])
    }
    out$handwritten <- list(
      or_logit = logit_or(x), rr1_logit = logit_rr(x, 1), rr2_logit = logit_rr(x, 2),
      breslow_day = safe(bd_test(x, FALSE)),
      breslow_day_tarone = safe(bd_test(x, TRUE))
    )
  }
  out
}

# ---------------------------------------------------------------- random tables
set.seed(20261031)
shapes <- list(c(2,2,2), c(2,2,5), c(2,2,12), c(2,3,3), c(3,2,4), c(3,3,3), c(2,5,2),
               c(4,4,2), c(3,5,6), c(5,3,3), c(4,6,8), c(2,4,20), c(6,2,10), c(3,3,1))
cases <- list()
for (s in shapes) {
  for (lambda in c(0.8, 3, 15)) {
    for (rep in 1:3) {
      repeat {
        x <- array(rpois(prod(s), lambda * runif(prod(s), 0.3, 1.7)), dim = s)
        ok_strata <- all(apply(x, 3, sum) >= 2)
        ok_levels <- all(apply(x, 1, sum) > 0) && all(apply(x, 2, sum) > 0)
        if (ok_strata && ok_levels) break
      }
      nm <- sprintf("random_%s_lambda%s_%d", paste(s, collapse = "x"), lambda, rep)
      cases[[length(cases) + 1]] <- reference_for(x, nm, "random")
    }
  }
}

# ------------------------------------------------- tables with single-observation strata
# R's mantelhaen.test refuses these and vcdExtra returns NaN. The reference is computed
# on the same table with those strata removed; cmhstats must return the same numbers
# from the full table.
sparse <- list()
for (rep in 1:12) {
  s <- c(sample(2:4, 1), sample(2:5, 1), sample(4:12, 1))
  repeat {
    x <- array(rpois(prod(s), 1.5), dim = s)
    n_h <- apply(x, 3, sum)
    x[, , 1] <- 0; x[sample(s[1], 1), sample(s[2], 1), 1] <- 1   # one stratum with n = 1
    n_h <- apply(x, 3, sum)
    keep <- n_h >= 2
    xr <- x[, , keep, drop = FALSE]
    if (sum(keep) >= 2 && all(apply(xr, 1, sum) > 0) && all(apply(xr, 2, sum) > 0)) break
  }
  ref <- reference_for(xr, sprintf("single_obs_stratum_%d", rep), "single_obs")
  ref$full_dim <- dim(x); ref$full_counts <- as.vector(x)
  if (all(dim(x)[1:2] == 2)) {
    # SAS's logit rule (add 0.5 to strata with a zero cell) is applied to every stratum,
    # including the single-observation one, so these references use the full table
    ref$handwritten_full <- list(or_logit = logit_or(x), rr1_logit = logit_rr(x, 1), rr2_logit = logit_rr(x, 2))
  }
  sparse[[length(sparse) + 1]] <- ref
}

# ----------------------------------------------------------- named textbook examples
Satisfaction <- as.table(array(c(1, 2, 0, 0, 3, 3, 1, 2, 11, 17, 8, 4, 2, 3, 5, 2,
                                 1, 0, 0, 0, 1, 3, 0, 1, 2, 5, 7, 9, 1, 1, 3, 6), dim = c(4, 4, 2)))
Rabbits <- array(c(0, 0, 6, 5, 3, 0, 3, 6, 6, 2, 0, 4, 5, 6, 1, 0, 2, 5, 0, 0), dim = c(2, 2, 5))
data(SexualFun, package = "vcd"); data(Arthritis, package = "vcd")
arth <- xtabs(~ Treatment + Improved + Sex, data = Arthritis)
named <- list(
  reference_for(unclass(UCBAdmissions), "UCBAdmissions", "named"),
  reference_for(unclass(Satisfaction), "Satisfaction (mantelhaen.test docs)", "named"),
  reference_for(Rabbits, "Rabbits penicillin (mantelhaen.test docs)", "named"),
  reference_for(array(unclass(arth), dim = dim(arth)), "Arthritis Treatment x Improved | Sex", "named")
)
named[[1]]$sexualfun_single_stratum <- list(
  counts = as.vector(SexualFun),
  vcd_table = cmh_rows(CMHtest(SexualFun)$table)
)

write_json(list(info = info, cases = c(cases, sparse, named)),
           "tests/fixtures/r_reference.json", digits = NA, auto_unbox = TRUE, pretty = FALSE)

# ------------------------------------------------------------- CAMIS scenarios in R
adcibc <- read.csv("tests/fixtures/adcibc.csv")
d2 <- subset(adcibc, TRTPN != 54 & AGEGR1 != ">80")
tab <- function(df, x, y, k) {
  t <- table(df[[x]], df[[y]], df[[k]])
  t
}
s9 <- tab(adcibc, "TRTP", "AVAL", "SITEID")
n9 <- apply(s9, 3, sum)
aval <- as.numeric(dimnames(s9)[[2]])
s9_drop <- s9[, , n9 >= 2, drop = FALSE]
err <- function(expr) tryCatch({expr; "ran without error"}, error = function(e) conditionMessage(e))
# which strata have a singular per-stratum covariance for the row mean scores statistic?
per_stratum_singular <- names(n9)[sapply(seq_along(n9), function(h) {
  xh <- s9[, , h]; nt <- sum(xh)
  if (nt < 2) return(FALSE)
  any(rowSums(xh) == 0) || sum(colSums(xh) > 0) < 2
})]
camis <- list(
  info = info,
  scenario9_dims = dim(s9),
  scenario9_aval_levels = aval,
  scenario9_single_obs_sites = names(n9)[n9 < 2],
  scenario9_sites_missing_a_level = per_stratum_singular,
  scenario9_vcdExtra_full_table = err(CMHtest(s9, overall = TRUE)),
  scenario9_vcdExtra_without_single_obs_sites = err(CMHtest(s9_drop, overall = TRUE)),
  scenario9_pooled_formula_in_R = pooled_cmh_r(s9_drop, "table", ctab = aval),
  scenario9_mantelhaen_without_single_obs_sites = {
    m <- mantelhaen.test(s9_drop); list(statistic = unname(m$statistic), df = unname(m$parameter), p = m$p.value)
  }
)
write_json(camis, "tests/fixtures/r_camis.json", digits = NA, auto_unbox = TRUE, pretty = TRUE)
cat("cases:", length(cases) + length(sparse) + length(named), "\n")
