# Phase 3 Physical Mechanism and Network Coupling

## Pre-Committed Criticality Criteria

These criteria were written into the Phase 3 script before reporting outcomes:

- **susceptibility:** Within at least two N-strata, Var(r_excess | lambda,N) must show an interior peak at least 1.5x the mean of the low/high edge bins, and peak lambda locations must be broadly consistent across N-strata.
- **bimodality:** Near the Phase-2 candidate lambda_c, a two-Gaussian mixture for r1 must improve BIC by >10 over one Gaussian, more strongly than in low/high lambda regimes.
- **n_dependence:** Apparent lambda_c estimated from the susceptibility peak should not drift systematically with N-stratum median N; |corr(lambda_peak, N_mid)| < 0.4 is treated as stable.
- **verdict_rule:** If at least two of the three signatures pass, report genuine transition signatures; otherwise report smooth crossover / no criticality.

## Headline Verdict

T1 verdict: **consistent with a smooth crossover / no criticality**. The Phase 2 piecewise threshold is therefore treated
as an apparent crossover scale, not an established critical point, unless future data satisfy
the physical signatures below.

Signature summary: susceptibility pass=False, bimodality
pass=False, N-stability pass=True;
lambda-peak/N correlation=0.282.

Apparent susceptibility peaks by N stratum:

| N_stratum | N_mean | rows | lambda_peak | peak_var_r_excess | edge_mean_var_r_excess | peak_edge_ratio | passes_peak_rule |
| --- | --- | --- | --- | --- | --- | --- | --- |
| N13_plus | 17.075 | 848 | 12.962 | 0.055 | 0.061 | 0.891 | False |
| N09_12 | 10.159 | 1813 | 55.894 | 0.063 | 0.055 | 1.139 | False |
| N04_05 | 4.477 | 2311 | 11.200 | 0.156 | 0.145 | 1.072 | False |
| N02_03 | 2.468 | 2452 | 4.000 | 0.557 | 0.526 | 1.059 | False |
| N06_08 | 6.884 | 2711 | 20.286 | 0.084 | 0.075 | 1.127 | False |

Bimodality / coexistence check:

| bic_1 | bic_2 | delta_bic_1_minus_2 | component_sep_sd | regime | rows |
| --- | --- | --- | --- | --- | --- |
| -562.610 | -952.805 | 390.195 | 3.859 | high_lambda | 2974 |
| 143.666 | -452.932 | 596.597 | 3.065 | low_lambda | 3345 |
| -43.863 | -61.859 | 17.996 | 3.143 | near_candidate | 371 |

## Dwell-Time Mechanism

Dwell events were detected from GPS as low-speed intervals <= 1.5 m/s within
50 m of an official stop. Events shorter than 15 s
or longer than 300 s were excluded from stop-dwell summaries; longer
terminal-like dwells were excluded as layover. Low-speed intervals away from stops were kept
separately as non-stop delay.

Detected dwell events: 1,875,802; non-stop delay events:
1,430,536; low-speed intervals near stops:
4,755,025; low-speed intervals away from stops:
5,339,664.

Mediation model rows: 10,082. The estimated indirect effect
lambda -> dwell CV -> r_excess is 0.0001
(bootstrap 95% CI -0.0002 to 0.0004); proportion mediated =
0.002. **No detectable dwell-variability mediation was found.** GPS dwell estimates are noisy and
traffic/signal delays remain confounders, so this should be treated as a null mechanistic
test rather than evidence against the physical mechanism.

## Spatial Amplification

Dense cross-sections were placed at 5%, every 10% from 10% to 90%, and 95% of route arc length.
Terminal proxy bands were excluded from the slope fit; headway-CV trend was fit only on the
interior 10%-90% sections.

Hourly slope rows: 6,669. Mean headway-CV slope =
0.1198; median slope =
0.1581; one-sample t-test two-sided p =
1.93e-58. The standardized-lambda
coefficient for slope is -0.0391. This means **downstream headway-CV amplification is present**, but it does not increase with demand in this fit.

## Corridor Coupling

Phase 3 uses within-pair controls: for each shared-trunk line pair, compare cross-line
correlation of segment-local corrected synchronization on the shared trunk against the same
pair's off-trunk segments.

Pairs tested: 176. Mean shared corr =
0.051; mean off-trunk corr =
-0.005; shared-minus-off =
0.055; sign-flip permutation p =
0.000. Thus **shared-trunk correlations exceed off-trunk controls**. Segment-local N is stored in
`outputs/phase3/segment_local_order.csv`.

## Abstract-Ready Claims, Updated

- **Supported:** Raw Kuramoto order parameters are strongly finite-size confounded; same-N
  null correction is mandatory.
- **Supported:** In this city-scale sample, corrected synchronization is lower on weekends
  despite higher raw r, resolving the Phase 1 paradox.
- **Supported:** Demand per active bus is positively associated with corrected bunching, but
  the effect is observational and modest.
- **Supported:** consistent with a smooth crossover / no criticality; do not claim a sharp critical transition from the
  current data.
- **Null/mechanism unresolved:** No detectable dwell-variability mediation was found; the dwell channel is not established by
  these GPS-derived dwell events.
- **Supported:** downstream headway-CV amplification is present, but it does not increase with demand in this fit.
- **Exploratory but positive:** shared-trunk correlations exceed off-trunk controls; this is the seed for a Phase 4
  corridor-network model, not a final causal claim.
