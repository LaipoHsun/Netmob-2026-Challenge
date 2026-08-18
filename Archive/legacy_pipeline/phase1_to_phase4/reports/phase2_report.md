# Phase 2 Finite-Size-Corrected Synchronization Analysis

## Headline Verdict

Phase 2 resolves the Phase 1 finite-size concern by comparing every hourly raw order parameter
to a same-`N` random-phase null with 5,000 simulations per `N`. The weekday/weekend
result after correction is: **corrected synchronization is lower on weekends**.

Raw weekend-minus-weekday `r1` difference is 0.047
(95% bootstrap CI 0.039 to
0.056). Corrected weekend-minus-weekday
`r_excess` difference is -0.067
(95% CI -0.085 to
-0.048); `z` difference is
-0.441 (95% CI -0.486
to -0.397). N is reported alongside every corrected
summary in `outputs/phase2/p2_weekday_weekend_peak_comparison.csv`.

## Scope and Exclusions

This run covers the mobility/ticketing overlap days with mobility files present:
2026-03-11, 2026-03-12, 2026-03-13, 2026-03-14, 2026-03-15, 2026-03-16, 2026-03-17, 2026-03-20, 2026-03-21, 2026-03-22, 2026-03-23, 2026-03-24, 2026-03-25, 2026-03-26, 2026-03-27, 2026-03-28, 2026-03-29, 2026-03-30, 2026-03-31. The analysis unit is `(line, direction, hour)`.
Demand remains line-hour level and is repeated across directions because Phase 1 found 0%
vehicle-id overlap.

Final quality-approved top ridership lines with usable direction-0 and direction-1 geometry:

| rank_by_boardings | line_code | ticket_codes | boardings |
| --- | --- | --- | --- |
| 1 | 30 | 30 | 279192 |
| 3 | 48 | 48 | 197406 |
| 4 | 33 | 33 | 193355 |
| 5 | 49.2 | 49.2 | 187719 |
| 6 | 46 | 46 | 183131 |
| 7 | 67 | 67 | 167373 |
| 8 | 49.1 | 49.1 | 156781 |
| 9 | 45 | 45 | 156314 |
| 10 | 36 | 36 | 152287 |
| 11 | OC3 | OC3 | 149502 |
| 12 | 39A | 39A | 149346 |
| 13 | 35 | 35 | 143018 |
| 14 | 62 | 62,62B | 140680 |
| 16 | 38A | 38A | 126653 |
| 17 | OC2 | OC2 | 121233 |
| 18 | 61 | 61 | 113313 |
| 19 | OC1 | OC1 | 109278 |
| 20 | 31 | 31 | 93399 |
| 21 | 44 | 44 | 89401 |
| 22 | 34A | 34A | 86343 |

Bus-hour active oscillator exclusions are logged in
`outputs/phase2/filtering_and_exclusion_log.json`. Terminal layover is defined as at least
10 minutes in a bus-hour, at least 75%
of samples within 5% of either route endpoint, and arc-length span <= 150 m.
This removed 3,920 bus-hours.
Sparse midpoint sampling removed 11,326
bus-hours, and high median residual removed
948. Active bus-hours
retained: 69,466.

Map-matching quality for selected lines:

| line_code | direction | rows | geometry_source | residual_m_median | residual_m_p95 | poor_match_flag |
| --- | --- | --- | --- | --- | --- | --- |
| 30 | 0 | 1028290 | GTFS:TransNit_20260320:shape_id=75279_I | 12.415 | 59.098 | False |
| 30 | 1 | 388565 | GTFS:TransNit_20260320:shape_id=75279_V | 4.025 | 15.667 | False |
| 31 | 0 | 289753 | line_routes.json | 4.888 | 98.602 | False |
| 31 | 1 | 209681 | GTFS:TransNit_20260320:shape_id=51376_V | 4.282 | 18.923 | False |
| 33 | 0 | 113559 | GTFS:TransOceanico_20260315:shape_id=25330_I | 5.068 | 60.353 | False |
| 33 | 1 | 85097 | GTFS:TransOceanico_20260315:shape_id=25330_V | 4.665 | 36.029 | False |
| 34A | 0 | 116787 | GTFS:TransOceanico_20260315:shape_id=35599_I | 4.448 | 21.866 | False |
| 34A | 1 | 118227 | GTFS:TransOceanico_20260315:shape_id=35599_V | 3.850 | 12.091 | False |
| 35 | 0 | 108761 | GTFS:TransOceanico_20260315:shape_id=37737_I | 3.718 | 28.859 | False |
| 35 | 1 | 137025 | GTFS:TransOceanico_20260315:shape_id=37737_V | 2.836 | 17.633 | False |
| 36 | 0 | 166448 | GTFS:TransOceanico_20260315:shape_id=35601_I | 4.335 | 28.040 | False |
| 36 | 1 | 220464 | GTFS:TransOceanico_20260315:shape_id=35601_V | 3.324 | 17.863 | False |
| 38A | 0 | 147792 | GTFS:TransOceanico_20260315:shape_id=37735_I | 3.671 | 26.338 | False |
| 38A | 1 | 163465 | GTFS:TransOceanico_20260315:shape_id=37735_V | 3.801 | 26.403 | False |
| 39A | 0 | 241084 | GTFS:TransOceanico_20260315:shape_id=35602_I | 4.594 | 22.399 | False |
| 39A | 1 | 308943 | GTFS:TransOceanico_20260315:shape_id=35602_V | 3.505 | 20.010 | False |
| 44 | 0 | 133839 | GTFS:TransOceanico_20260315:shape_id=35603_I | 4.465 | 28.582 | False |
| 44 | 1 | 163732 | GTFS:TransOceanico_20260315:shape_id=35603_V | 4.042 | 18.280 | False |
| 45 | 0 | 275095 | GTFS:TransOceanico_20260315:shape_id=35604_I | 4.309 | 28.939 | False |
| 45 | 1 | 298438 | GTFS:TransOceanico_20260315:shape_id=35604_V | 3.986 | 16.881 | False |
| 46 | 0 | 167988 | GTFS:TransOceanico_20260315:shape_id=37694_I | 3.640 | 26.219 | False |
| 46 | 1 | 144279 | GTFS:TransOceanico_20260315:shape_id=37694_V | 3.760 | 28.293 | False |
| 48 | 0 | 123298 | GTFS:TransOceanico_20260315:shape_id=37712_I | 3.552 | 29.516 | False |
| 48 | 1 | 149095 | GTFS:TransOceanico_20260315:shape_id=37712_V | 3.001 | 22.920 | False |
| 49.1 | 0 | 221089 | line_routes.json | 6.659 | 19.168 | False |
| 49.1 | 1 | 372724 | GTFS:TransNit_20260320:shape_id=51334_V | 4.026 | 12.736 | False |
| 49.2 | 0 | 441242 | line_routes.json | 3.812 | 14.968 | False |
| 49.2 | 1 | 246381 | GTFS:TransNit_20260320:shape_id=51358_V | 4.447 | 14.128 | False |
| 61 | 0 | 252670 | GTFS:TransNit_20260320:shape_id=98210_I | 3.239 | 16.380 | False |
| 61 | 1 | 303343 | GTFS:TransNit_20260320:shape_id=98210_V | 3.694 | 35.233 | False |
| 62 | 0 | 297700 | GTFS:TransNit_20260320:shape_id=50962_I | 4.437 | 16.937 | False |
| 62 | 1 | 228785 | GTFS:TransNit_20260320:shape_id=50962_V | 4.123 | 14.145 | False |
| 67 | 0 | 330562 | GTFS:TransNit_20260320:shape_id=58707_I | 2.537 | 10.431 | False |
| 67 | 1 | 343117 | GTFS:TransNit_20260320:shape_id=58707_V | 2.872 | 11.103 | False |
| OC1 | 0 | 185166 | GTFS:TransOceanico_20260315:shape_id=35624_I | 5.057 | 18.370 | False |
| OC1 | 1 | 229189 | GTFS:TransOceanico_20260315:shape_id=35624_V | 5.569 | 26.667 | False |
| OC2 | 0 | 163432 | GTFS:TransOceanico_20260315:shape_id=36831_I | 4.147 | 19.470 | False |
| OC2 | 1 | 185523 | GTFS:TransOceanico_20260315:shape_id=36831_V | 5.894 | 23.022 | False |
| OC3 | 0 | 258033 | GTFS:TransOceanico_20260315:shape_id=58591_I | 4.501 | 19.629 | False |
| OC3 | 1 | 255924 | GTFS:TransOceanico_20260315:shape_id=58591_V | 6.370 | 24.294 | False |

## Finite-Size Correction

For each `N`, the null distribution is generated from uniform random phases. The perfect splay
state is also recorded (`r_splay`, zero up to numerical precision for N>=2). We use:

`r_excess = (r1 - E_null[r|N]) / (1 - E_null[r|N])`

and

`z = (r1 - E_null[r|N]) / sd_null[r|N]`.

Rows with `N=1` are retained in the file but excluded from corrected modeling because the
random null has `r=1` and zero variance. The final hourly table has 11,641 rows,
of which 10,135 have `N>=2`.

Corrected weekday/weekend and peak/off-peak table:

| day_category | rows | mean_N | mean_r1 | mean_r_excess | mean_z | mean_lambda | comparison | period |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| weekday | 6441 | 7.988 | 0.482 | 0.179 | 0.806 | 57.554 | weekday_weekend |  |
| weekend | 3694 | 4.470 | 0.529 | 0.112 | 0.366 | 40.333 | weekday_weekend |  |
| weekday | 5077 | 7.227 | 0.499 | 0.178 | 0.758 | 63.683 | day_period | off_peak |
| weekday | 1364 | 10.820 | 0.420 | 0.180 | 0.985 | 34.742 | day_period | peak |
| weekend | 2899 | 4.339 | 0.537 | 0.113 | 0.356 | 41.156 | day_period | off_peak |
| weekend | 795 | 4.945 | 0.504 | 0.109 | 0.400 | 37.330 | day_period | peak |

## Demand-Coupling and Transition Tests

Primary load is `lambda = boardings / N`. The main association model is a Gaussian random-
intercept model with random intercepts for line-direction groups and fixed effects for
standardized lambda, N, hour-of-day sine/cosine, weekend, rain, and lambda x rain. It is
implemented directly with `numpy`/`scipy` maximum likelihood because `statsmodels` is not
installed in this environment.

For `r_excess`, the standardized-lambda coefficient is 0.039
(SE 0.005, p=2.89e-15). The lambda x rain
interaction is -0.006 (SE 0.013,
p=0.63).

Transition-shape comparison, using line-direction fixed effects plus the same controls:

| model | aic | delta_aic | rss | lambda_c | scale |
| --- | --- | --- | --- | --- | --- |
| piecewise | 10179.134 | 0.000 | 1604.802 | 135.740 |  |
| linear | 10208.515 | 29.380 | 1610.096 |  |  |
| sigmoid | 10265.356 | 86.222 | 1618.513 | 101.760 | 24.814 |

The piecewise model is selected by AIC; candidate critical load lambda_c=135.74 boardings/bus (bootstrap median 135.74, 95% CI 92.20-135.74).

These are observational associations, not causal estimates.

## Weather Modifier

Weather is converted from the `Timestamp (UTC)` column to local BRT hours and merged by hour.
Rain hours in the mobility window occur on: 2026-03-11, 2026-03-12, 2026-03-13, 2026-03-20, 2026-03-21, 2026-03-23.
The matched dry-hour comparison pairs line-direction/hour-of-day cells observed under both rain
and dry conditions. Rain-minus-dry mean `r_excess` is
-0.008 across
530 paired cells (95% CI -0.038 to
0.024). This is the cleanest quasi-experimental check, but it still
requires the assumption that rain affects bunching mainly through dwell/service variability
conditional on observed load.

## Corridor Coupling

Shared trunks are estimated without GIS dependencies by sampling each route every 50 m and
counting route-sample overlap within a 50 m buffer; shared stops are official stops within
50 m of both sampled routes. A pair is flagged as sharing a corridor if overlap is at least
1000 m or shared stops >= 5.

Shared-corridor pairs: 176; non-sharing pairs:
14. Mean hourly `r_excess` correlation is
0.039 for shared pairs vs
0.049 for non-shared pairs; label-permutation p-value
is 0.771. This is descriptive evidence for or against
shared-medium coupling, not a full coupled-oscillator network model.

## Decisions for Phase 3 and Abstract-Ready Claims

- Supported: "Raw Kuramoto order parameters in bus data are strongly confounded by active
  fleet size; same-`N` null correction is necessary before interpreting bunching."
- Supported: "corrected synchronization is lower on weekends."
- Supported with observational caveat: "Corrected synchronization has the reported association
  with load per bus after line-direction random intercepts and time/weather controls."
- Not yet causal: "Passenger demand causes synchronization." Weather provides a useful
  exogenous modifier but does not by itself prove the demand mechanism.
- Transition claim: "The piecewise model is selected by AIC; candidate critical load lambda_c=135.74 boardings/bus (bootstrap median 135.74, 95% CI 92.20-135.74)."
- Exploratory only: "Shared corridors show the reported cross-line synchronization correlation;
  Phase 3 should model corridor-level coupling explicitly."
