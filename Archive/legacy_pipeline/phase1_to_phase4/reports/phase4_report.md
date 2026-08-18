# Phase 4 Common-Shock Controls and Physical Synthesis

## Pre-Committed Coupling Decision Rule

- **speed_survival:** The controlled shared-minus-offtrunk correlation gap must be positive, its pair bootstrap 95% CI must exclude 0, and it must retain at least 50% of the raw gap.
- **lead_lag:** At least 25% of pairs must have a significant non-zero-lag cross-correlation peak (permutation p<0.05) or the signed net transfer-entropy asymmetry must exceed its pair-shuffle 95% null.
- **verdict:** If both tests pass: coupling survives common-shock controls. If neither passes: largely common-shock. Otherwise: mixed.

## Gate Verdict

T1 verdict: **largely common-shock**. The raw shared-minus-offtrunk gap is
0.055 (95% bootstrap CI 0.028
to 0.085). After residualizing both shared-trunk line
series on the hourly shared-segment mean speed, the gap is
0.018 (95% CI
-0.015 to
0.054), retaining
0.33 of the raw gap.

Lead-lag support is False: the fraction of pairs with a
significant non-zero-lag peak is 0.058,
with typical significant lag -1.0 h.
Mean absolute transfer-entropy asymmetry is
0.0466
and signed net asymmetry is 0.0011
(signed permutation p=0.874).

Interpretation: the speed-controlled correlation gap tests against a road-speed common shock,
but the lead-lag/TE evidence decides whether the residual signal looks propagating or
symmetric. This is why the final verdict can be mixed even when the speed-controlled gap
remains positive.

## Spatial Amplification Drivers

Amplification-driver model rows: 5,544; line-direction groups:
38. Standardized random-intercept coefficients:

| term | estimate | ci_low | ci_high | p_value |
| --- | --- | --- | --- | --- |
| initial_headway_cv_std | -0.283 | -0.296 | -0.270 | 0.000 |
| sinuosity_std | -0.070 | -0.154 | 0.015 | 0.105 |
| route_length_km_std | 0.040 | -0.022 | 0.103 | 0.205 |
| lambda_boardings_per_bus_std | 0.015 | 0.002 | 0.027 | 0.021 |
| stop_density_per_km_std | -0.005 | -0.078 | 0.068 | 0.887 |

Initial CV is the dominant driver, but with a negative sign: hours that are already irregular at the first interior cross-section have less additional downstream CV growth. This is consistent with saturation/ceiling behavior, not with demand-triggered growth. In this fit, the spatial amplification story is therefore empirical
rather than demand-critical: amplification is tied more to the ranked driver above than to a
sharp demand threshold.

## Consolidated Physical Picture

Across Phases 1-4 the organizing principle is spatial/networked service instability, not an
equilibrium-like critical transition. Raw r must be finite-size corrected; after correction,
demand per bus has a modest positive association with bunching but the pre-committed
criticality signatures fail. Headway irregularity amplifies downstream, and shared corridors
show a small residual synchronization signal after speed control, but lead-lag evidence is
needed before interpreting this as propagating inter-line coupling rather than a common road
state.

## Abstract-Ready Claims, Final

- **Supported:** Same-N finite-size correction is mandatory before interpreting empirical bus
  Kuramoto order parameters.
- **Supported:** The Niterói data are consistent with demand-modulated bunching as a smooth
  crossover, not a sharp critical transition at city scale.
- **Supported:** Downstream headway-CV amplification is present; its driver ranking is given
  by the standardized effects above.
- **Suggestive:** Shared-trunk synchronization has a residual speed-controlled component, but
  the Phase 4 gate verdict is `largely common-shock`.
- **Null:** GPS-derived dwell variability did not mediate demand -> corrected bunching in
  Phase 3.
- **Not supported:** A causal claim that passenger demand alone produces a critical
  synchronization transition.
