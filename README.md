# Finite-size-corrected synchronization in a city-scale bus network

This repository contains the scientific code, selected analysis outputs, and
complete report for the NetMob 2026 Niter\'oi bus-synchronization study.  The
analysis reconstructs the corrected empirical results from the released raw
GPS, ticket, GTFS, weather, route-geometry, and stop data.

The report is available as
[`report/NetMob_2026_full_report.pdf`](report/NetMob_2026_full_report.pdf).
Its LaTeX source and Optica template support files are retained in `report/`.

## Repository contents

- `run_end_to_end.py`: one orchestration layer for the corrected raw-to-result
  calculation.
- `code/stages/`: raw-data construction and the Phase 2--4 empirical analyses.
- `code/pipelines/`: cell construction, estimator/null sweeps, route-position
  checks, theory tests, and one-plot-per-file figure generation.
- `code/src/netmob/`: reusable phase, estimator, null, statistics, and theory
  functions.
- `output/data/`: ten selected analysis datasets consumed by downstream code.
- `output/tables/`: thirty machine-readable result tables and summaries.
- `output/figures/`: eighteen independent PDF figures; no subplot grid is used.
- `report/`: the complete English report, LaTeX source, and template support.

The released raw data are intentionally not duplicated in this repository.
They must retain the original NetMob data layout under `data_original/` when
the reconstruction is run.  Generated working intermediates are temporary;
only the selected paper data, numerical tables, and figures are retained.

The analysis treats a bus as a moving phase oscillator on an open route.  It
asks whether apparent synchronization remains after correcting the Kuramoto
order parameter for the strong finite-fleet baseline, and whether demand,
route position, dwell variability, spatial amplification, or shared-corridor
conditions explain the remaining structure.

## Analysis sequence

`run_end_to_end.py` is the orchestration layer.  It calls the scientific code
in the following order:

1. `phase2_analysis.py`: raw mobility and ticket files to bus-hour phases and
   route-hour observables.
2. `phase3_analysis.py`: criticality gate, dwell mediation, downstream
   headway amplification, and within-corridor coupling.
3. `phase4_analysis.py`: shared-speed corridor control and standardized
   amplification-driver model.
4. `build_cells.py`: reconstruct the paper's hourly analysis cells under arc
   and travel-time phase definitions.
5. `sweep.py`: compare estimators, null models, and marginal versus
   fleet-size-matched weekend contrasts.
6. `terminal_sensitivity.py` and `route_position.py`: quantify terminal
   occupancy, trimming sensitivity, zone-specific synchronization, and the
   terminal-fraction control.
7. `theory_tests.py`: Newell--Potts, mean-field Kuramoto, Daido harmonic, and
   scaling-collapse tests.
8. `make_figures.py`: create eighteen independent paper plots.  No combined
   multi-panel figure is created.

The phase scripts create temporary intermediate objects because downstream
calculations require per-bus and per-event data.  The orchestrator retains only
the paper data, numerical tables, and plots after all calculations finish.

## Raw-data interpretation

The raw mobility files contain timestamped GPS fixes with route, direction,
vehicle telemetry identifier, latitude, and longitude.  Ticket records contain
route, reported BRT time, company, operator vehicle number, fare type, and user
identifier.  The telemetry vehicle identifier and ticket vehicle number have
no overlap and no released crosswalk.  Therefore ticket demand cannot be
assigned to an individual GPS bus.

Demand is consequently defined at the route-hour level:

\[
\lambda_{\ell t}=\frac{B_{\ell t}}{N_{\ell d t}},
\]

where \(B_{\ell t}\) is all ticket boardings reported for route \(\ell\) in
hour \(t\), and \(N_{\ell d t}\) is the number of active GPS buses for route
\(\ell\), direction \(d\), and hour \(t\).  Because tickets have no direction,
both directions receive the same route-hour numerator but have their own
active-bus denominator.  This is realized load per active bus, not a
vehicle-level exposure or an exogenous demand treatment.

## `code/stages/phase1_kuramoto_pipeline.py`

Phase 1 is retained as a shared library: Phase 2--4 import its geometry,
projection, route-selection, phase, and formatting functions.  The end-to-end
analysis does not execute its earlier four-day diagnostic report.

The important corrected geometry calculation is the assembly of a static
route represented by several `MultiLineString` members.  The historical code
joined members in file order and could introduce kilometre-scale artificial
connectors.  The corrected function jointly chooses member order and direction
to minimize first the maximum connector and then total connector distance.  A
route is rejected if the best remaining member-to-member gap exceeds 250 m.

For a GPS point \(p\), projection on a route segment from \(a\) to \(b\) uses

\[
t^*=\operatorname{clip}_{[0,1]}
\frac{(p-a)^\top(b-a)}{(b-a)^\top(b-a)},\qquad
p^*=a+t^*(b-a).
\]

The segment cumulative distance plus \(t^*\lVert b-a\rVert\) gives route
position \(s\).  Arc phase is

\[
\phi^{\rm arc}=2\pi s/L,
\]

where \(L\) is the selected route length.  GPS residual distance, route span,
duration, sampling density, and continuity determine whether a bus-hour is
eligible for the synchronization calculation.

## `code/stages/phase2_analysis.py`

Phase 2 is the raw-data entry point used by the full report.

It first selects the high-volume candidate routes from ticket counts, processes
all raw GPS days, applies the corrected geometry, and selects the final twenty
routes using map-matching coverage rather than outcome values.  A bus-hour is
represented by its circular mean route phase.  For a cell with \(N\) active
buses, the first Kuramoto order parameter is

\[
r_1=\left|\frac{1}{N}\sum_{j=1}^{N}e^{i\phi_j}\right|.
\]

This raw statistic is high by chance when \(N\) is small.  Phase 2 constructs
the same-\(N\) uniform-phase null distribution and records its mean, standard
deviation, and quantiles.  The main corrected observable is

\[
r_{\rm exc}=\frac{r_1-\mu_N}{1-\mu_N},
\]

where \(\mu_N=E_0[r_1\mid N]\).  The standardized alternative is

\[
z=\frac{r_1-\mu_N}{\sigma_N}.
\]

When \(N=1\), corrected estimators are undefined because \(r_1=1\) and the
null denominator is zero; the code preserves raw \(r_1\) but records corrected
values as missing.

The route-hour demand numerator is aggregated across ticket-file chunks before
distinct users are counted.  This prevents one user appearing in two chunks
from being counted twice.

The principal association model is a Gaussian random-intercept model

\[
Y_{ig}=x_{ig}^{\top}\beta+u_g+\epsilon_{ig},\qquad
u_g\sim N(0,\tau^2),\quad \epsilon_{ig}\sim N(0,\sigma^2),
\]

where \(g\) is line-direction, \(Y\) is raw or corrected synchronization, and
the covariates include standardized load, fleet size, weekend, peak period,
rain, and temperature.  The code evaluates the block covariance
\(V_g=\sigma^2I+\tau^2\mathbf 1\mathbf 1^\top\) and maximizes the Gaussian
likelihood with numerically bounded variance parameters.

Linear, piecewise-linear, and sigmoid demand-response models are compared by
AIC.  The selected piecewise model estimates a descriptive crossover scale;
it is not interpreted as a causal threshold.

## `code/stages/phase3_analysis.py`

Phase 3 evaluates the proposed mechanisms rather than simply refitting the
main association.

### Criticality gate

The code evaluates three signatures:

1. an interior peak in the conditional variance of corrected order;
2. finite-size drift of the apparent peak location;
3. bimodality or coexistence near the proposed crossover.

A demand crossover is not called a critical transition unless the joint gate
is supported.  The corrected results support only one of the three signatures.

### Dwell mediation

Stops are matched to low-speed GPS episodes to form dwell events.  The hourly
dwell mediator is related to load (path \(a\)), and corrected synchronization
is related jointly to load and dwell variability (path \(b\)).  The indirect
effect is \(ab\).  Route-level bootstrap resampling supplies its confidence
interval.  The released identifiers do not permit passenger-to-bus linkage,
so this remains an aggregate and noisy test.

### Spatial amplification

Routes are divided into ordered interior sections.  Headway coefficient of
variation is calculated within each route section and hour.  The slope of
headway CV on normalized downstream position measures amplification.  The
paper distinguishes a positive average downstream slope from the separate
question of which covariates predict between-cell slope differences.

### Shared-corridor calculation

Pairs of lines with overlapping route geometry are evaluated on their shared
and off-trunk segments.  Segment-local phases are corrected for their own
finite \(N\), and the paired shared-minus-off correlation gap is assessed by a
sign-flip permutation test.

## `code/stages/phase4_analysis.py`

Phase 4 asks whether the shared-corridor gap survives a common road-state
control.  Shared-segment bus speed is estimated from route-distance span over
observed duration.  Each line's segment-local corrected synchronization series
is residualized on the shared mean speed before the between-line correlation
is recomputed.  Pair bootstrap intervals quantify the controlled
shared-minus-off gap.

The propagation tests search non-zero hourly lags and compare the largest
absolute lag correlation with a permutation null.  Quantile-discretized
transfer entropy is also computed in both directions.  These tests distinguish
a shared contemporaneous shock from directional inter-line propagation.

The amplification-driver model uses standardized initial headway CV, realized
load per bus, stop density, route length, and sinuosity with a line-direction
random intercept.  Its coefficient table is the source of the paper's driver
plot.

The original Phase 2--4 diagnostic-figure functions are intentionally replaced
by no-ops in this package.  They did not contribute figures to the corrected
paper and several contained multi-panel layouts.

## `code/pipelines/build_cells.py`

This script reconstructs the final hourly cell table from active bus-hours.  It
retains both arc phase and a travel-time phase.  Travel-time phase uses the
empirical cumulative crossing-time profile, so equal phase increments
correspond approximately to equal expected travel time rather than equal route
distance.

For each phase definition and harmonic \(m=1,2,3\), it computes the Daido order
parameter

\[
r_m=\left|\frac{1}{N}\sum_{j=1}^{N}e^{im\phi_j}\right|.
\]

Calendar, weather, demand, and eligibility covariates are then attached to the
same line-direction-date-hour key.

## `code/src/netmob/estimators.py`

This module defines the five corrected estimators used in the sensitivity
sweep:

- finite-size excess \((r_1-\mu_N)/(1-\mu_N)\);
- same-\(N\) z score;
- pairwise phase coherence
  \(\mathrm{PPC}=(Nr_1^2-1)/(N-1)\), which is unbiased under independent
  uniform phases;
- Rayleigh statistic \(Nr_1^2\);
- probit-transformed probability-integral-transform score based on the full
  null CDF.

All estimators explicitly preserve missingness when their mathematical
denominator is undefined.

## `code/src/netmob/nulls.py`

The null module constructs:

- independent uniform phases;
- a speed-weighted arc-phase null derived from the empirical travel-time map;
- an occupancy null sampled from the empirical phase distribution;
- a line-direction-conditioned permutation null that preserves each route's
  phase occupancy while breaking within-hour simultaneity.

Sampling without replacement is used for the route-conditioned permutation so
one synthetic cell cannot select the same observed bus-hour phase twice.

## `code/src/netmob/stats.py`

This module computes the three-part comparison table used throughout the
paper:

- marginal weekend-minus-weekday contrast;
- exactly fleet-size-matched contrast within \(N\) strata;
- cluster-aware bootstrap confidence intervals.

The marginal and matched quantities are different estimands.  The former
includes changes operating through fleet size; the latter asks for the
weekend difference conditional on the same \(N\).

The module also supplies bounded random-intercept likelihood calculations and
batched NumPy bootstrap operations used by the downstream scripts.

## `code/pipelines/sweep.py`

For both arc and travel-time phase, the sweep evaluates all applicable
combinations of estimator and null model.  It produces seven distinct
phase-by-null specifications and five corrected estimators, for 35 main
contrasts.  It also measures residual finite-size dependence as the slope of
each estimator's mean on \(\log N\).

The route-conditioned permutation is executed explicitly.  This corrects the
older project version in which the paper described the permutation null but
the sweep never called it.

Daido harmonics \(m=1,2,3\) are each corrected using a same-\(N\) null generated
for that harmonic rather than reusing the \(m=1\) baseline.

## `code/pipelines/terminal_sensitivity.py`

This script trims increasing fractions from both route ends, rebuilds eligible
cells after each trim, recalculates \(N\), raw order, and same-\(N\) corrected
order, and records the terminal occupancy share.  Trimming changes both phase
distribution and fleet size, so each trim is a separate estimand.

## `code/pipelines/route_position.py`

Bus-hours are classified as terminal (within 2% of an end), near-terminal
(2--10%), or interior (10--90%).  Order parameters are recomputed within each
zone rather than assigning a zone label to a whole-route statistic.

Zone-specific load models use line-direction fixed effects and clustered
uncertainty.  The terminal-fraction control estimates

\[
r_{\rm exc}=\beta_0+\beta_1\lambda+\beta_2 f_{terminal}
 +\beta_3\log N+\alpha_{\ell d}+\epsilon,
\]

which tests how much of the whole-route load coefficient is attenuated after
the fraction of active buses at terminals is explicitly controlled.

## `code/src/netmob/theory.py` and `code/pipelines/theory_tests.py`

For a Newell--Potts perturbation mode \(\theta\), the per-stop amplification is

\[
A(\theta)=(1+k)-k e^{-i\theta},
\]

under the implemented convention, with growth summarized by \(|A|\) or its
equivalent recurrence form used in the code.  The empirical load parameter is
measured as mean dwell time per stop divided by mean headway.  The script
compares predicted and measured log-CV growth per stop; matching the average
scale does not imply that \(k\) predicts which individual cells amplify.

The mean-field Kuramoto threshold is estimated from each route's empirical
frequency density at zero,

\[
K_c=\frac{2}{\pi g(0)}.
\]

The scaling test fits

\[
r_{\rm exc}=N^{-b}F\big((\lambda-\lambda_c)N^a\big).
\]

Because the binned spread loss is discontinuous and the free exponents can
move to their bounds, the code records narrow-start and broad-start solutions,
parameter-bound hits, permutation improvements, and an explicit
`collapse_identified` decision.  The corrected data do not identify a unique
scaling collapse.

## `code/pipelines/make_figures.py`

The plotting module contains eighteen independent plotting functions.  Each
function creates one figure with one main plotting axis and writes one PDF:

1. raw \(r_1\) by fleet size;
2. Rayleigh and Monte Carlo null error against the exact null;
3. marginal and fleet-size-matched weekend contrasts;
4. bus-position distribution;
5. corrected synchronization by route zone;
6. terminal-trim sensitivity;
7. load coefficient by route zone;
8. analytic Newell--Potts mode growth;
9. empirical Newell--Potts prediction test;
10. Kuramoto threshold against frequency spread;
11. whole-route scaling fit;
12. interior-only scaling fit;
13. optimizer and permutation-null scaling diagnostic;
14. corrected Daido harmonics;
15. corridor gap before and after speed control;
16. significant non-zero-lag distribution;
17. standardized amplification-driver coefficients.
18. corrected load response with linear, piecewise, and sigmoid model
    comparisons.

The former paper composites are therefore represented without any subplot
grid.  The numerical content of each panel is retained, but panel lettering and
layout coupling are removed.
