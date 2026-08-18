# Legacy pipeline inventory

The files here predate the strict `original -> cleaned -> processed -> analysis -> output` architecture. They are retained unchanged so past results remain auditable. They are not canonical inputs for new work.

## `phase1_to_phase4/code/`

### `phase1_kuramoto_pipeline.py`

Mixed raw-data audit, cleaning, feature engineering, analysis, and plotting. It:

1. scans Mobility and Ticket files and checks whether vehicle identifiers intersect;
2. reads selected Mobility dates and routes;
3. normalizes route codes and drops unusable service/time/coordinate rows;
4. chooses static or GTFS route geometry and projects GPS positions to route arc length;
5. splits trajectory gaps and resamples positions to 30-second intervals;
6. calculates crossing events, headway CV, Kuramoto order, and 10-minute demand;
7. writes both intermediate CSV tables and four figures.

Because it performs all four stages in one script, it must not be used as the new canonical pipeline.

### `phase2_analysis.py`

Recomputes the study over the full overlap period. It selects high-demand routes, chooses geometry by empirical residual, builds one representative bus state per hour, excludes layover/sparse/bad-map-match bus-hours, calculates hourly order and finite-size nulls, joins Ticket demand and weather, fits mixed/piecewise models, tests corridor correlations, and writes figures and statistical tables.

### `phase3_analysis.py`

Reads Phase 2 outputs and raw GPS. It detects stop dwell and non-stop delay events, builds dense route-section crossings and headway CV, tests criticality signatures, mediation, spatial amplification, and within-pair corridor coupling, then writes figures and summary tables.

### `phase4_analysis.py`

Reads Phase 2/3 outputs. It controls corridor coupling for shared-segment speed, computes lag and transfer-entropy diagnostics, fits amplification-driver models, creates the Phase 4 synthesis table, and writes publication figures.

## `phase1_to_phase4/outputs/`

Frozen outputs created by the four scripts above:

- `phase1/`: map-matched samples, resampled trajectories, crossing/headway/order/demand tables and exploratory figures.
- `phase2/`: bus-hour candidates, hourly observables, finite-size nulls, demand/weather/corridor tables, regression outputs and figures.
- `phase3/`: dwell/delay/crossing events, criticality, mediation, spatial amplification, corridor coupling and figures.
- `phase4/`: speed-controlled corridor and amplification-driver results plus historical
  figure inputs. Three byte-identical publication PDFs were removed from this archive
  copy and retained canonically in `Output/Figure/`.

These outputs are useful for reproducing old manuscripts but must not be mixed into `data_after_all_processed/`.

## `phase1_to_phase4/reports/`

The original Phase 1–4 reports and their combined detailed summary. They document the decisions and conclusions of the mixed pipeline.

## `full_report_refactor/code/`

### `pipelines/build_cells.py`

Rebuilds cell-level order parameters under arc-length and travel-time phase definitions from old Phase 2/3 per-bus tables.

### `pipelines/sweep.py`

Runs estimator × null × phase robustness comparisons, including PPC, finite-size corrections, marginal contrasts, and N-matched contrasts.

### `pipelines/terminal_sensitivity.py`

Recomputes synchronization after progressively trimming route endpoints.

### `pipelines/route_position.py`

Splits buses into terminal, near-terminal, and route-interior zones and estimates zone-specific synchronization and demand effects.

### `pipelines/theory_tests.py`

Runs Newell–Potts amplification, Kuramoto critical-coupling, and finite-size scaling-collapse tests.

### `pipelines/figures.py`

Creates the four full-report figures from the old `week1_2` and `week3` output tables.
Its data/output path constants were updated after archiving so it reads the frozen
tables here and writes the rebuilt PDFs to `Output/Figure/`; analytical logic is
unchanged.

### `pipelines/make_preview.py`

Creates a local A4 PDF preview from `overleaf/main.tex` by temporarily replacing the unavailable publisher document class.

### `pipelines/verify_vehicle_join.py`

Independently verifies the lack of exact identifier overlap between Mobility `id` and Ticket `vehicle_number` and summarizes route overlap.

### `src/netmob/`

Reusable implementations of phase transformations, finite-size-aware synchronization estimators, null models, contrast statistics, Kluyver random-walk theory, Newell–Potts dynamics, and Kuramoto critical coupling.

### `tests/test_theory.py`

Twenty-one tests covering the exact null, PPC, estimator properties, Daido harmonics, Newell–Potts dynamics, and Kuramoto-related theory.

## `full_report_refactor/outputs/`

- `week1_2/`: cell tables, travel-time phase, estimator/null sweep, terminal sensitivity, and weekend contrasts.
- `week3/`: route-position, Newell–Potts, Kuramoto, and scaling-collapse results.

`full_report_refactor/RESULTS_week1_2.md` is the historical Week 1–2 result narrative
associated with these tables.

## `vehicle_matching_audit/`

- `01_dataset_integrity_vehicle_and_route_audit.ipynb`: raw integrity, identifier, route, full-name, and route-detail audit.
- `02_time_alignment_and_single_timestamp_ambiguity.ipynb`: timezone/clock-offset and single-time candidate ambiguity.
- `03_longitudinal_matching_stability_and_event_join.ipynb`: longitudinal candidate scoring, bidirectional uniqueness, day stability, and nearest-GPS diagnostics.
- `04_final_audit_report.ipynb`: conclusion that route × time supplies candidate evidence but does not defensibly identify a full vehicle crosswalk without another anchor.

These notebooks originally read `data/` directly and are archived rather than silently rewritten.
