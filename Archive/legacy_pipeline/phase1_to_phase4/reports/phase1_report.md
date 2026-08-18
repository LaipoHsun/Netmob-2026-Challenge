# Phase 1 Bus-Bunching Observables

## Scope and Data Discovery

This Phase 1 run uses the local workspace because `/mnt/user-data/uploads` and `/mnt/data`
are not mounted in this environment, and `data_description.pdf` was not available. The
empirical CSV schemas and local README files were used instead. Mobility data covers
2026-03-11 00:00:07 to 2026-03-31 23:37:32
with 13,850,467 rows across 19 files.
Ticketing covers 2026-03-01 00:00:02-03:00 to
2026-03-31 23:59:57-03:00 with 6,790,681 rows
across 31 files.

Actual mobility columns are `id, timestamp, tripId, lat, lng, heading, lineId, lineName, headsign, direction`. This differs
from the prompt/PDF-style names: the observed files use `lineId`, `lineName`, `heading`, and
`direction` rather than `linha`, `nomeLinha`, `angle`, and `sentido`. Ticketing columns match
the expected fields except that route codes must be read as strings to preserve values such
as `49.2` and `62B`.

## Critical Join Check

Telemetry vehicle ids and ticketing vehicle numbers do **not** join cleanly. Mobility has
527 distinct `id` values, ticketing has
556 distinct `vehicle_number` values, and their value
overlap is 0 (0.0% of telemetry ids,
0.0% of ticket vehicles). Demand is therefore joined
only at line-time-bin level and repeated across directions in the tidy table; no per-vehicle
demand is inferred.

## Filtering and Focus Lines

Focus days are March 11, 12, 13, and 14, 2026. Out-of-service filtering follows the available
schema: rows with missing `lineId` or missing `direction` are removed. In these four days,
that removed 0 rows; missing
lat/lng removed 0 rows. The
requested lines were retained: `49.2`, `45`, `49.1`, `48`, `35`, and `62`. Ticketing uses
`62B` for the same service family, so `62B` demand is mapped to telemetry line `62`.

| line_code | direction | rows |
| --- | --- | --- |
| 35 | 0 | 61103 |
| 35 | 1 | 75357 |
| 45 | 0 | 69851 |
| 45 | 1 | 75642 |
| 48 | 0 | 88061 |
| 48 | 1 | 108439 |
| 49.1 | 0 | 55601 |
| 49.1 | 1 | 92208 |
| 49.2 | 0 | 109783 |
| 49.2 | 1 | 57358 |
| 62 | 0 | 80387 |
| 62 | 1 | 58064 |

## Phase and Map Matching

For each `(lineId, direction)` route, GPS points are projected to route arc length `s` and
phase is `phi = 2*pi*s/L`. The phase definition treats each direction as its own closed
empirical ring. This is a pragmatic Phase 1 choice: it makes `r(t)` comparable across
linear and circular services, but terminal wrap-around should be revisited in Phase 2 if an
out-and-back topology is modeled explicitly.

`line_routes.json` is used where it contains an exact focus-route geometry. It lacks `45`,
`48`, and `35`, so GTFS `shapes.txt` is used for those lines. Line `62` is coded as `62` in
telemetry/GTFS but appears as `62B` in ticketing/static route names, so demand uses the
`62B` alias while geometry selection prefers the exact GTFS `62` shape. Map-matching quality is:

| line_code | direction | rows | route_length_m | geometry_source | residual_m_median | residual_m_p95 | poor_match_flag |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 35 | 0 | 61103 | 15475.115 | GTFS:TransOceanico_20260315:shape_id=37737_I | 4.247 | 47.900 | False |
| 35 | 1 | 75357 | 16492.735 | GTFS:TransOceanico_20260315:shape_id=37737_V | 2.767 | 18.044 | False |
| 45 | 0 | 69851 | 7413.533 | GTFS:TransOceanico_20260315:shape_id=35604_I | 4.399 | 28.939 | False |
| 45 | 1 | 75642 | 8843.913 | GTFS:TransOceanico_20260315:shape_id=35604_V | 3.948 | 18.568 | False |
| 48 | 0 | 88061 | 18769.442 | GTFS:TransOceanico_20260315:shape_id=37712_I | 3.552 | 29.516 | False |
| 48 | 1 | 108439 | 19400.548 | GTFS:TransOceanico_20260315:shape_id=37712_V | 3.001 | 27.009 | False |
| 49.1 | 0 | 55601 | 15911.747 | line_routes.json | 6.679 | 19.797 | False |
| 49.1 | 1 | 92208 | 15911.747 | line_routes.json | 4.979 | 16.782 | False |
| 49.2 | 0 | 109783 | 21284.020 | line_routes.json | 3.888 | 17.017 | False |
| 49.2 | 1 | 57358 | 21284.020 | line_routes.json | 4.796 | 15.053 | False |
| 62 | 0 | 80387 | 19809.876 | GTFS:TransNit_20260320:shape_id=50962_I | 4.311 | 16.084 | False |
| 62 | 1 | 58064 | 16478.602 | GTFS:TransNit_20260320:shape_id=50962_V | 4.214 | 14.111 | False |

Rows with 95th percentile residual above 150 m are flagged as poor matches.

## Trajectories, Headways, and Order Parameter

Vehicle trajectories are sorted by time and resampled to a 30s grid. Interpolation
is not performed across telemetry gaps longer than 3 minutes; this run split
3,620 such gaps. Large route-position jumps over
50% of route length are also split to avoid interpolating across
terminal wrap/noisy map-matching jumps; this occurred 4,203
times.

Headways are estimated at five fixed cross-sections per route direction: 5%, 25%, 50%, 75%,
and 95% of arc length. The 5% and 95% locations are terminal proxies, chosen to avoid unstable
exact endpoint crossings. Rolling headway CV uses a 30min event-time
window. A bunching event is defined as an observed headway below 25% of the local median
observed headway for the same line, direction, day, and cross-section. This is empirical, not
a schedule-based threshold, because exact scheduled passage times at these synthetic
cross-sections were not reconstructed in Phase 1. Overall event-level bunching rate under
this definition is 0.069.

The tidy line-direction-time-bin table is saved to `/Users/pirin/Desktop/Netmob2026_data_challenge/data/outputs/phase1/phase1_line_timebin_observables.csv`.
Summary means by line and day type:

| line_code | day_category | mean_r | mean_boardings | mean_cv |
| --- | --- | --- | --- | --- |
| 35 | weekday | 0.421 | 62.940 | 0.484 |
| 35 | weekend | 0.547 | 27.988 | 0.507 |
| 45 | weekday | 0.437 | 68.391 | 0.517 |
| 45 | weekend | 0.480 | 29.247 | 0.424 |
| 48 | weekday | 0.423 | 85.367 | 0.536 |
| 48 | weekend | 0.472 | 52.469 | 0.509 |
| 49.1 | weekday | 0.588 | 69.209 | 0.441 |
| 49.1 | weekend | 0.591 | 36.027 | 0.330 |
| 49.2 | weekday | 0.503 | 81.186 | 0.525 |
| 49.2 | weekend | 0.524 | 44.425 | 0.462 |
| 62 | weekday | 0.398 | 59.950 | 0.359 |
| 62 | weekend | 0.472 | 30.537 | 0.258 |

## First Observations

1. The zero vehicle-id overlap is the main integration constraint; line-level demand is the
   defensible unit for this phase.
2. Requested focus lines are present, but the supposed low-volume contrast (`35`, `62`) is
   moderate rather than extremely quiet in this four-day slice.
3. The highest-demand and lowest-demand focus lines for the first weekday figure are
   `48` and `62`.
4. Route geometry coverage is incomplete in `line_routes.json`; GTFS fallback is necessary
   for several high-demand routes.
5. The resulting `r(t)` series and headway CV are ready for visual demand-coupling checks,
   but no critical-threshold or transition model has been fit.

## Data Issues and Phase 2 Decisions

- Recover or cite the original `data_description.pdf`; it was not present locally.
- Decide whether Phase 2 should use a per-direction closed ring, an out-and-back 2*pi cycle,
  or route-specific topology for circular lines.
- If vehicle-level demand is required, an external vehicle-id crosswalk is necessary.
- Replace the empirical bunching threshold with scheduled headways if GTFS stop-time
  interpolation to the same cross-sections is added.
- Inspect poor map-matching flags before treating route-level comparisons as causal.
