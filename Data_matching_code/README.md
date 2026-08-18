# Data matching and feature code

This stage reads only `data_after_cleaning/` and writes only `data_after_all_processed/`.

## Run

```bash
python Data_matching_code/build_processed_data.py
```

## What it calculates

- Ticket line-hour demand under two explicit clock bases: provider-reported BRT and the unconfirmed +3-hour diagnostic candidate.
- Ticket and Mobility vehicle persistence/route summaries.
- Conservative route-code correspondence and route selection.
- GPS supply by line, direction, date, and hour.
- Static/GTFS route geometry selection and GPS-to-route arc-length projection.
- Per-bus hourly route position and arc-length phase.
- Explicit layover, sparse-midpoint, and poor-map-match exclusion flags.
- Hourly Kuramoto order, same-N random-phase null, corrected `r_excess`, z-score, and PPC.
- Reusable line-direction-hour feature tables joining supply, synchronization indices, and line-level demand.

This stage does not fit scientific models, create publication tables, or draw figures. The vehicle-ID robustness notebooks are retained under `Archive/legacy_pipeline/vehicle_matching_audit/`; they concluded that route × time does not justify a complete physical-vehicle crosswalk without an external anchor.
