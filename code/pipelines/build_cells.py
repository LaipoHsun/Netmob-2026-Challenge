"""Rebuild the cell-level synchronization table under both phase definitions.

Inputs are the Phase 2/3 per-bus outputs, which are reproduced exactly:
aggregating bus_hour_phase_candidates.csv back to (line, direction, date, hour)
recovers phase2_hourly_observables.csv's r1 to machine precision.

Outputs one row per cell with, for each phase definition (arc length and travel
time) and each harmonic m = 1, 2, 3, the Daido order parameter r_m -- plus the
demand, weather and calendar covariates carried over from Phase 2.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from netmob import phase as phasemod  # noqa: E402

PHASE2 = ROOT / "data" / "outputs" / "phase2"
PHASE3 = ROOT / "data" / "outputs" / "phase3"
OUT = ROOT / "data" / "outputs" / "week1_2"
CELL_KEYS = ["line_code", "direction", "service_date", "hour"]
GROUP = ("line_code", "direction")
HARMONICS = (1, 2, 3)


def load_bus_hours() -> pd.DataFrame:
    df = pd.read_csv(PHASE2 / "bus_hour_phase_candidates.csv")
    df = df[df["active_for_order"] == True].copy()  # noqa: E712
    df["phase_arc"] = df["phase_mid_rad"].astype(float)
    return df


def load_crossings() -> pd.DataFrame:
    return pd.read_csv(PHASE3 / "dense_crossing_events.csv")


def load_covariates() -> pd.DataFrame:
    keep = CELL_KEYS + [
        "boardings", "lambda_boardings_per_bus", "day_category", "hour_of_day",
        "is_peak", "period", "rain_mm", "temp_c", "analysis_eligible",
    ]
    df = pd.read_csv(PHASE2 / "phase2_hourly_observables.csv")
    return df[[c for c in keep if c in df.columns]]


def validate_against_phase2(cells: pd.DataFrame) -> dict:
    reference = pd.read_csv(PHASE2 / "phase2_hourly_observables.csv")
    check = cells[CELL_KEYS + ["N_arc", "r1_arc"]].merge(
        reference[CELL_KEYS + ["N", "r1"]], on=CELL_KEYS, how="outer", indicator=True
    )
    complete = bool((check["_merge"] == "both").all())
    n_equal = bool(complete and (check["N_arc"] == check["N"]).all())
    max_abs = float(np.nanmax(np.abs(check["r1_arc"] - check["r1"]))) if complete else np.nan
    result = {
        "rebuilt_cells": int(len(cells)),
        "reference_cells": int(len(reference)),
        "all_keys_match": complete,
        "all_N_match": n_equal,
        "max_abs_r1_difference": max_abs,
        "tolerance": 1e-12,
        "passed": bool(complete and n_equal and max_abs <= 1e-12),
    }
    if not result["passed"]:
        raise AssertionError(f"cell reaggregation validation failed: {result}")
    return result


def order_parameters(group: pd.DataFrame, col: str, tag: str) -> dict:
    phi = group[col].to_numpy(float)
    phi = phi[np.isfinite(phi)]
    out = {f"N_{tag}": int(phi.size)}
    for m in HARMONICS:
        out[f"r{m}_{tag}"] = (
            float(np.abs(np.mean(np.exp(1j * m * phi)))) if phi.size else np.nan
        )
    return out


def main() -> None:
    global PHASE2, PHASE3, OUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase2-dir", type=Path, default=PHASE2)
    parser.add_argument("--phase3-dir", type=Path, default=PHASE3)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    PHASE2 = args.phase2_dir.resolve()
    PHASE3 = args.phase3_dir.resolve()
    OUT = args.out_dir.resolve()
    OUT.mkdir(parents=True, exist_ok=True)

    bus_hours = load_bus_hours()
    print(f"active bus-hours: {len(bus_hours):,}")

    crossings = load_crossings()
    profiles = phasemod.travel_time_profile(crossings, group_cols=GROUP)
    n_profiles = profiles.groupby(list(GROUP)).ngroups if len(profiles) else 0
    print(f"travel-time profiles built for {n_profiles} line-directions")
    profiles.to_csv(OUT / "travel_time_profiles.csv", index=False)

    bus_hours = phasemod.attach_time_phase(bus_hours, profiles, group_cols=GROUP)
    have_time = bus_hours["phase_time"].notna()
    print(f"time-phase coverage: {have_time.mean():.1%} of active bus-hours")
    bus_hours.to_csv(OUT / "bus_hours_with_phases.csv", index=False)

    rows = []
    for key, grp in bus_hours.groupby(CELL_KEYS, sort=False):
        row = dict(zip(CELL_KEYS, key))
        row.update(order_parameters(grp, "phase_arc", "arc"))
        sub = grp[grp["phase_time"].notna()]
        row.update(order_parameters(sub, "phase_time", "time"))
        rows.append(row)

    cells = pd.DataFrame(rows)
    cells["N"] = cells["N_arc"]
    validation = validate_against_phase2(cells)
    (OUT / "cell_reaggregation_validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    cells = cells.merge(load_covariates(), on=CELL_KEYS, how="left")
    cells["weekend"] = np.where(
        cells["day_category"].astype(str).str.contains("weekend", case=False),
        "weekend", "weekday",
    )
    cells.to_csv(OUT / "cells.csv", index=False)

    print(f"cells: {len(cells):,}  eligible: {int(cells['analysis_eligible'].sum()):,}")
    print(f"cells with a time phase for every bus: "
          f"{(cells['N_time'] == cells['N_arc']).mean():.1%}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
