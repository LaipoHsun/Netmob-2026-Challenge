"""Rebuild the cell-level synchronization table under both phase definitions.

Inputs are the Phase 2/3 per-bus outputs, which are reproduced exactly:
aggregating bus_hour_phase_candidates.csv back to (line, direction, date, hour)
recovers phase2_hourly_observables.csv's r1 to machine precision.

Outputs one row per cell with, for each phase definition (arc length and travel
time) and each harmonic m = 1, 2, 3, the Daido order parameter r_m -- plus the
demand, weather and calendar covariates carried over from Phase 2.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from netmob import phase as phasemod  # noqa: E402

DATA = ROOT / "data"
OUT = DATA / "outputs" / "week1_2"
CELL_KEYS = ["line_code", "direction", "service_date", "hour"]
GROUP = ("line_code", "direction")
HARMONICS = (1, 2, 3)


def load_bus_hours() -> pd.DataFrame:
    df = pd.read_csv(DATA / "outputs" / "phase2" / "bus_hour_phase_candidates.csv")
    df = df[df["active_for_order"] == True].copy()  # noqa: E712
    df["phase_arc"] = df["phase_mid_rad"].astype(float)
    return df


def load_crossings() -> pd.DataFrame:
    return pd.read_csv(DATA / "outputs" / "phase3" / "dense_crossing_events.csv")


def load_covariates() -> pd.DataFrame:
    keep = CELL_KEYS + [
        "boardings", "lambda_boardings_per_bus", "day_category", "hour_of_day",
        "is_peak", "period", "rain_mm", "temp_c", "analysis_eligible",
    ]
    df = pd.read_csv(DATA / "outputs" / "phase2" / "phase2_hourly_observables.csv")
    return df[[c for c in keep if c in df.columns]]


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
