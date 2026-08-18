"""Route position as a first-class variable.

Half of all active bus-hours sit within 10% of a terminal, and the sign of the
corrected order parameter depends on whether they are included. That means the
whole-route order parameter is a mixture of two regimes with opposite signs:

  terminal zone   buses queued or turning at a route end. Clustering here is
                  dispatch scheduling, not a demand feedback.
  route interior  buses in service between terminals. This is the only place the
                  Newell-Potts mechanism can operate.

This script estimates the load effect separately in each zone, with
line-direction fixed effects, and rechecks the scaling collapse on interior-only
cells.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from netmob import estimators as est  # noqa: E402
from netmob import nulls  # noqa: E402

DATA = ROOT / "data" / "outputs"
OUT = DATA / "week3"
CELL = ["line_code", "direction", "service_date", "hour"]
SEED = 20260814


def zone_of(s: pd.Series) -> pd.Series:
    return pd.cut(np.minimum(s, 1 - s), bins=[-0.01, 0.02, 0.10, 0.51],
                  labels=["terminal", "near_terminal", "interior"])


def build_zone_cells(bus_hours: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Cell-level order parameters computed separately within each route zone."""
    bus_hours = bus_hours.copy()
    bus_hours["zone"] = zone_of(bus_hours.s_mid_frac)

    rows = []
    for key, grp in bus_hours.groupby(CELL + ["zone"], sort=False, observed=True):
        phi = grp.phase_arc.to_numpy(float)
        if len(phi) < 2:
            continue
        rows.append(dict(zip(CELL + ["zone"], key)) | {
            "N": len(phi),
            "r1": float(np.abs(np.mean(np.exp(1j * phi)))),
        })
    zc = pd.DataFrame(rows).merge(meta, on=CELL, how="left")

    table = nulls.uniform_null(int(zc.N.max()), draws=20_000, seed=SEED)
    zc = est.add_all_estimators(zc, table.table, null_samples=table.samples,
                                r_col="r1", n_col="N")
    return zc


def load_effect(df: pd.DataFrame, label: str) -> dict:
    """Standardized load effect on corrected synchronization, line-direction FE."""
    d = df[["r_excess", "ppc", "lambda_boardings_per_bus", "N",
            "line_code", "direction"]].dropna()
    if len(d) < 200:
        return {"spec": label, "n": len(d), "note": "too few cells"}
    d = d.assign(
        lam_z=(d.lambda_boardings_per_bus - d.lambda_boardings_per_bus.mean())
        / d.lambda_boardings_per_bus.std(),
        logN=np.log(d.N),
        ld=d.line_code.astype(str) + "_" + d.direction.astype(str),
    )
    out = {"spec": label, "n": int(len(d))}
    for dep in ("r_excess", "ppc"):
        m = smf.ols(f"{dep} ~ lam_z + logN + C(ld)", data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d.ld})
        out[f"{dep}_lam_coef"] = float(m.params["lam_z"])
        out[f"{dep}_lam_se"] = float(m.bse["lam_z"])
        out[f"{dep}_lam_p"] = float(m.pvalues["lam_z"])
        out[f"{dep}_lam_ci_lo"] = float(m.conf_int().loc["lam_z", 0])
        out[f"{dep}_lam_ci_hi"] = float(m.conf_int().loc["lam_z", 1])
        out[f"{dep}_mean"] = float(d[dep].mean())
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    bus_hours = pd.read_csv(DATA / "week1_2" / "bus_hours_with_phases.csv")
    cells = pd.read_csv(DATA / "week1_2" / "cells.csv")
    meta = cells[CELL + ["lambda_boardings_per_bus", "boardings", "day_category",
                         "analysis_eligible", "is_peak"]]

    # --- where are the buses ---
    zones = zone_of(bus_hours.s_mid_frac)
    occupancy = zones.value_counts(normalize=True).to_dict()
    resid = bus_hours.groupby(zones, observed=True).median_residual_m.median().to_dict()
    print("share of active bus-hours by zone:",
          {k: round(float(v), 4) for k, v in occupancy.items()})
    print("median map-matching residual (m) by zone:",
          {k: round(float(v), 2) for k, v in resid.items()})

    # --- order parameter within each zone ---
    zc = build_zone_cells(bus_hours, meta)
    zc = zc[zc.analysis_eligible == True]  # noqa: E712
    zc.to_csv(OUT / "zone_cells.csv", index=False)

    by_zone = (zc.groupby("zone", observed=True)
                 .agg(cells=("N", "size"), median_N=("N", "median"),
                      mean_r1=("r1", "mean"), mean_r_excess=("r_excess", "mean"),
                      mean_ppc=("ppc", "mean"),
                      frac_above_chance=("ppc", lambda s: float((s > 0).mean())))
                 .reset_index())
    print("\n--- corrected synchronization within each zone ---")
    print(by_zone.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    by_zone.to_csv(OUT / "synchronization_by_zone.csv", index=False)

    # --- does load predict synchronization, zone by zone ---
    effects = [load_effect(zc[zc.zone == z], f"zone={z}")
               for z in ("terminal", "near_terminal", "interior")]

    scored = pd.read_csv(DATA / "week1_2" / "cells_scored_arc_uniform.csv")
    effects.append(load_effect(scored, "whole route (as submitted)"))

    # interior-only whole-cell version, for comparability with the submitted spec
    interior = bus_hours[(bus_hours.s_mid_frac >= 0.10) & (bus_hours.s_mid_frac <= 0.90)]
    rows = []
    for key, grp in interior.groupby(CELL, sort=False):
        phi = grp.phase_arc.to_numpy(float)
        if len(phi) < 2:
            continue
        rows.append(dict(zip(CELL, key)) | {
            "N": len(phi), "r1": float(np.abs(np.mean(np.exp(1j * phi))))})
    ic = pd.DataFrame(rows).merge(meta, on=CELL, how="left")
    ic = ic[ic.analysis_eligible == True]  # noqa: E712
    tab = nulls.uniform_null(int(ic.N.max()), draws=20_000, seed=SEED)
    ic = est.add_all_estimators(ic, tab.table, null_samples=tab.samples,
                                r_col="r1", n_col="N")
    ic.to_csv(OUT / "interior_only_cells.csv", index=False)
    effects.append(load_effect(ic, "interior only (10-90%)"))

    eff = pd.DataFrame(effects)
    eff.to_csv(OUT / "load_effect_by_zone.csv", index=False)
    print("\n--- standardized load effect on r_excess (line-direction FE, clustered SE) ---")
    cols = ["spec", "n", "r_excess_lam_coef", "r_excess_lam_ci_lo",
            "r_excess_lam_ci_hi", "r_excess_lam_p", "ppc_lam_coef", "ppc_lam_p"]
    print(eff[[c for c in cols if c in eff.columns]].to_string(
        index=False, float_format=lambda v: f"{v:+.4f}"))

    summary = {
        "zone_occupancy": {str(k): float(v) for k, v in occupancy.items()},
        "zone_median_matching_residual_m": {str(k): float(v) for k, v in resid.items()},
        "note": ("Terminal and interior zones carry opposite-signed corrected "
                 "synchronization, so the whole-route order parameter is a mixture "
                 "of dispatch queueing and in-service headway dynamics."),
    }
    (OUT / "route_position_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
