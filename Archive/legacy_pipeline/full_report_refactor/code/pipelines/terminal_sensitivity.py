"""How much of the measured synchronization is terminal accumulation?

Phase 2 already excluded terminal *layover* bus-hours (>=10 min, >=75% of samples
within 5% of an endpoint, arc span <= 150 m). What survives that filter is still
heavily concentrated at the route ends: about half of all active bus-hours sit
within 10% of a terminal.

That matters because buses queued at a terminal are not oscillators bunching
under a demand feedback -- they are vehicles waiting to be dispatched. If they
dominate r_1, the order parameter is measuring dispatch queueing rather than
Newell-Potts instability.

This script recomputes the corrected order parameter over progressively more
restrictive route interiors and reports where the conclusions change.

Caveat carried through to the output: trimming buses changes N as well as the
phase distribution, so interior-only and full-route numbers are different
estimands, not a like-for-like comparison. Both are reported with their N.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from netmob import estimators as est  # noqa: E402
from netmob import nulls, stats  # noqa: E402

OUT = ROOT / "data" / "outputs" / "week1_2"
CELL_KEYS = ["line_code", "direction", "service_date", "hour"]
TRIMS = [0.00, 0.02, 0.05, 0.10, 0.15, 0.20]
SEED = 20260814


def main() -> None:
    bus_hours = pd.read_csv(OUT / "bus_hours_with_phases.csv")
    cells = pd.read_csv(OUT / "cells.csv")
    meta = cells[CELL_KEYS + ["day_category", "analysis_eligible",
                              "lambda_boardings_per_bus"]]

    occupancy = {
        "frac_within_2pct_of_terminal": float(
            ((bus_hours.s_mid_frac < 0.02) | (bus_hours.s_mid_frac > 0.98)).mean()),
        "frac_within_5pct_of_terminal": float(
            ((bus_hours.s_mid_frac < 0.05) | (bus_hours.s_mid_frac > 0.95)).mean()),
        "frac_within_10pct_of_terminal": float(
            ((bus_hours.s_mid_frac < 0.10) | (bus_hours.s_mid_frac > 0.90)).mean()),
    }
    print("terminal occupancy among active bus-hours:")
    for k, v in occupancy.items():
        print(f"  {k}: {v:.1%}")

    rows = []
    for trim in TRIMS:
        sub = bus_hours[(bus_hours.s_mid_frac >= trim) & (bus_hours.s_mid_frac <= 1 - trim)]

        agg = []
        for key, grp in sub.groupby(CELL_KEYS, sort=False):
            arc = grp.phase_arc.to_numpy(float)
            tim = grp.phase_time.to_numpy(float)
            agg.append(dict(zip(CELL_KEYS, key)) | {
                "N": len(arc),
                "r1_arc": float(np.abs(np.mean(np.exp(1j * arc)))),
                "r1_time": float(np.abs(np.mean(np.exp(1j * tim)))),
            })
        frame = pd.DataFrame(agg).merge(meta, on=CELL_KEYS, how="left")
        frame = frame[(frame.analysis_eligible == True) & (frame.N >= 2)]  # noqa: E712
        frame["weekend"] = np.where(
            frame.day_category.astype(str).str.contains("weekend", case=False),
            "weekend", "weekday")

        table = nulls.uniform_null(int(frame.N.max()), draws=20_000, seed=SEED)
        for phase_def in ("arc", "time"):
            scored = est.add_all_estimators(
                frame, table.table, null_samples=table.samples,
                r_col=f"r1_{phase_def}", n_col="N")
            row = {
                "trim": trim,
                "phase_def": phase_def,
                "n_cells": int(len(scored)),
                "median_N": float(scored.N.median()),
                "mean_r1": float(scored[f"r1_{phase_def}"].mean()),
                "mean_r_excess": float(scored.r_excess.mean()),
                "mean_ppc": float(scored.ppc.mean()),
                "frac_cells_above_chance": float((scored.ppc > 0).mean()),
            }
            for kind in ("marginal", "n_matched"):
                res = stats.bootstrap_contrast(
                    scored, "r_excess", "weekend", "weekday", "weekend",
                    kind=kind, draws=800, seed=SEED)
                row[f"weekend_{kind}"] = res["point"]
                row[f"weekend_{kind}_lo"] = res["ci_low"]
                row[f"weekend_{kind}_hi"] = res["ci_high"]
            # Does load still predict synchronization once terminals are gone?
            ok = scored[["lambda_boardings_per_bus", "r_excess"]].dropna()
            row["corr_lambda_r_excess"] = float(
                ok.lambda_boardings_per_bus.corr(ok.r_excess)) if len(ok) > 100 else np.nan
            rows.append(row)

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "terminal_sensitivity.csv", index=False)
    pd.Series(occupancy).to_csv(OUT / "terminal_occupancy.csv")

    print("\n--- corrected synchronization vs interior trim ---")
    cols = ["trim", "phase_def", "n_cells", "median_N", "mean_r1", "mean_r_excess",
            "mean_ppc", "frac_cells_above_chance", "corr_lambda_r_excess"]
    print(out[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n--- weekend contrast vs interior trim (r_excess) ---")
    print(out[["trim", "phase_def", "weekend_marginal", "weekend_marginal_lo",
               "weekend_marginal_hi", "weekend_n_matched", "weekend_n_matched_lo",
               "weekend_n_matched_hi"]].to_string(
        index=False, float_format=lambda v: f"{v:+.4f}"))
    print(f"\nwrote {OUT / 'terminal_sensitivity.csv'}")


if __name__ == "__main__":
    main()
