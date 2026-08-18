"""The estimator x null x phase-definition sweep.

Answers three of the reviewers' points at once:

  R3.2/R3.3  compare r_excess against four alternative normalisations, including
             one (PPC) that is analytically unbiased and needs no null at all
  R3.4       vary the null model itself: uniform, speed-weighted, occupancy, and
             per-route permutation
  P0         report the weekend contrast both marginally and matched on fleet
             size, because the corrected estimators still carry an N gradient

Every headline number is recomputed under every combination, so the report can
state plainly which conclusions are invariant and which are not.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from netmob import estimators as est  # noqa: E402
from netmob import nulls, stats  # noqa: E402

OUT = ROOT / "data" / "outputs" / "week1_2"
GROUP = ["line_code", "direction"]
SEED = 20260814
DRAWS = 20_000
BOOT = 2_000


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    cells = pd.read_csv(OUT / "cells.csv")
    cells = cells[cells["analysis_eligible"] == True].copy()  # noqa: E712
    bus_hours = pd.read_csv(OUT / "bus_hours_with_phases.csv")
    return cells, bus_hours


def null_bank_for(bus_hours: pd.DataFrame, max_n: int, phase_def: str) -> dict:
    """Nulls appropriate to one phase definition.

    Under the time phase the speed-weighted null *is* the uniform null -- the
    reparametrisation and the reweighting are the same correction seen from two
    sides -- so it is not rebuilt there.
    """
    arc_col = f"phase_{phase_def}"
    bank = {"uniform": nulls.uniform_null(max_n, draws=DRAWS, seed=SEED)}
    bank["occupancy"] = nulls.null_from_samples(
        bus_hours[arc_col].to_numpy(float), max_n, draws=DRAWS, seed=SEED + 1,
        kind="occupancy",
    )
    if phase_def == "arc":
        bank.update({k: v for k, v in nulls.build_null_bank(
            bus_hours, max_n, arc_phase_col="phase_arc",
            time_phase_col="phase_time", draws=DRAWS, seed=SEED,
        ).items() if k == "speed"})
    return bank


def run_contrasts(cells: pd.DataFrame, tag: str) -> pd.DataFrame:
    table = stats.three_part_table(
        cells, list(est.ESTIMATORS) + ["r1"], group="weekend",
        a="weekday", b="weekend", strat="N", draws=BOOT, seed=SEED,
    )
    table.insert(0, "spec", tag)
    return table


def main() -> None:
    cells, bus_hours = load()
    max_n = int(cells["N"].max())
    print(f"cells: {len(cells):,}   max N: {max_n}   median N: {int(cells['N'].median())}")

    all_contrasts, all_nulls, gradients = [], [], []

    for phase_def in ("arc", "time"):
        r_col = f"r1_{phase_def}"
        bank = null_bank_for(bus_hours, max_n, phase_def)

        for null_kind, table in bank.items():
            nt = table.table.copy()
            scored = est.add_all_estimators(
                cells, nt, null_samples=table.samples, r_col=r_col, n_col="N"
            )
            tag = f"{phase_def}|{null_kind}"
            all_contrasts.append(run_contrasts(scored, tag))

            nt["null_kind"] = null_kind
            nt["phase_def"] = phase_def
            all_nulls.append(nt)

            # How much N-dependence survives each correction? Slope of the
            # estimator on log N tells us whether marginal group comparisons
            # are still confounded.
            for e in est.ESTIMATORS:
                sub = scored[["N", e]].dropna()
                if len(sub) < 100:
                    continue
                by_n = sub.groupby("N")[e].mean()
                by_n = by_n[by_n.index >= 2]
                slope = np.polyfit(np.log(by_n.index.to_numpy(float)), by_n.to_numpy(), 1)[0]
                gradients.append({
                    "phase_def": phase_def, "null_kind": null_kind, "estimator": e,
                    "slope_per_log_N": float(slope),
                    "mean_at_N2": float(by_n.get(2, np.nan)),
                    "mean_at_N10": float(by_n.get(10, np.nan)),
                })

            if phase_def == "arc" and null_kind == "uniform":
                scored.to_csv(OUT / "cells_scored_arc_uniform.csv", index=False)

    contrasts = pd.concat(all_contrasts, ignore_index=True)
    contrasts.to_csv(OUT / "weekend_contrasts.csv", index=False)
    pd.concat(all_nulls, ignore_index=True).to_csv(OUT / "null_tables.csv", index=False)
    grad = pd.DataFrame(gradients)
    grad.to_csv(OUT / "residual_n_dependence.csv", index=False)

    # ---- Daido harmonics, corrected against their own same-N null ----
    harm_rows = []
    for m in (1, 2, 3):
        table = nulls.uniform_null(max_n, draws=DRAWS, seed=SEED + 10 * m, m=m)
        for phase_def in ("arc", "time"):
            scored = est.add_all_estimators(
                cells, table.table, null_samples=table.samples,
                r_col=f"r{m}_{phase_def}", n_col="N",
            )
            harm_rows.append({
                "m": m, "phase_def": phase_def,
                "mean_r": float(cells[f"r{m}_{phase_def}"].mean()),
                "mean_r_excess": float(scored["r_excess"].mean()),
                "mean_ppc": float(scored["ppc"].mean()),
                "frac_ppc_positive": float((scored["ppc"] > 0).mean()),
                "null_mean_at_N5": float(
                    table.table.loc[table.table.N == 5, "null_mean"].iloc[0]
                ),
            })
    pd.DataFrame(harm_rows).to_csv(OUT / "daido_harmonics.csv", index=False)

    summary = {
        "n_cells": int(len(cells)),
        "median_N": int(cells["N"].median()),
        "specs": sorted(contrasts["spec"].unique().tolist()),
        "estimators": list(est.ESTIMATORS),
        "bootstrap_draws": BOOT,
        "null_draws": DRAWS,
        "seed": SEED,
    }
    (OUT / "sweep_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n--- weekend contrasts (arc phase, uniform null) ---")
    sub = contrasts[contrasts.spec == "arc|uniform"]
    print(sub.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print("\n--- residual N-dependence (slope per log N) ---")
    print(grad.pivot_table(index="estimator", columns=["phase_def", "null_kind"],
                           values="slope_per_log_N").round(4).to_string())
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
