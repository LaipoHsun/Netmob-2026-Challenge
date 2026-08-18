"""Turn the descriptive results into quantitative theory tests.

Three tests, all with independently measured parameters:

  T1  Newell-Potts amplification. The load parameter k = (dwell per stop) /
      (headway) is measured directly, and the model then *predicts* the per-stop
      growth rate of headway irregularity, ln(1 + 2k), with nothing left to fit.
      Compared against the measured section-wise growth.

  T2  Kuramoto critical coupling. The natural-frequency dispersion g(omega) is
      estimated from observed round-trip times, giving K_c = 2 / (pi g(0)). If
      the coupling a bus route can plausibly supply is far below K_c, the
      absence of a mean-field transition is a prediction, not a null result.

  T3  Finite-size scaling collapse. At a genuine critical point, r_excess versus
      lambda should collapse across N-strata under the scaling ansatz
      r = N^{-b} F((lambda - lambda_c) N^{a}). Failure to collapse is evidence
      against criticality in the statistical-physics idiom rather than the
      pre-registered gate's.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from netmob import theory  # noqa: E402

DATA = ROOT / "data" / "outputs"
OUT = DATA / "week3"
CELL = ["line_code", "direction", "service_date", "hour"]


# ---------------------------------------------------------------- T1
def newell_potts_test() -> tuple[pd.DataFrame, dict]:
    dwell = pd.read_csv(DATA / "phase3" / "dwell_hourly.csv")
    cv = pd.read_csv(DATA / "phase3" / "headway_cv_by_section_hour.csv")
    topo = pd.read_csv(DATA / "phase4" / "route_topology_features.csv")
    cells = pd.read_csv(DATA / "week1_2" / "cells.csv")[CELL + ["N"]]

    # Measured growth: slope of log headway CV across the route interior.
    interior = cv[(cv.section_frac >= 0.10) & (cv.section_frac <= 0.90)].copy()
    interior = interior[(interior.headway_cv > 0) & (interior.headway_events >= 3)]
    interior["log_cv"] = np.log(interior.headway_cv)

    rows = []
    for key, grp in interior.groupby(CELL, sort=False):
        if grp.section_frac.nunique() < 5:
            continue
        slope, intercept, _, p, _ = stats.linregress(grp.section_frac, grp.log_cv)
        rows.append(dict(zip(CELL, key)) | {
            "log_cv_slope_per_route_frac": float(slope),
            "log_cv_intercept": float(intercept),
            "fit_p": float(p),
            "sections": int(grp.section_frac.nunique()),
        })
    measured = pd.DataFrame(rows)

    # Load parameter, measured not assumed.
    head = (cv.groupby(CELL)
              .agg(headway_mean_min=("headway_mean_min", "median"))
              .reset_index())
    k_frame = dwell.merge(head, on=CELL)
    k_frame["k"] = k_frame.dwell_mean_min / k_frame.headway_mean_min
    k_frame = k_frame[(k_frame.headway_mean_min.between(0.5, 60))
                      & (k_frame.dwell_events >= 5)
                      & (k_frame.k.between(0.001, 1.0))]

    df = measured.merge(k_frame[CELL + ["k", "dwell_mean_min", "headway_mean_min"]], on=CELL)
    df = df.merge(topo, on=["line_code", "direction"], how="left")
    df = df.merge(cells, on=CELL, how="left")
    df["n_stops"] = df.stop_density_per_km * df.route_length_km
    df = df[df.n_stops >= 5]

    # The route interior spans 0.8 of the route, so a slope per unit route
    # fraction becomes a per-stop rate on dividing by the stop count.
    df["measured_growth_per_stop"] = df.log_cv_slope_per_route_frac / df.n_stops
    df["predicted_growth_per_stop"] = np.log(1.0 + 2.0 * df.k)
    df["predicted_mode1_per_stop"] = [
        float(np.log(theory.newell_potts_mode1_growth(k, max(int(n), 2))))
        if np.isfinite(n) else np.nan
        for k, n in zip(df.k, df.N)
    ]

    ok = df[["measured_growth_per_stop", "predicted_growth_per_stop"]].dropna()
    corr, corr_p = stats.pearsonr(ok.measured_growth_per_stop, ok.predicted_growth_per_stop)
    summary = {
        "n_cells": int(len(df)),
        "k_median": float(df.k.median()),
        "k_p05": float(df.k.quantile(0.05)),
        "k_p95": float(df.k.quantile(0.95)),
        "measured_growth_per_stop_median": float(df.measured_growth_per_stop.median()),
        "predicted_growth_per_stop_median": float(df.predicted_growth_per_stop.median()),
        "ratio_measured_over_predicted": float(
            df.measured_growth_per_stop.median() / df.predicted_growth_per_stop.median()),
        "corr_measured_predicted": float(corr),
        "corr_p": float(corr_p),
        "frac_cells_growing": float((df.measured_growth_per_stop > 0).mean()),
        "predicted_mode1_median": float(df.predicted_mode1_per_stop.median()),
    }
    return df, summary


# ---------------------------------------------------------------- T2
def kuramoto_kc_test() -> tuple[pd.DataFrame, dict]:
    """Natural frequencies from observed end-to-end run times."""
    cross = pd.read_csv(DATA / "phase3" / "dense_crossing_events.csv")
    cross["event_time"] = pd.to_datetime(cross.event_time)
    keys = ["line_code", "direction", "vehicle_id", "service_date"]
    cross = cross.sort_values(keys + ["event_time"])
    prev = cross.groupby(keys)["section_frac"].shift()
    cross["trip_id"] = ((cross.section_frac < prev).fillna(False).astype(int)
                        .groupby([cross[c] for c in keys]).cumsum())

    trips = (cross.groupby(keys + ["trip_id"])
                  .agg(t0=("event_time", "min"), t1=("event_time", "max"),
                       s0=("section_frac", "min"), s1=("section_frac", "max"),
                       n=("section_frac", "size"))
                  .reset_index())
    # Only near-complete traversals, so run time means the same thing everywhere.
    trips = trips[(trips.n >= 9) & (trips.s0 <= 0.10) & (trips.s1 >= 0.90)]
    trips["run_min"] = (trips.t1 - trips.t0).dt.total_seconds() / 60.0
    trips = trips[trips.run_min.between(5, 240)]
    trips["omega"] = 2.0 * np.pi / trips.run_min

    rows = []
    for (line, direction), grp in trips.groupby(["line_code", "direction"]):
        if len(grp) < 50:
            continue
        res = theory.kuramoto_critical_coupling(grp.omega.to_numpy())
        rows.append({"line_code": line, "direction": direction,
                     "n_trips": len(grp),
                     "run_min_median": float(grp.run_min.median()),
                     "omega_mean": res["omega_mean"], "omega_sd": res["omega_sd"],
                     "g0": res["g0"], "K_c": res["K_c"],
                     "K_c_over_omega_mean": float(res["K_c"] / res["omega_mean"]),
                     "K_c_over_omega_sd": res["K_c_over_omega_sd"],
                     "cv_run_time": float(grp.run_min.std() / grp.run_min.mean())})
    kc = pd.DataFrame(rows)

    summary = {
        "n_line_directions": int(len(kc)),
        "n_trips_used": int(len(trips)),
        "K_c_median_rad_per_min": float(kc.K_c.median()),
        "omega_mean_median_rad_per_min": float(kc.omega_mean.median()),
        "omega_sd_median_rad_per_min": float(kc.omega_sd.median()),
        "K_c_over_omega_mean_median": float(kc.K_c_over_omega_mean.median()),
        "run_time_cv_median": float(kc.cv_run_time.median()),
    }
    return kc, summary


# ---------------------------------------------------------------- T3
def scaling_collapse_test(
    source: Path | None = None,
) -> tuple[pd.DataFrame, dict, np.ndarray]:
    """Does r_excess vs lambda collapse across N-strata under a scaling ansatz?

    Returns the collapsed curves, a summary, and the permutation-null draws of
    the improvement ratio, which are the actual evidence and are plotted.
    """
    source = source or (DATA / "week1_2" / "cells_scored_arc_uniform.csv")
    cells = pd.read_csv(source)
    df = cells[["N", "lambda_boardings_per_bus", "r_excess"]].dropna()
    df = df[(df.N >= 2) & (df.lambda_boardings_per_bus > 0)]

    strata = [(2, 3), (4, 5), (6, 8), (9, 12), (13, 40)]
    binned = []
    for lo, hi in strata:
        sub = df[(df.N >= lo) & (df.N <= hi)].copy()
        if len(sub) < 200:
            continue
        sub["lam_bin"] = pd.qcut(sub.lambda_boardings_per_bus, 12, duplicates="drop")
        agg = (sub.groupby("lam_bin", observed=True)
                  .agg(lam=("lambda_boardings_per_bus", "median"),
                       r=("r_excess", "mean"), cells=("N", "size"))
                  .reset_index(drop=True))
        agg["stratum"] = f"N{lo}-{hi}"
        agg["N_mid"] = float(sub.N.median())
        binned.append(agg)
    curves = pd.concat(binned, ignore_index=True)

    strat_id = pd.factorize(curves.stratum)[0]

    def spread(x: np.ndarray, y: np.ndarray, nbins: int = 8, min_strata: int = 3) -> float:
        """Within-bin scatter of y relative to its own range.

        Two guards make this a real collapse test rather than an optimiser
        artefact. Normalising by the range stops the fit from driving the y
        exponent to a boundary, which would shrink every value toward zero and
        report a spectacular but meaningless collapse. Requiring several strata
        per bin stops it from sliding the strata into disjoint x ranges, where
        within-bin scatter is trivially zero because each bin holds one stratum.
        """
        rng = np.ptp(y)
        if not np.isfinite(rng) or rng <= 0:
            return np.inf
        edges = np.unique(np.quantile(x, np.linspace(0, 1, nbins + 1)))
        if len(edges) < 3:
            return np.inf
        idx = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
        vals = []
        for b in range(len(edges) - 1):
            sel = idx == b
            if sel.sum() >= 3 and len(np.unique(strat_id[sel])) >= min_strata:
                vals.append(np.std(y[sel]))
        if len(vals) < 3:
            return np.inf
        return float(np.mean(vals) / rng)

    lam = curves.lam.to_numpy(float)
    r = curves.r.to_numpy(float)
    nmid = curves.N_mid.to_numpy(float)

    def cost(params: np.ndarray) -> float:
        a, b, lam_c = params
        if not (-3 < a < 3 and -3 < b < 3):
            return 1e6
        # lambda_c must lie inside the observed range to mean anything.
        if not (lam.min() <= lam_c <= lam.max()):
            return 1e6
        return spread((lam - lam_c) * nmid ** a, r * nmid ** b)

    def fit_collapse(r_vals: np.ndarray) -> tuple[float, float, float, float]:
        def c(params: np.ndarray) -> float:
            a_, b_, lc_ = params
            if not (-3 < a_ < 3 and -3 < b_ < 3):
                return 1e6
            if not (lam.min() <= lc_ <= lam.max()):
                return 1e6
            return spread((lam - lc_) * nmid ** a_, r_vals * nmid ** b_)

        starts = [np.array([a0, b0, lc]) for a0 in (-0.5, 0.0, 0.5)
                  for b0 in (-0.5, 0.0, 0.5)
                  for lc in np.quantile(lam, [0.25, 0.5, 0.75])]
        best_ = min(
            (optimize.minimize(c, x0=s, method="Nelder-Mead",
                               options={"maxiter": 4000, "xatol": 1e-4, "fatol": 1e-7})
             for s in starts),
            key=lambda res: res.fun,
        )
        a_, b_, lc_ = best_.x
        return float(a_), float(b_), float(lc_), float(best_.fun)

    a, b, lam_c, spread_scaled = fit_collapse(r)
    spread_raw = spread(lam, r)
    improvement = spread_raw / spread_scaled if spread_scaled else np.nan

    # Permutation control. Shuffling lambda within each N-stratum destroys any
    # genuine lambda dependence while preserving the N-gradient of r_excess. If
    # shuffled data collapses as well as the real data, the collapse is just the
    # N-gradient being absorbed by the exponent b, not evidence of criticality.
    rng = np.random.default_rng(20260814)
    null_improvements = []
    raw_cells = pd.read_csv(source)
    raw_cells = raw_cells[["N", "lambda_boardings_per_bus", "r_excess"]].dropna()
    raw_cells = raw_cells[(raw_cells.N >= 2) & (raw_cells.lambda_boardings_per_bus > 0)]

    for _ in range(60):
        shuffled = raw_cells.copy()
        for lo, hi in strata:
            m = (shuffled.N >= lo) & (shuffled.N <= hi)
            vals = shuffled.loc[m, "r_excess"].to_numpy()
            shuffled.loc[m, "r_excess"] = rng.permutation(vals)
        parts = []
        for lo, hi in strata:
            sub = shuffled[(shuffled.N >= lo) & (shuffled.N <= hi)].copy()
            if len(sub) < 200:
                continue
            sub["lam_bin"] = pd.qcut(sub.lambda_boardings_per_bus, 12, duplicates="drop")
            parts.append(sub.groupby("lam_bin", observed=True)
                            .agg(r=("r_excess", "mean")).reset_index(drop=True))
        r_null = pd.concat(parts, ignore_index=True).r.to_numpy(float)
        if len(r_null) != len(r):
            continue
        _, _, _, s_null = fit_collapse(r_null)
        if np.isfinite(s_null) and s_null > 0:
            null_improvements.append(spread(lam, r_null) / s_null)

    null_improvements = np.array(null_improvements)
    p_value = (float(np.mean(null_improvements >= improvement))
               if null_improvements.size else np.nan)

    curves["x_scaled"] = (lam - lam_c) * nmid ** a
    curves["y_scaled"] = r * nmid ** b

    summary = {
        "exponent_a": float(a), "exponent_b": float(b), "lambda_c": float(lam_c),
        "residual_spread_raw": float(spread_raw),
        "residual_spread_after_collapse": float(spread_scaled),
        "improvement_ratio": float(improvement),
        "null_improvement_median": (float(np.median(null_improvements))
                                    if null_improvements.size else np.nan),
        "null_improvement_p95": (float(np.percentile(null_improvements, 95))
                                 if null_improvements.size else np.nan),
        "n_null_draws": int(null_improvements.size),
        "permutation_p": p_value,
        "n_strata": int(curves.stratum.nunique()),
        "note": ("Improvement ratio must be read against the permutation control, "
                 "which preserves the N-gradient of r_excess but destroys any real "
                 "lambda dependence."),
    }
    return curves, summary, null_improvements


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    np_df, np_sum = newell_potts_test()
    np_df.to_csv(OUT / "newell_potts_test.csv", index=False)
    print("--- T1 Newell-Potts ---")
    print(json.dumps(np_sum, indent=2))

    kc_df, kc_sum = kuramoto_kc_test()
    kc_df.to_csv(OUT / "kuramoto_kc.csv", index=False)
    print("\n--- T2 Kuramoto K_c ---")
    print(json.dumps(kc_sum, indent=2))

    summaries = {"newell_potts": np_sum, "kuramoto_kc": kc_sum}
    for tag, src in [("", None),
                     ("_interior", OUT / "interior_only_cells.csv")]:
        if src is not None and not src.exists():
            print(f"\n[skip] {src.name} missing; run route_position.py first")
            continue
        sc_df, sc_sum, sc_null = scaling_collapse_test(src)
        sc_df.to_csv(OUT / f"scaling_collapse{tag}.csv", index=False)
        np.savetxt(OUT / f"scaling_collapse{tag}_null.txt", sc_null)
        summaries[f"scaling_collapse{tag or '_whole'}"] = sc_sum
        print(f"\n--- T3 scaling collapse{tag or ' (whole route)'} ---")
        print(json.dumps(sc_sum, indent=2))

    (OUT / "theory_tests_summary.json").write_text(json.dumps(summaries, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
