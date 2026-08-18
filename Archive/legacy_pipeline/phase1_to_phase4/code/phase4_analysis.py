#!/usr/bin/env python3
"""
Phase 4 corridor-coupling controls and synthesis figures.

This phase builds on Phase 3 outputs. It does not reopen the critical-transition question:
Phase 3 established a smooth crossover/no-criticality verdict. The main gate here tests
whether shared-trunk cross-line synchronization survives a common-speed control.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import ttest_1samp

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from phase1_kuramoto_pipeline import (  # noqa: E402
    RouteGeometry,
    dataframe_to_markdown,
    load_gtfs_geometries,
    load_static_geometries,
    select_geometry,
)
from phase2_analysis import (  # noqa: E402
    RNG_SEED,
    SHARED_BUFFER_M,
    fixed_effect_design,
    global_lonlat_to_xy,
    random_intercept_ml,
)
from phase3_analysis import load_stops, route_samples_with_s  # noqa: E402


OUTPUT_SUBDIR = "outputs/phase4"
FIG_SUBDIR = "figures"
MAX_REASONABLE_SPEED_MPS = 25.0
MIN_PAIR_HOURS = 12
LAG_HOURS = 4
PERMUTATIONS = 500
BOOTSTRAP_DRAWS = 1000
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]


PRECOMMITTED_T1_RULE = {
    "speed_survival": (
        "The controlled shared-minus-offtrunk correlation gap must be positive, its pair "
        "bootstrap 95% CI must exclude 0, and it must retain at least 50% of the raw gap."
    ),
    "lead_lag": (
        "At least 25% of pairs must have a significant non-zero-lag cross-correlation peak "
        "(permutation p<0.05) or the signed net transfer-entropy asymmetry must exceed "
        "its pair-shuffle 95% null."
    ),
    "verdict": (
        "If both tests pass: coupling survives common-shock controls. If neither passes: "
        "largely common-shock. Otherwise: mixed."
    ),
}


def set_house_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "semibold",
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#d9d9d9",
            "grid.linewidth": 0.5,
            "grid.alpha": 0.65,
            "axes.edgecolor": "#333333",
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def mm_to_in(mm: float) -> float:
    return mm / 25.4


def save_figure(fig: mpl.figure.Figure, fig_dir: Path, stem: str, caption: str) -> None:
    fig.savefig(fig_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / f"{stem}.pdf", bbox_inches="tight")
    (fig_dir / f"{stem}_caption.txt").write_text(caption + "\n")
    plt.close(fig)


def load_inputs(root: Path) -> dict[str, Any]:
    phase2 = root / "outputs" / "phase2"
    phase3 = root / "outputs" / "phase3"
    return {
        "phase3": pd.read_csv(phase3 / "phase3_observables.csv", parse_dates=["hour"]),
        "segment": pd.read_csv(phase3 / "segment_local_order.csv", parse_dates=["hour"]),
        "phase3_pairs": pd.read_csv(phase3 / "corridor_within_pair_coupling.csv"),
        "headway_sections": pd.read_csv(phase3 / "headway_cv_by_section_hour.csv", parse_dates=["hour"]),
        "bus_hours": pd.read_csv(phase2 / "bus_hour_phase_candidates.csv", parse_dates=["hour", "first_ts", "last_ts"]),
        "line_selection": pd.read_csv(phase2 / "line_selection.csv"),
        "finite_size": pd.read_csv(phase2 / "finite_size_null_by_N.csv"),
        "p1_validation": pd.read_csv(phase3 / "phase3_observables.csv", parse_dates=["hour"]),
    }


def compute_segment_speed(segment: pd.DataFrame, bus_hours: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    ranges = (
        segment[["pair_id", "line_code", "direction", "segment_lo", "segment_hi"]]
        .drop_duplicates()
        .copy()
    )
    active = bus_hours[bus_hours["active_for_order"]].copy()
    active["line_code"] = active["line_code"].astype(str)
    active["speed_mps"] = active["s_span_m"] / (active["duration_min"] * 60.0)
    active = active[(active["duration_min"] > 0) & (active["speed_mps"] > 0) & (active["speed_mps"] <= MAX_REASONABLE_SPEED_MPS)]

    pieces = []
    for _, row in ranges.iterrows():
        bh = active[(active["line_code"] == str(row["line_code"])) & (active["direction"] == int(row["direction"]))]
        if bh.empty:
            continue
        in_seg = bh[(bh["s_mid_frac"] >= float(row["segment_lo"])) & (bh["s_mid_frac"] <= float(row["segment_hi"]))].copy()
        if in_seg.empty:
            continue
        in_seg["pair_id"] = row["pair_id"]
        pieces.append(in_seg[["pair_id", "line_code", "direction", "hour", "vehicle_id", "speed_mps"]])
    if not pieces:
        return pd.DataFrame()
    speed_obs = pd.concat(pieces, ignore_index=True)
    pair_speed = (
        speed_obs.groupby(["pair_id", "hour"], as_index=False)
        .agg(
            shared_segment_speed_mps=("speed_mps", "mean"),
            shared_segment_speed_sd_mps=("speed_mps", "std"),
            speed_bus_hours=("speed_mps", "size"),
            speed_vehicles=("vehicle_id", "nunique"),
        )
    )
    speed_obs.to_csv(output_dir / "segment_speed_bus_hour_observations.csv", index=False)
    pair_speed.to_csv(output_dir / "segment_speed_pair_hour.csv", index=False)
    return pair_speed


def pair_line_series(segment: pd.DataFrame, pair_id: str, segment_name: str) -> pd.DataFrame:
    pair = segment[(segment["pair_id"] == pair_id) & (segment["segment"] == segment_name)].copy()
    if pair.empty:
        return pd.DataFrame()
    agg = (
        pair.groupby(["line_code", "hour"], as_index=False)
        .agg(r_excess=("r_excess", "mean"), N_segment=("N_segment", "sum"))
    )
    lines = sorted(agg["line_code"].unique())
    if len(lines) < 2:
        return pd.DataFrame()
    wide_r = agg.pivot(index="hour", columns="line_code", values="r_excess")
    wide_n = agg.pivot(index="hour", columns="line_code", values="N_segment")
    out = pd.DataFrame(index=wide_r.index)
    out["a"] = wide_r[lines[0]]
    out["b"] = wide_r[lines[1]]
    out["N_a"] = wide_n[lines[0]]
    out["N_b"] = wide_n[lines[1]]
    out["line_a"] = lines[0]
    out["line_b"] = lines[1]
    return out.reset_index()


def corr_safe(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < MIN_PAIR_HOURS or np.nanstd(x) <= 1e-12 or np.nanstd(y) <= 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def residualize(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    X = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def transfer_entropy(source: np.ndarray, target: np.ndarray, bins: int = 3) -> float:
    data = pd.DataFrame({"source": source, "target": target}).dropna()
    if len(data) < 20:
        return np.nan
    s = discretize_quantile(data["source"].to_numpy(float), bins)
    t = discretize_quantile(data["target"].to_numpy(float), bins)
    x_prev = s[:-1]
    y_prev = t[:-1]
    y_now = t[1:]
    n = len(y_now)
    if n < 10:
        return np.nan
    te = 0.0
    for yn in range(bins):
        for yp in range(bins):
            for xp in range(bins):
                mask_xyz = (y_now == yn) & (y_prev == yp) & (x_prev == xp)
                p_xyz = mask_xyz.mean()
                if p_xyz <= 0:
                    continue
                mask_yp_xp = (y_prev == yp) & (x_prev == xp)
                mask_yp = y_prev == yp
                p_y_given_yx = mask_xyz.sum() / max(mask_yp_xp.sum(), 1)
                p_y_given_y = ((y_now == yn) & mask_yp).sum() / max(mask_yp.sum(), 1)
                if p_y_given_yx > 0 and p_y_given_y > 0:
                    te += p_xyz * math.log(p_y_given_yx / p_y_given_y)
    return float(te)


def discretize_quantile(values: np.ndarray, bins: int) -> np.ndarray:
    qs = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)[1:-1]))
    return np.digitize(values, qs, right=False)


def lag_summary(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    rows = []
    series = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(series) < MIN_PAIR_HOURS:
        return {
            "zero_lag_corr": np.nan,
            "best_nonzero_lag_hours": np.nan,
            "best_nonzero_lag_corr": np.nan,
            "nonzero_lag_perm_p": np.nan,
        }
    for lag in range(-LAG_HOURS, LAG_HOURS + 1):
        shifted = series["y"].shift(lag)
        tmp = pd.concat([series["x"], shifted], axis=1).dropna()
        c = corr_safe(tmp.iloc[:, 0].to_numpy(float), tmp.iloc[:, 1].to_numpy(float))
        rows.append((lag, c))
    zero = [c for lag, c in rows if lag == 0][0]
    nonzero = [(lag, c) for lag, c in rows if lag != 0 and np.isfinite(c)]
    if not nonzero:
        return {
            "zero_lag_corr": zero,
            "best_nonzero_lag_hours": np.nan,
            "best_nonzero_lag_corr": np.nan,
            "nonzero_lag_perm_p": np.nan,
        }
    best_lag, best_corr = max(nonzero, key=lambda item: abs(item[1]))
    null = []
    yvals = series["y"].to_numpy(float)
    xvals = series["x"].to_numpy(float)
    for _ in range(PERMUTATIONS):
        perm_y = rng.permutation(yvals)
        max_abs = 0.0
        perm_df = pd.DataFrame({"x": xvals, "y": perm_y})
        for lag in range(-LAG_HOURS, LAG_HOURS + 1):
            if lag == 0:
                continue
            tmp = pd.concat([perm_df["x"], perm_df["y"].shift(lag)], axis=1).dropna()
            c = corr_safe(tmp.iloc[:, 0].to_numpy(float), tmp.iloc[:, 1].to_numpy(float))
            if np.isfinite(c):
                max_abs = max(max_abs, abs(c))
        null.append(max_abs)
    p = float((np.array(null) >= abs(best_corr)).mean())
    return {
        "zero_lag_corr": zero,
        "best_nonzero_lag_hours": float(best_lag),
        "best_nonzero_lag_corr": best_corr,
        "nonzero_lag_perm_p": p,
    }


def coupling_analysis(segment: pd.DataFrame, pair_speed: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(RNG_SEED)
    pair_ids = sorted(segment["pair_id"].unique())
    rows = []
    for pair_id in pair_ids:
        shared = pair_line_series(segment, pair_id, "shared")
        off = pair_line_series(segment, pair_id, "offtrunk")
        if shared.empty or off.empty:
            continue
        shared = shared.merge(pair_speed[pair_speed["pair_id"] == pair_id], on="hour", how="left")
        shared_clean = shared.dropna(subset=["a", "b", "shared_segment_speed_mps"]).copy()
        off_clean = off.dropna(subset=["a", "b"]).copy()
        if len(shared_clean) < MIN_PAIR_HOURS or len(off_clean) < MIN_PAIR_HOURS:
            continue
        raw_shared = corr_safe(shared_clean["a"].to_numpy(float), shared_clean["b"].to_numpy(float))
        raw_off = corr_safe(off_clean["a"].to_numpy(float), off_clean["b"].to_numpy(float))
        speed = shared_clean["shared_segment_speed_mps"].to_numpy(float)
        resid_a = residualize(shared_clean["a"].to_numpy(float), speed)
        resid_b = residualize(shared_clean["b"].to_numpy(float), speed)
        controlled_shared = corr_safe(resid_a, resid_b)
        off_match = off_clean.merge(shared_clean[["hour", "shared_segment_speed_mps"]], on="hour", how="inner")
        if len(off_match) >= MIN_PAIR_HOURS:
            off_resid_a = residualize(off_match["a"].to_numpy(float), off_match["shared_segment_speed_mps"].to_numpy(float))
            off_resid_b = residualize(off_match["b"].to_numpy(float), off_match["shared_segment_speed_mps"].to_numpy(float))
            controlled_off = corr_safe(off_resid_a, off_resid_b)
        else:
            controlled_off = np.nan
        lag = lag_summary(shared_clean["a"].to_numpy(float), shared_clean["b"].to_numpy(float), rng)
        te_a_b = transfer_entropy(shared_clean["a"].to_numpy(float), shared_clean["b"].to_numpy(float))
        te_b_a = transfer_entropy(shared_clean["b"].to_numpy(float), shared_clean["a"].to_numpy(float))
        rows.append(
            {
                "pair_id": pair_id,
                "line_a": shared_clean["line_a"].iloc[0],
                "line_b": shared_clean["line_b"].iloc[0],
                "shared_hours": int(len(shared_clean)),
                "offtrunk_hours": int(len(off_clean)),
                "speed_hours": int(shared_clean["shared_segment_speed_mps"].notna().sum()),
                "mean_segment_speed_mps": float(shared_clean["shared_segment_speed_mps"].mean()),
                "mean_N_shared_a": float(shared_clean["N_a"].mean()),
                "mean_N_shared_b": float(shared_clean["N_b"].mean()),
                "raw_shared_corr": raw_shared,
                "raw_offtrunk_corr": raw_off,
                "raw_shared_minus_off": raw_shared - raw_off,
                "speed_controlled_shared_corr": controlled_shared,
                "speed_controlled_offtrunk_corr": controlled_off,
                "speed_controlled_shared_minus_off": controlled_shared - controlled_off,
                "speed_gap_retention": (controlled_shared - controlled_off) / (raw_shared - raw_off)
                if np.isfinite(raw_shared - raw_off) and abs(raw_shared - raw_off) > 1e-9
                else np.nan,
                "zero_lag_corr": lag["zero_lag_corr"],
                "best_nonzero_lag_hours": lag["best_nonzero_lag_hours"],
                "best_nonzero_lag_corr": lag["best_nonzero_lag_corr"],
                "nonzero_lag_perm_p": lag["nonzero_lag_perm_p"],
                "significant_nonzero_lag": bool(np.isfinite(lag["nonzero_lag_perm_p"]) and lag["nonzero_lag_perm_p"] < 0.05),
                "te_a_to_b": te_a_b,
                "te_b_to_a": te_b_a,
                "te_net_a_minus_b": te_a_b - te_b_a if np.isfinite(te_a_b) and np.isfinite(te_b_a) else np.nan,
            }
        )
    pair_table = pd.DataFrame(rows)
    pair_table.to_csv(output_dir / "corridor_coupling_speed_controlled.csv", index=False)
    summary = summarize_coupling(pair_table, output_dir)
    return pair_table, summary


def bootstrap_ci(values: np.ndarray, draws: int = BOOTSTRAP_DRAWS) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(RNG_SEED)
    boots = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(draws)])
    return float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def summarize_coupling(pair_table: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    raw = pair_table["raw_shared_minus_off"].dropna().to_numpy(float)
    controlled = pair_table["speed_controlled_shared_minus_off"].dropna().to_numpy(float)
    raw_ci = bootstrap_ci(raw)
    controlled_ci = bootstrap_ci(controlled)
    raw_mean = float(np.mean(raw)) if len(raw) else np.nan
    controlled_mean = float(np.mean(controlled)) if len(controlled) else np.nan
    retention = controlled_mean / raw_mean if np.isfinite(raw_mean) and abs(raw_mean) > 1e-12 else np.nan
    finite_lag_fraction = float(pair_table["significant_nonzero_lag"].mean()) if len(pair_table) else np.nan
    typical_lag = float(pair_table.loc[pair_table["significant_nonzero_lag"], "best_nonzero_lag_hours"].median()) if pair_table["significant_nonzero_lag"].any() else 0.0
    te_abs = pair_table["te_net_a_minus_b"].abs().dropna().to_numpy(float)
    te_signed = pair_table["te_net_a_minus_b"].dropna().to_numpy(float)
    te_mean_abs = float(np.mean(te_abs)) if len(te_abs) else np.nan
    te_mean_signed = float(np.mean(te_signed)) if len(te_signed) else np.nan
    rng = np.random.default_rng(RNG_SEED)
    if len(te_signed):
        signs = rng.choice([-1, 1], size=(PERMUTATIONS, len(te_signed)))
        te_null = np.abs(np.mean(signs * te_signed, axis=1))
        te_p = float((te_null >= abs(te_mean_signed)).mean())
    else:
        te_p = np.nan
    speed_survives = bool(
        np.isfinite(controlled_mean)
        and controlled_ci[0] > 0
        and np.isfinite(retention)
        and retention >= 0.5
    )
    lead_lag_support = bool(finite_lag_fraction >= 0.25 or (np.isfinite(te_p) and te_p < 0.05))
    if speed_survives and lead_lag_support:
        verdict = "coupling survives common-shock controls"
    elif (not speed_survives) and (not lead_lag_support):
        verdict = "largely common-shock"
    else:
        verdict = "mixed"
    summary = {
        "pairs_tested": int(len(pair_table)),
        "raw_gap_mean": raw_mean,
        "raw_gap_ci_low": raw_ci[0],
        "raw_gap_ci_high": raw_ci[1],
        "speed_controlled_gap_mean": controlled_mean,
        "speed_controlled_gap_ci_low": controlled_ci[0],
        "speed_controlled_gap_ci_high": controlled_ci[1],
        "controlled_gap_retention": retention,
        "fraction_significant_nonzero_lag": finite_lag_fraction,
        "typical_significant_lag_hours": typical_lag,
        "mean_abs_transfer_entropy_asymmetry": te_mean_abs,
        "mean_signed_transfer_entropy_asymmetry": te_mean_signed,
        "transfer_entropy_asymmetry_perm_p": te_p,
        "speed_survival_pass": speed_survives,
        "lead_lag_pass": lead_lag_support,
        "verdict": verdict,
    }
    (output_dir / "corridor_coupling_gate_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def route_topology_features(
    selected_lines: list[str],
    static_geoms: dict[tuple[str, int | None], RouteGeometry],
    gtfs_geoms: dict[tuple[str, int], RouteGeometry],
    root: Path,
) -> pd.DataFrame:
    stop_xy, _ = load_stops(root)
    rows = []
    for line in selected_lines:
        for direction in [0, 1]:
            geom = select_geometry(line, direction, static_geoms, gtfs_geoms)
            if geom is None:
                continue
            samples = route_samples_with_s(geom, step_m=50.0)
            xy = samples[["x", "y"]].to_numpy(float)
            route_length = float(np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1)).sum())
            endpoint_dist = float(np.linalg.norm(xy[-1] - xy[0]))
            sinuosity = route_length / endpoint_dist if endpoint_dist > 100 else np.nan
            tree = cKDTree(xy)
            stop_dist, _ = tree.query(stop_xy, k=1)
            stops_near = int((stop_dist <= SHARED_BUFFER_M).sum())
            rows.append(
                {
                    "line_code": line,
                    "direction": direction,
                    "route_length_km": route_length / 1000.0,
                    "stops_near_route": stops_near,
                    "stop_density_per_km": stops_near / (route_length / 1000.0) if route_length > 0 else np.nan,
                    "sinuosity": sinuosity,
                }
            )
    return pd.DataFrame(rows)


def amplification_driver_analysis(
    phase3: pd.DataFrame,
    headway_sections: pd.DataFrame,
    topology: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    initial = headway_sections[np.isclose(headway_sections["section_frac"], 0.10)].copy()
    initial = initial.rename(columns={"headway_cv": "initial_headway_cv", "headway_events": "initial_headway_events"})
    cols = ["line_code", "direction", "service_date", "hour", "initial_headway_cv", "initial_headway_events"]
    data = phase3.merge(initial[cols], on=["line_code", "direction", "service_date", "hour"], how="left")
    data = data.merge(topology, on=["line_code", "direction"], how="left")
    model = data[data["headway_cv_slope"].notna() & data["analysis_eligible"]].copy()
    model = model.dropna(
        subset=[
            "headway_cv_slope",
            "initial_headway_cv",
            "lambda_boardings_per_bus",
            "stop_density_per_km",
            "route_length_km",
            "sinuosity",
            "N",
        ]
    )
    model["group"] = model["line_code"].astype(str) + "_d" + model["direction"].astype(str)
    model["weekend"] = (model["day_category"] == "weekend").astype(float)
    model["rain"] = model["rain_hour"].astype(float)
    model["hour_sin"] = np.sin(2.0 * np.pi * model["hour_of_day"] / 24.0)
    model["hour_cos"] = np.cos(2.0 * np.pi * model["hour_of_day"] / 24.0)
    predictor_cols = [
        "initial_headway_cv",
        "lambda_boardings_per_bus",
        "stop_density_per_km",
        "route_length_km",
        "sinuosity",
        "N",
    ]
    names = ["intercept"]
    X_cols = [np.ones(len(model))]
    for col in predictor_cols:
        std = model[col].std(ddof=0)
        mean = model[col].mean()
        zcol = f"{col}_std"
        model[zcol] = (model[col] - mean) / std if std > 0 else 0.0
        X_cols.append(model[zcol].to_numpy(float))
        names.append(zcol)
    for col in ["hour_sin", "hour_cos", "weekend", "rain"]:
        X_cols.append(model[col].to_numpy(float))
        names.append(col)
    X = np.column_stack(X_cols)
    fit = random_intercept_ml(model["headway_cv_slope"].to_numpy(float), X, model["group"].to_numpy(str), names)
    coef = fit["coef"].copy()
    coef["ci_low"] = coef["estimate"] - 1.96 * coef["std_error"]
    coef["ci_high"] = coef["estimate"] + 1.96 * coef["std_error"]
    coef.to_csv(output_dir / "amplification_driver_coefficients.csv", index=False)
    model.to_csv(output_dir / "amplification_driver_model_frame.csv", index=False)
    summary = {
        "rows": int(len(model)),
        "groups": int(model["group"].nunique()),
        "top_effect_term": str(
            coef[coef["term"].isin([f"{c}_std" for c in predictor_cols])]
            .assign(abs_est=lambda d: d["estimate"].abs())
            .sort_values("abs_est", ascending=False)
            .iloc[0]["term"]
        )
        if len(model)
        else "",
    }
    (output_dir / "amplification_driver_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return data, coef, summary


def create_phase4_observables(
    phase3: pd.DataFrame,
    initial_topology: pd.DataFrame,
    pair_speed: pd.DataFrame,
    pair_table: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    phase4 = initial_topology.copy()
    speed_by_line = []
    pair_speed_expanded = pair_speed.merge(
        pair_table[["pair_id", "line_a", "line_b", "speed_controlled_shared_minus_off", "best_nonzero_lag_hours", "te_net_a_minus_b"]],
        on="pair_id",
        how="left",
    )
    for line_col in ["line_a", "line_b"]:
        tmp = pair_speed_expanded.rename(columns={line_col: "line_code"})
        speed_by_line.append(tmp[["line_code", "hour", "shared_segment_speed_mps", "speed_bus_hours"]])
    if speed_by_line:
        line_speed = (
            pd.concat(speed_by_line, ignore_index=True)
            .dropna(subset=["line_code"])
            .groupby(["line_code", "hour"], as_index=False)
            .agg(
                shared_segment_speed_mps=("shared_segment_speed_mps", "mean"),
                shared_segment_speed_bus_hours=("speed_bus_hours", "sum"),
            )
        )
        phase4 = phase4.merge(line_speed, on=["line_code", "hour"], how="left")
    line_pair_stats = []
    for line_col in ["line_a", "line_b"]:
        tmp = pair_table.rename(columns={line_col: "line_code"})
        line_pair_stats.append(
            tmp[
                [
                    "line_code",
                    "speed_controlled_shared_minus_off",
                    "best_nonzero_lag_hours",
                    "te_net_a_minus_b",
                    "raw_shared_minus_off",
                ]
            ]
        )
    if line_pair_stats:
        stats = (
            pd.concat(line_pair_stats, ignore_index=True)
            .dropna(subset=["line_code"])
            .groupby("line_code", as_index=False)
            .agg(
                mean_pair_raw_gap=("raw_shared_minus_off", "mean"),
                mean_pair_speed_controlled_gap=("speed_controlled_shared_minus_off", "mean"),
                mean_pair_best_lag_hours=("best_nonzero_lag_hours", "mean"),
                mean_pair_te_net=("te_net_a_minus_b", "mean"),
                corridor_pair_memberships=("raw_shared_minus_off", "size"),
            )
        )
        phase4 = phase4.merge(stats, on="line_code", how="left")
    phase4.to_csv(output_dir / "phase4_observables.csv", index=False)
    return phase4


def figure_p4_1(fig_dir: Path, pair_table: pd.DataFrame, summary: dict[str, Any]) -> None:
    fig = plt.figure(figsize=(mm_to_in(180), mm_to_in(92)), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[2.2, 0.15, 1.0])
    ax = fig.add_subplot(gs[0, 0])
    rng = np.random.default_rng(RNG_SEED)
    plot = pair_table.dropna(subset=["raw_shared_minus_off", "speed_controlled_shared_minus_off"]).copy()
    x0 = rng.normal(0, 0.015, len(plot))
    x1 = rng.normal(1, 0.015, len(plot))
    for i, row in plot.iterrows():
        ax.plot([x0[plot.index.get_loc(i)], 1 + (x1[plot.index.get_loc(i)] - 1)], [row["raw_shared_minus_off"], row["speed_controlled_shared_minus_off"]], color="#bdbdbd", linewidth=0.6, alpha=0.45)
    ax.scatter(x0, plot["raw_shared_minus_off"], s=14, color=OKABE_ITO[0], alpha=0.8)
    ax.scatter(x1, plot["speed_controlled_shared_minus_off"], s=14, color=OKABE_ITO[1], alpha=0.8)
    ax.errorbar([0, 1], [summary["raw_gap_mean"], summary["speed_controlled_gap_mean"]], yerr=[
        [summary["raw_gap_mean"] - summary["raw_gap_ci_low"], summary["speed_controlled_gap_mean"] - summary["speed_controlled_gap_ci_low"]],
        [summary["raw_gap_ci_high"] - summary["raw_gap_mean"], summary["speed_controlled_gap_ci_high"] - summary["speed_controlled_gap_mean"]],
    ], fmt="none", color="#222222", capsize=3, linewidth=1)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Raw", "Speed-controlled"])
    ax.set_ylabel("Shared minus off-trunk corr($r_{excess}$)")
    ax.text(0.02, 0.96, f"pairs n={len(plot):,}\nretention={summary['controlled_gap_retention']:.2f}", transform=ax.transAxes, va="top", fontsize=8)

    axh = fig.add_subplot(gs[0, 2])
    lags = pair_table.loc[pair_table["significant_nonzero_lag"], "best_nonzero_lag_hours"].dropna()
    axh.hist(lags, bins=np.arange(-LAG_HOURS - 0.5, LAG_HOURS + 1.5, 1), color=OKABE_ITO[2], edgecolor="white")
    axh.axvline(0, color="#333333", linewidth=0.8)
    axh.set_xlabel("Best non-zero lag (h)")
    axh.set_ylabel("Pairs")
    caption = (
        "P4-1. Pair-level shared-minus-offtrunk synchronization correlations before and after "
        "residualizing both shared-trunk line series on segment mean speed; inset shows significant non-zero lags."
    )
    save_figure(fig, fig_dir, "p4_1_corridor_speed_gate", caption)


def figure_p4_2(fig_dir: Path, pair_table: pd.DataFrame, selected_lines: list[str]) -> None:
    mat = pd.DataFrame(0.0, index=selected_lines, columns=selected_lines)
    for _, row in pair_table.dropna(subset=["te_net_a_minus_b"]).iterrows():
        a, b = row["line_a"], row["line_b"]
        if a in mat.index and b in mat.columns:
            mat.loc[a, b] = row["te_net_a_minus_b"]
            mat.loc[b, a] = -row["te_net_a_minus_b"]
    fig, ax = plt.subplots(figsize=(mm_to_in(180), mm_to_in(145)), constrained_layout=True)
    vmax = np.nanpercentile(np.abs(mat.to_numpy()), 95)
    im = ax.imshow(mat.to_numpy(float), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(selected_lines)))
    ax.set_xticklabels(selected_lines, rotation=90)
    ax.set_yticks(np.arange(len(selected_lines)))
    ax.set_yticklabels(selected_lines)
    ax.set_xlabel("Target line")
    ax.set_ylabel("Source line")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("TE(source→target) - TE(reverse)")
    caption = "P4-2. Directional transfer-entropy asymmetry matrix for shared-trunk segment-local corrected bunching series."
    save_figure(fig, fig_dir, "p4_2_lead_lag_transfer_entropy", caption)


def figure_p4_3(fig_dir: Path, coef: pd.DataFrame) -> None:
    terms = {
        "initial_headway_cv_std": "Initial CV",
        "lambda_boardings_per_bus_std": "Demand per bus",
        "stop_density_per_km_std": "Stops/km",
        "route_length_km_std": "Route length",
        "sinuosity_std": "Sinuosity",
    }
    plot = coef[coef["term"].isin(terms)].copy()
    plot["label"] = plot["term"].map(terms)
    plot = plot.iloc[::-1]
    fig, ax = plt.subplots(figsize=(mm_to_in(115), mm_to_in(78)), constrained_layout=True)
    y = np.arange(len(plot))
    ax.errorbar(plot["estimate"], y, xerr=[plot["estimate"] - plot["ci_low"], plot["ci_high"] - plot["estimate"]], fmt="o", color=OKABE_ITO[0], ecolor="#555555", capsize=3)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(plot["label"])
    ax.set_xlabel("Standardized effect on amplification slope", fontsize=8)
    caption = "P4-3. Standardized random-intercept model coefficients for spatial amplification slope; intervals are Wald 95% CIs."
    save_figure(fig, fig_dir, "p4_3_amplification_drivers", caption)


def figure_p4_4(fig_dir: Path, headway_sections: pd.DataFrame, phase4: pd.DataFrame) -> None:
    hs = headway_sections.merge(
        phase4[["line_code", "direction", "service_date", "hour", "initial_headway_cv"]],
        on=["line_code", "direction", "service_date", "hour"],
        how="left",
    ).dropna(subset=["initial_headway_cv", "headway_cv"])
    hs["initial_cv_tertile"] = pd.qcut(hs["initial_headway_cv"], 3, labels=["Low initial CV", "Mid initial CV", "High initial CV"], duplicates="drop")
    curve = hs.groupby(["initial_cv_tertile", "section_frac"], observed=True, as_index=False).agg(
        mean_cv=("headway_cv", "mean"),
        se=("headway_cv", lambda s: s.std(ddof=1) / math.sqrt(len(s)) if len(s) > 1 else 0.0),
        n=("headway_cv", "size"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(mm_to_in(180), mm_to_in(60)), sharey=True, constrained_layout=True)
    for ax, (tertile, group) in zip(axes, curve.groupby("initial_cv_tertile", observed=True)):
        ax.plot(group["section_frac"], group["mean_cv"], color=OKABE_ITO[0], linewidth=1.5)
        ax.fill_between(group["section_frac"].to_numpy(float), (group["mean_cv"] - 1.96 * group["se"]).to_numpy(float), (group["mean_cv"] + 1.96 * group["se"]).to_numpy(float), color=OKABE_ITO[0], alpha=0.18)
        ax.set_xlabel("Arc length fraction")
        ax.text(0.05, 0.92, f"n={int(group['n'].sum()):,}", transform=ax.transAxes, fontsize=8)
    axes[0].set_ylabel("Headway CV")
    caption = "P4-4. Headway-CV profiles along the route by initial-CV tertile; shaded bands are approximate 95% CIs."
    save_figure(fig, fig_dir, "p4_4_headway_cv_arc_initial_cv", caption)


def figure_p4_5(fig_dir: Path, phase4: pd.DataFrame, finite_size: pd.DataFrame, pair_table: pd.DataFrame, summary: dict[str, Any]) -> None:
    eligible = phase4[phase4["analysis_eligible"]].copy()
    fig, axes = plt.subplots(1, 3, figsize=(mm_to_in(180), mm_to_in(62)), constrained_layout=True)

    sample = eligible.sample(min(6000, len(eligible)), random_state=RNG_SEED)
    axes[0].scatter(sample["N"], sample["r1"], s=5, alpha=0.18, color=OKABE_ITO[0], rasterized=True)
    axes[0].plot(finite_size["N"], finite_size["r_null_mean"], color="#222222", linewidth=1.3)
    axes[0].fill_between(finite_size["N"].to_numpy(float), finite_size["r_null_q025"].to_numpy(float), finite_size["r_null_q975"].to_numpy(float), color="#777777", alpha=0.2)
    axes[0].set_xlabel("Active buses N")
    axes[0].set_ylabel("Raw r1")

    eligible["lambda_bin"] = pd.qcut(eligible["lambda_boardings_per_bus"], 28, duplicates="drop")
    binned = eligible.groupby("lambda_bin", observed=True, as_index=False).agg(lambda_mid=("lambda_boardings_per_bus", "median"), mean_r=("r_excess", "mean"), n=("r_excess", "size"))
    X = np.column_stack([np.ones(len(eligible)), eligible["lambda_boardings_per_bus"].to_numpy(float)])
    beta, *_ = np.linalg.lstsq(X, eligible["r_excess"].to_numpy(float), rcond=None)
    xgrid = np.linspace(eligible["lambda_boardings_per_bus"].quantile(0.01), eligible["lambda_boardings_per_bus"].quantile(0.99), 200)
    axes[1].scatter(binned["lambda_mid"], binned["mean_r"], s=np.clip(binned["n"] / 8, 10, 80), color=OKABE_ITO[2], alpha=0.8)
    axes[1].plot(xgrid, beta[0] + beta[1] * xgrid, color="#222222", linewidth=1.3)
    axes[1].axhline(0, color="#555555", linewidth=0.7)
    axes[1].set_xlabel("Load per bus")
    axes[1].set_ylabel("Corrected r_excess")

    plot = pair_table.dropna(subset=["raw_shared_minus_off", "speed_controlled_shared_minus_off"])
    axes[2].bar([0, 1], [summary["raw_gap_mean"], summary["speed_controlled_gap_mean"]], color=[OKABE_ITO[0], OKABE_ITO[1]], width=0.55)
    axes[2].errorbar([0, 1], [summary["raw_gap_mean"], summary["speed_controlled_gap_mean"]], yerr=[
        [summary["raw_gap_mean"] - summary["raw_gap_ci_low"], summary["speed_controlled_gap_mean"] - summary["speed_controlled_gap_ci_low"]],
        [summary["raw_gap_ci_high"] - summary["raw_gap_mean"], summary["speed_controlled_gap_ci_high"] - summary["speed_controlled_gap_mean"]],
    ], fmt="none", color="#222222", capsize=3, linewidth=1)
    axes[2].axhline(0, color="#555555", linewidth=0.7)
    axes[2].set_xticks([0, 1])
    axes[2].set_xticklabels(["Raw", "Speed\ncontrolled"])
    axes[2].set_ylabel("Shared - off corr")
    axes[2].text(0.04, 0.95, f"pairs n={len(plot):,}", transform=axes[2].transAxes, va="top", fontsize=8)
    caption = "P4-5. Synthesis: finite-size correction, smooth demand crossover, and corridor coupling after speed control."
    save_figure(fig, fig_dir, "p4_5_synthesis", caption)


def write_report(
    root: Path,
    output_dir: Path,
    coupling_summary: dict[str, Any],
    driver_coef: pd.DataFrame,
    driver_summary: dict[str, Any],
    phase4: pd.DataFrame,
) -> None:
    top_terms = driver_coef[driver_coef["term"].isin(
        ["initial_headway_cv_std", "lambda_boardings_per_bus_std", "stop_density_per_km_std", "route_length_km_std", "sinuosity_std"]
    )].copy()
    top_terms["abs_estimate"] = top_terms["estimate"].abs()
    top_terms = top_terms.sort_values("abs_estimate", ascending=False)
    terms_md = dataframe_to_markdown(top_terms[["term", "estimate", "ci_low", "ci_high", "p_value"]].round(4))
    top_row = top_terms.iloc[0].to_dict() if len(top_terms) else {"term": "", "estimate": np.nan}
    if top_row["term"] == "initial_headway_cv_std" and top_row["estimate"] < 0:
        driver_interpretation = (
            "Initial CV is the dominant driver, but with a negative sign: hours that are already "
            "irregular at the first interior cross-section have less additional downstream CV growth. "
            "This is consistent with saturation/ceiling behavior, not with demand-triggered growth."
        )
    elif top_row["term"] == "initial_headway_cv_std":
        driver_interpretation = (
            "Initial CV is the dominant positive driver: initial dispatch irregularity is amplified downstream."
        )
    else:
        driver_interpretation = f"The dominant standardized driver is `{top_row['term']}`."
    criteria_md = "\n".join([f"- **{k}:** {v}" for k, v in PRECOMMITTED_T1_RULE.items()])
    report = f"""# Phase 4 Common-Shock Controls and Physical Synthesis

## Pre-Committed Coupling Decision Rule

{criteria_md}

## Gate Verdict

T1 verdict: **{coupling_summary['verdict']}**. The raw shared-minus-offtrunk gap is
{coupling_summary['raw_gap_mean']:.3f} (95% bootstrap CI {coupling_summary['raw_gap_ci_low']:.3f}
to {coupling_summary['raw_gap_ci_high']:.3f}). After residualizing both shared-trunk line
series on the hourly shared-segment mean speed, the gap is
{coupling_summary['speed_controlled_gap_mean']:.3f} (95% CI
{coupling_summary['speed_controlled_gap_ci_low']:.3f} to
{coupling_summary['speed_controlled_gap_ci_high']:.3f}), retaining
{coupling_summary['controlled_gap_retention']:.2f} of the raw gap.

Lead-lag support is {coupling_summary['lead_lag_pass']}: the fraction of pairs with a
significant non-zero-lag peak is {coupling_summary['fraction_significant_nonzero_lag']:.3f},
with typical significant lag {coupling_summary['typical_significant_lag_hours']:.1f} h.
Mean absolute transfer-entropy asymmetry is
{coupling_summary['mean_abs_transfer_entropy_asymmetry']:.4f}
and signed net asymmetry is {coupling_summary['mean_signed_transfer_entropy_asymmetry']:.4f}
(signed permutation p={coupling_summary['transfer_entropy_asymmetry_perm_p']:.3f}).

Interpretation: the speed-controlled correlation gap tests against a road-speed common shock,
but the lead-lag/TE evidence decides whether the residual signal looks propagating or
symmetric. This is why the final verdict can be mixed even when the speed-controlled gap
remains positive.

## Spatial Amplification Drivers

Amplification-driver model rows: {driver_summary['rows']:,}; line-direction groups:
{driver_summary['groups']}. Standardized random-intercept coefficients:

{terms_md}

{driver_interpretation} In this fit, the spatial amplification story is therefore empirical
rather than demand-critical: amplification is tied more to the ranked driver above than to a
sharp demand threshold.

## Consolidated Physical Picture

Across Phases 1-4 the organizing principle is spatial/networked service instability, not an
equilibrium-like critical transition. Raw r must be finite-size corrected; after correction,
demand per bus has a modest positive association with bunching but the pre-committed
criticality signatures fail. Headway irregularity amplifies downstream, and shared corridors
show a small residual synchronization signal after speed control, but lead-lag evidence is
needed before interpreting this as propagating inter-line coupling rather than a common road
state.

## Abstract-Ready Claims, Final

- **Supported:** Same-N finite-size correction is mandatory before interpreting empirical bus
  Kuramoto order parameters.
- **Supported:** The Niterói data are consistent with demand-modulated bunching as a smooth
  crossover, not a sharp critical transition at city scale.
- **Supported:** Downstream headway-CV amplification is present; its driver ranking is given
  by the standardized effects above.
- **Suggestive:** Shared-trunk synchronization has a residual speed-controlled component, but
  the Phase 4 gate verdict is `{coupling_summary['verdict']}`.
- **Null:** GPS-derived dwell variability did not mediate demand -> corrected bunching in
  Phase 3.
- **Not supported:** A causal claim that passenger demand alone produces a critical
  synchronization transition.
"""
    (root / "phase4_report.md").write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("."), help="Path to data root")
    parser.add_argument("--output-dir", type=Path, default=Path(OUTPUT_SUBDIR), help="Phase 4 output directory")
    args = parser.parse_args()
    root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    fig_dir = output_dir / FIG_SUBDIR
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    set_house_style()

    inputs = load_inputs(root)
    selected_lines = inputs["line_selection"][inputs["line_selection"]["selected"]]["line_code"].astype(str).tolist()
    segment = inputs["segment"].copy()
    segment["line_code"] = segment["line_code"].astype(str)
    bus_hours = inputs["bus_hours"].copy()
    bus_hours["line_code"] = bus_hours["line_code"].astype(str)

    pair_speed = compute_segment_speed(segment, bus_hours, output_dir)
    pair_table, coupling_summary = coupling_analysis(segment, pair_speed, output_dir)

    static_geoms = load_static_geometries(root)
    gtfs_geoms = load_gtfs_geometries(root)
    topology = route_topology_features(selected_lines, static_geoms, gtfs_geoms, root)
    topology.to_csv(output_dir / "route_topology_features.csv", index=False)
    phase4_base, driver_coef, driver_summary = amplification_driver_analysis(
        inputs["phase3"], inputs["headway_sections"], topology, output_dir
    )
    phase4 = create_phase4_observables(phase4_base, phase4_base, pair_speed, pair_table, output_dir)

    figure_p4_1(fig_dir, pair_table, coupling_summary)
    figure_p4_2(fig_dir, pair_table, selected_lines)
    figure_p4_3(fig_dir, driver_coef)
    figure_p4_4(fig_dir, inputs["headway_sections"], phase4)
    figure_p4_5(fig_dir, phase4, inputs["finite_size"], pair_table, coupling_summary)

    diagnostics = {
        "precommitted_t1_rule": PRECOMMITTED_T1_RULE,
        "selected_lines": selected_lines,
        "pair_speed_rows": int(len(pair_speed)),
        "coupling_summary": coupling_summary,
        "amplification_driver_summary": driver_summary,
        "phase4_rows": int(len(phase4)),
    }
    (output_dir / "phase4_counts_and_diagnostics.json").write_text(json.dumps(diagnostics, indent=2, default=str))
    write_report(root, output_dir, coupling_summary, driver_coef, driver_summary, phase4)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "phase4_rows": int(len(phase4)),
                "pairs_tested": coupling_summary["pairs_tested"],
                "verdict": coupling_summary["verdict"],
                "top_amplification_driver": driver_summary["top_effect_term"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
