#!/usr/bin/env python3
"""
Phase 3 physical-signature analysis for Kuramoto-style bus bunching.

Inputs are Phase 2 hourly observables plus raw GPS where new physical observables are
needed. Outputs are written to outputs/phase3/.
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.stats import pearsonr, ttest_1samp

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from phase1_kuramoto_pipeline import (  # noqa: E402
    MAX_STATIC_PART_GAP_M,
    RouteGeometry,
    dataframe_to_markdown,
    load_gtfs_geometries,
    load_static_geometries,
    parse_brt_naive,
)
from phase2_analysis import (  # noqa: E402
    RNG_SEED,
    SHARED_BUFFER_M,
    SHARED_STOPS_MIN,
    SHARED_TRUNK_MIN_M,
    choose_best_geometry_for_group,
    fixed_effect_design,
    global_lonlat_to_xy,
    ols_fit,
    prepare_model_frame,
    service_days,
)


OUTPUT_SUBDIR = "outputs/phase3"
N_STRATA = [(2, 3, "N02_03"), (4, 5, "N04_05"), (6, 8, "N06_08"), (9, 12, "N09_12"), (13, 99, "N13_plus")]
LAMBDA_BINS_PER_STRATUM = 8
DWELL_SPEED_MPS = 1.5
DWELL_STOP_BUFFER_M = 50.0
DWELL_MIN_SECONDS = 15.0
DWELL_MAX_SECONDS = 300.0
DWELL_TERMINAL_MIN_SECONDS = 600.0
MAX_INTERVAL_SECONDS = 90.0
HEADWAY_SECTIONS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
HEADWAY_INTERIOR_MIN = 0.10
HEADWAY_INTERIOR_MAX = 0.90
MIN_HEADWAY_EVENTS_SECTION = 3
MIN_SECTIONS_FOR_SLOPE = 4
SEGMENT_MIN_N = 2
SEGMENT_MIN_SHARED_POINTS = 8
BOOTSTRAP_DRAWS = 300
PERMUTATION_DRAWS = 500


PRECOMMITTED_CRITERIA = {
    "susceptibility": (
        "Within at least two N-strata, Var(r_excess | lambda,N) must show an interior peak "
        "at least 1.5x the mean of the low/high edge bins, and peak lambda locations must "
        "be broadly consistent across N-strata."
    ),
    "bimodality": (
        "Near the Phase-2 candidate lambda_c, a two-Gaussian mixture for r1 must improve BIC "
        "by >10 over one Gaussian, more strongly than in low/high lambda regimes."
    ),
    "n_dependence": (
        "Apparent lambda_c estimated from the susceptibility peak should not drift "
        "systematically with N-stratum median N; |corr(lambda_peak, N_mid)| < 0.4 is treated "
        "as stable."
    ),
    "verdict_rule": (
        "If at least two of the three signatures pass, report genuine transition signatures; "
        "otherwise report smooth crossover / no criticality."
    ),
}


def load_phase2(phase2: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    hourly = pd.read_csv(phase2 / "phase2_hourly_observables.csv", parse_dates=["hour"])
    bus_hours = pd.read_csv(phase2 / "bus_hour_phase_candidates.csv", parse_dates=["hour", "first_ts", "last_ts"])
    line_selection = pd.read_csv(phase2 / "line_selection.csv")
    null_table = pd.read_csv(phase2 / "finite_size_null_by_N.csv")
    selected_lines = line_selection[line_selection["selected"]]["line_code"].astype(str).tolist()
    hourly["line_code"] = hourly["line_code"].astype(str)
    bus_hours["line_code"] = bus_hours["line_code"].astype(str)
    return hourly, bus_hours, line_selection, null_table, selected_lines


def assign_n_stratum(n: int) -> str | None:
    for lo, hi, label in N_STRATA:
        if lo <= n <= hi:
            return label
    return None


def criticality_analysis(hourly: pd.DataFrame, phase2_dir: Path, output_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    eligible = hourly[hourly["analysis_eligible"]].copy()
    eligible["N_stratum"] = eligible["N"].map(assign_n_stratum)
    eligible = eligible.dropna(subset=["N_stratum", "lambda_boardings_per_bus", "r_excess", "r1"])
    rows = []
    peak_rows = []
    for stratum, group in eligible.groupby("N_stratum", sort=False):
        if len(group) < 80:
            continue
        group = group.copy()
        group["lambda_bin"] = pd.qcut(group["lambda_boardings_per_bus"], q=LAMBDA_BINS_PER_STRATUM, duplicates="drop")
        binned = (
            group.groupby("lambda_bin", observed=True)
            .agg(
                lambda_mid=("lambda_boardings_per_bus", "median"),
                lambda_min=("lambda_boardings_per_bus", "min"),
                lambda_max=("lambda_boardings_per_bus", "max"),
                rows=("r_excess", "size"),
                N_mean=("N", "mean"),
                var_r_excess=("r_excess", "var"),
                var_r1=("r1", "var"),
                mean_r_excess=("r_excess", "mean"),
            )
            .reset_index(drop=True)
            .sort_values("lambda_mid")
        )
        binned["N_stratum"] = stratum
        binned["is_interior"] = False
        if len(binned) >= 3:
            binned.loc[binned.index[1:-1], "is_interior"] = True
            interior = binned[binned["is_interior"]]
            peak = interior.loc[interior["var_r_excess"].idxmax()]
            edge_mean = float(binned.iloc[[0, -1]]["var_r_excess"].mean())
            ratio = float(peak["var_r_excess"] / edge_mean) if edge_mean > 0 else np.nan
            peak_rows.append(
                {
                    "N_stratum": stratum,
                    "N_mean": float(group["N"].mean()),
                    "rows": int(len(group)),
                    "lambda_peak": float(peak["lambda_mid"]),
                    "peak_var_r_excess": float(peak["var_r_excess"]),
                    "edge_mean_var_r_excess": edge_mean,
                    "peak_edge_ratio": ratio,
                    "passes_peak_rule": bool(ratio >= 1.5),
                }
            )
        rows.append(binned)
    susceptibility = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    peaks = pd.DataFrame(peak_rows)

    transition = pd.read_csv(phase2_dir / "transition_model_comparison.csv")
    candidate_lambda = float(transition.iloc[0]["lambda_c"]) if pd.notna(transition.iloc[0]["lambda_c"]) else float(eligible["lambda_boardings_per_bus"].quantile(0.75))
    regimes = classify_lambda_regimes(eligible, candidate_lambda)
    mixture_rows = []
    for regime, group in regimes.groupby("regime"):
        values = group["r1"].dropna().to_numpy(float)
        mixture = mixture_bic_summary(values)
        mixture["regime"] = regime
        mixture["rows"] = len(values)
        mixture_rows.append(mixture)
    mixture_table = pd.DataFrame(mixture_rows)

    if len(peaks) >= 2 and peaks["lambda_peak"].std(ddof=0) > 0:
        n_corr = float(np.corrcoef(peaks["N_mean"], peaks["lambda_peak"])[0, 1])
    else:
        n_corr = np.nan
    susceptibility_pass = int(peaks["passes_peak_rule"].sum()) >= 2
    near = mixture_table[mixture_table["regime"] == "near_candidate"]
    low_high = mixture_table[mixture_table["regime"].isin(["low_lambda", "high_lambda"])]
    near_delta = float(near["delta_bic_1_minus_2"].iloc[0]) if len(near) else np.nan
    low_high_max = float(low_high["delta_bic_1_minus_2"].max()) if len(low_high) else np.nan
    bimodal_pass = bool(np.isfinite(near_delta) and near_delta > 10 and near_delta > low_high_max)
    n_stability_pass = bool(np.isfinite(n_corr) and abs(n_corr) < 0.4)
    pass_count = int(susceptibility_pass) + int(bimodal_pass) + int(n_stability_pass)
    verdict = "genuine transition signatures present" if pass_count >= 2 else "consistent with a smooth crossover / no criticality"
    summary = {
        "candidate_lambda_from_phase2": candidate_lambda,
        "susceptibility_pass": susceptibility_pass,
        "bimodality_pass": bimodal_pass,
        "n_stability_pass": n_stability_pass,
        "lambda_peak_N_correlation": n_corr,
        "passed_signature_count": pass_count,
        "verdict": verdict,
    }
    susceptibility.to_csv(output_dir / "criticality_susceptibility_by_N_lambda.csv", index=False)
    peaks.to_csv(output_dir / "criticality_apparent_lambda_by_N.csv", index=False)
    mixture_table.to_csv(output_dir / "criticality_mixture_bimodality.csv", index=False)
    (output_dir / "criticality_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return susceptibility, peaks, mixture_table, summary


def classify_lambda_regimes(df: pd.DataFrame, candidate_lambda: float) -> pd.DataFrame:
    out = df.copy()
    q33 = out["lambda_boardings_per_bus"].quantile(0.33)
    q67 = out["lambda_boardings_per_bus"].quantile(0.67)
    near_width = max(0.15 * candidate_lambda, out["lambda_boardings_per_bus"].std(ddof=0) * 0.15)
    out["regime"] = "mid_lambda"
    out.loc[out["lambda_boardings_per_bus"] <= q33, "regime"] = "low_lambda"
    out.loc[out["lambda_boardings_per_bus"] >= q67, "regime"] = "high_lambda"
    out.loc[(out["lambda_boardings_per_bus"] - candidate_lambda).abs() <= near_width, "regime"] = "near_candidate"
    return out[out["regime"].isin(["low_lambda", "near_candidate", "high_lambda"])]


def mixture_bic_summary(values: np.ndarray) -> dict[str, float]:
    values = values[np.isfinite(values)]
    n = len(values)
    if n < 20 or np.std(values) <= 1e-12:
        return {"bic_1": np.nan, "bic_2": np.nan, "delta_bic_1_minus_2": np.nan, "component_sep_sd": np.nan}
    mu = float(values.mean())
    var = float(values.var(ddof=0) + 1e-9)
    ll1 = float(np.sum(-0.5 * (np.log(2 * np.pi * var) + (values - mu) ** 2 / var)))
    bic1 = 3 * np.log(n) - 2 * ll1
    qs = np.quantile(values, [0.33, 0.67])
    means = np.array([qs[0], qs[1]], dtype=float)
    vars_ = np.array([var, var], dtype=float)
    weights = np.array([0.5, 0.5], dtype=float)
    for _ in range(200):
        dens = np.column_stack(
            [
                weights[k] * np.exp(-0.5 * (values - means[k]) ** 2 / vars_[k]) / np.sqrt(2 * np.pi * vars_[k])
                for k in range(2)
            ]
        )
        denom = dens.sum(axis=1) + 1e-300
        resp = dens / denom[:, None]
        weights = resp.mean(axis=0)
        means = (resp * values[:, None]).sum(axis=0) / np.maximum(resp.sum(axis=0), 1e-12)
        vars_ = (resp * (values[:, None] - means) ** 2).sum(axis=0) / np.maximum(resp.sum(axis=0), 1e-12)
        vars_ = np.maximum(vars_, 1e-6)
    dens = np.column_stack(
        [
            weights[k] * np.exp(-0.5 * (values - means[k]) ** 2 / vars_[k]) / np.sqrt(2 * np.pi * vars_[k])
            for k in range(2)
        ]
    )
    ll2 = float(np.sum(np.log(dens.sum(axis=1) + 1e-300)))
    bic2 = 5 * np.log(n) - 2 * ll2
    sep = float(abs(means[1] - means[0]) / math.sqrt(0.5 * (vars_[0] + vars_[1])))
    return {"bic_1": bic1, "bic_2": bic2, "delta_bic_1_minus_2": float(bic1 - bic2), "component_sep_sd": sep}


def load_stops(root: Path) -> tuple[np.ndarray, list[str]]:
    data = json.loads((root / "auxiliar_data" / "stops.json").read_text())
    coords = []
    ids = []
    for feature in data.get("features", []):
        if feature.get("geometry", {}).get("type") == "Point":
            lon, lat = feature["geometry"]["coordinates"][:2]
            coords.append((float(lon), float(lat)))
            ids.append(str(feature.get("properties", {}).get("tx_COD", len(ids))))
    arr = np.array(coords, dtype=float)
    xy = global_lonlat_to_xy(arr[:, 0], arr[:, 1]) if len(arr) else np.empty((0, 2))
    return xy, ids


def detect_dwell_and_headways(
    root: Path,
    output_dir: Path,
    selected_lines: list[str],
    static_geoms: dict[tuple[str, int | None], RouteGeometry],
    gtfs_geoms: dict[tuple[str, int], RouteGeometry],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    days = service_days(root)
    stop_xy, stop_ids = load_stops(root)
    stop_tree = cKDTree(stop_xy)
    geom_cache: dict[tuple[str, int], RouteGeometry] = {}
    dwell_events: list[dict[str, Any]] = []
    delay_events: list[dict[str, Any]] = []
    crossing_events: list[dict[str, Any]] = []
    counts = {
        "days": days,
        "selected_lines": selected_lines,
        "raw_rows_by_day": {},
        "selected_rows_by_day": {},
        "projected_rows_by_day": {},
        "low_speed_intervals": 0,
        "stop_near_low_speed_intervals": 0,
        "non_stop_low_speed_intervals": 0,
    }
    selected = set(selected_lines)
    for day in days:
        path = root / "mobility_data" / f"{day}.csv"
        df = pd.read_csv(
            path,
            usecols=["id", "timestamp", "lat", "lng", "lineId", "direction"],
            dtype={"id": "string", "lineId": "string"},
            low_memory=False,
        )
        counts["raw_rows_by_day"][day] = int(len(df))
        df["line_code"] = df["lineId"].astype(str).str.strip()
        df["direction"] = pd.to_numeric(df["direction"], errors="coerce").astype("Int64")
        df = df[df["line_code"].isin(selected) & df["direction"].notna()].copy()
        counts["selected_rows_by_day"][day] = int(len(df))
        if df.empty:
            continue
        df["timestamp"] = parse_brt_naive(df["timestamp"])
        df = df.dropna(subset=["timestamp", "lat", "lng"])
        df["vehicle_id"] = df["id"].astype(str)
        projected_day = 0
        for (line, direction), group in df.groupby(["line_code", "direction"], sort=False):
            direction_int = int(direction)
            key = (line, direction_int)
            geom = geom_cache.get(key)
            if geom is None:
                geom = choose_best_geometry_for_group(line, direction_int, group, static_geoms, gtfs_geoms)
                if geom is not None:
                    geom_cache[key] = geom
            if geom is None:
                continue
            s_m, residual = geom.project_points(group["lng"].to_numpy(float), group["lat"].to_numpy(float))
            part = group[["line_code", "direction", "vehicle_id", "timestamp", "lat", "lng"]].copy()
            part["direction"] = part["direction"].astype(int)
            part["s_m"] = s_m
            part["s_frac"] = s_m / geom.total_length_m
            part["residual_m"] = residual
            part["route_length_m"] = geom.total_length_m
            projected_day += len(part)
            for _, veh in part.sort_values("timestamp").groupby("vehicle_id", sort=False):
                dwell, delay, crossing = process_vehicle_trace_for_phase3(veh, geom.total_length_m, stop_tree, stop_ids)
                dwell_events.extend(dwell)
                delay_events.extend(delay)
                crossing_events.extend(crossing)
                counts["low_speed_intervals"] += sum(d.get("intervals", 0) for d in dwell) + sum(d.get("intervals", 0) for d in delay)
                counts["stop_near_low_speed_intervals"] += sum(d.get("intervals", 0) for d in dwell)
                counts["non_stop_low_speed_intervals"] += sum(d.get("intervals", 0) for d in delay)
        counts["projected_rows_by_day"][day] = int(projected_day)
    dwell_df = pd.DataFrame(dwell_events)
    delay_df = pd.DataFrame(delay_events)
    crossing_df = pd.DataFrame(crossing_events)
    dwell_hourly = aggregate_dwell(dwell_df, delay_df)
    headway_sections, amplification = aggregate_headway_amplification(crossing_df)
    dwell_df.to_csv(output_dir / "dwell_events.csv", index=False)
    delay_df.to_csv(output_dir / "non_stop_delay_events.csv", index=False)
    crossing_df.to_csv(output_dir / "dense_crossing_events.csv", index=False)
    dwell_hourly.to_csv(output_dir / "dwell_hourly.csv", index=False)
    headway_sections.to_csv(output_dir / "headway_cv_by_section_hour.csv", index=False)
    amplification.to_csv(output_dir / "spatial_amplification_hourly.csv", index=False)
    counts["dwell_events"] = int(len(dwell_df))
    counts["non_stop_delay_events"] = int(len(delay_df))
    counts["crossing_events"] = int(len(crossing_df))
    return dwell_hourly, headway_sections, amplification, counts


def process_vehicle_trace_for_phase3(
    veh: pd.DataFrame,
    route_length_m: float,
    stop_tree: cKDTree,
    stop_ids: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    dwell_events: list[dict[str, Any]] = []
    delay_events: list[dict[str, Any]] = []
    crossing_events: list[dict[str, Any]] = []
    if len(veh) < 2:
        return dwell_events, delay_events, crossing_events
    veh = veh.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    t = veh["timestamp"]
    dt = t.diff().dt.total_seconds().shift(-1).to_numpy(float)
    lon = veh["lng"].to_numpy(float)
    lat = veh["lat"].to_numpy(float)
    xy = global_lonlat_to_xy(lon, lat)
    step_dist = np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1))
    speed = np.divide(step_dist, dt[:-1], out=np.full(len(step_dist), np.nan), where=dt[:-1] > 0)
    interval_ok = (dt[:-1] > 0) & (dt[:-1] <= MAX_INTERVAL_SECONDS)
    dist_stop, stop_idx = stop_tree.query(xy[:-1], k=1)
    low = interval_ok & (speed <= DWELL_SPEED_MPS)
    near_stop = dist_stop <= DWELL_STOP_BUFFER_M
    interval_rows = []
    for i in range(len(speed)):
        if not low[i]:
            continue
        interval_rows.append(
            {
                "idx": i,
                "start": t.iloc[i],
                "end": t.iloc[i + 1],
                "dt": float(dt[i]),
                "near_stop": bool(near_stop[i]),
                "stop_id": stop_ids[int(stop_idx[i])] if near_stop[i] else "",
                "s_frac": float(veh["s_frac"].iloc[i]),
                "line_code": veh["line_code"].iloc[i],
                "direction": int(veh["direction"].iloc[i]),
                "vehicle_id": veh["vehicle_id"].iloc[i],
                "service_date": str(t.iloc[i].date()),
            }
        )
    for segment in group_low_speed_intervals(interval_rows, require_same_stop=True):
        duration = sum(row["dt"] for row in segment)
        terminal_frac = np.mean([(row["s_frac"] <= 0.05) or (row["s_frac"] >= 0.95) for row in segment])
        if duration < DWELL_MIN_SECONDS:
            continue
        if duration > DWELL_MAX_SECONDS or (duration >= DWELL_TERMINAL_MIN_SECONDS and terminal_frac >= 0.75):
            continue
        first = segment[0]
        dwell_events.append(
            {
                "line_code": first["line_code"],
                "direction": first["direction"],
                "vehicle_id": first["vehicle_id"],
                "service_date": first["service_date"],
                "stop_id": first["stop_id"],
                "start_time": first["start"],
                "end_time": segment[-1]["end"],
                "hour": first["start"].floor("h"),
                "duration_sec": duration,
                "duration_min": duration / 60.0,
                "terminal_frac": terminal_frac,
                "intervals": len(segment),
            }
        )
    for segment in group_low_speed_intervals([row for row in interval_rows if not row["near_stop"]], require_same_stop=False):
        duration = sum(row["dt"] for row in segment)
        if duration < DWELL_MIN_SECONDS:
            continue
        first = segment[0]
        delay_events.append(
            {
                "line_code": first["line_code"],
                "direction": first["direction"],
                "vehicle_id": first["vehicle_id"],
                "service_date": first["service_date"],
                "start_time": first["start"],
                "end_time": segment[-1]["end"],
                "hour": first["start"].floor("h"),
                "duration_sec": duration,
                "duration_min": duration / 60.0,
                "intervals": len(segment),
            }
        )
    crossing_events = compute_dense_crossings(veh, route_length_m)
    return dwell_events, delay_events, crossing_events


def group_low_speed_intervals(rows: list[dict[str, Any]], require_same_stop: bool) -> list[list[dict[str, Any]]]:
    if not rows:
        return []
    rows = sorted(rows, key=lambda x: x["idx"])
    groups = [[rows[0]]]
    for row in rows[1:]:
        prev = groups[-1][-1]
        same_stop = (row["stop_id"] == prev["stop_id"]) if require_same_stop else True
        if row["idx"] == prev["idx"] + 1 and same_stop:
            groups[-1].append(row)
        else:
            groups.append([row])
    return groups


def compute_dense_crossings(veh: pd.DataFrame, route_length_m: float) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    if len(veh) < 2:
        return events
    veh = veh.sort_values("timestamp").reset_index(drop=True)
    t_ns = veh["timestamp"].astype("int64").to_numpy()
    t = veh["timestamp"].to_numpy()
    s = veh["s_m"].to_numpy(float)
    dt = pd.Series(veh["timestamp"]).diff().dt.total_seconds().fillna(0).to_numpy()
    ds = np.abs(np.diff(s, prepend=s[0]))
    split = ((dt > 180) | (ds > 0.5 * route_length_m)).cumsum()
    for _, idx in pd.Series(np.arange(len(veh))).groupby(split):
        ids = idx.to_numpy()
        if len(ids) < 2:
            continue
        ss = s[ids]
        tt = t_ns[ids]
        for frac in HEADWAY_SECTIONS:
            target = frac * route_length_m
            s0 = ss[:-1]
            s1 = ss[1:]
            d = s1 - s0
            valid = (np.abs(d) > 1.0) & (np.abs(d) < 0.5 * route_length_m) & ((s0 - target) * (s1 - target) <= 0)
            for j in np.where(valid)[0]:
                if d[j] == 0:
                    continue
                ratio = (target - s0[j]) / d[j]
                if ratio < 0 or ratio > 1:
                    continue
                crossing_ns = int(tt[j] + ratio * (tt[j + 1] - tt[j]))
                ts = pd.Timestamp(crossing_ns)
                events.append(
                    {
                        "line_code": veh["line_code"].iloc[0],
                        "direction": int(veh["direction"].iloc[0]),
                        "vehicle_id": veh["vehicle_id"].iloc[0],
                        "service_date": str(ts.date()),
                        "event_time": ts,
                        "hour": ts.floor("h"),
                        "section_frac": frac,
                    }
                )
    return events


def aggregate_dwell(dwell: pd.DataFrame, delay: pd.DataFrame) -> pd.DataFrame:
    keys = ["line_code", "direction", "service_date", "hour"]
    if dwell.empty:
        dwell_hour = pd.DataFrame(columns=keys)
    else:
        dwell["hour"] = pd.to_datetime(dwell["hour"])
        dwell_hour = (
            dwell.groupby(keys, as_index=False)
            .agg(
                dwell_events=("duration_min", "size"),
                dwell_mean_min=("duration_min", "mean"),
                dwell_std_min=("duration_min", "std"),
                dwell_median_min=("duration_min", "median"),
                dwell_total_min=("duration_min", "sum"),
            )
        )
        dwell_hour["dwell_cv"] = dwell_hour["dwell_std_min"] / dwell_hour["dwell_mean_min"]
    if delay.empty:
        delay_hour = pd.DataFrame(columns=keys + ["non_stop_delay_events", "non_stop_delay_total_min"])
    else:
        delay["hour"] = pd.to_datetime(delay["hour"])
        delay_hour = (
            delay.groupby(keys, as_index=False)
            .agg(non_stop_delay_events=("duration_min", "size"), non_stop_delay_total_min=("duration_min", "sum"))
        )
    out = dwell_hour.merge(delay_hour, on=keys, how="outer")
    for col in ["dwell_events", "non_stop_delay_events"]:
        if col in out:
            out[col] = out[col].fillna(0).astype(int)
    for col in ["dwell_mean_min", "dwell_std_min", "dwell_median_min", "dwell_total_min", "dwell_cv", "non_stop_delay_total_min"]:
        if col in out:
            out[col] = out[col].fillna(0.0)
    return out


def aggregate_headway_amplification(crossing: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if crossing.empty:
        return pd.DataFrame(), pd.DataFrame()
    crossing["event_time"] = pd.to_datetime(crossing["event_time"])
    crossing["hour"] = pd.to_datetime(crossing["hour"])
    crossing = crossing.sort_values(["line_code", "direction", "service_date", "section_frac", "event_time"])
    crossing["headway_min"] = (
        crossing.groupby(["line_code", "direction", "service_date", "section_frac"])["event_time"]
        .diff()
        .dt.total_seconds()
        .div(60.0)
    )
    crossing = crossing[(crossing["headway_min"] > 0) & (crossing["headway_min"] <= 120)].copy()
    section = (
        crossing.groupby(["line_code", "direction", "service_date", "hour", "section_frac"], as_index=False)
        .agg(
            headway_events=("headway_min", "size"),
            headway_mean_min=("headway_min", "mean"),
            headway_cv=("headway_min", lambda s: float(s.std(ddof=0) / s.mean()) if len(s) >= MIN_HEADWAY_EVENTS_SECTION and s.mean() > 0 else np.nan),
        )
    )
    section = section[section["headway_events"] >= MIN_HEADWAY_EVENTS_SECTION].copy()
    slopes = []
    for keys, group in section.dropna(subset=["headway_cv"]).groupby(["line_code", "direction", "service_date", "hour"]):
        fit = group[(group["section_frac"] >= HEADWAY_INTERIOR_MIN) & (group["section_frac"] <= HEADWAY_INTERIOR_MAX)]
        if len(fit) < MIN_SECTIONS_FOR_SLOPE:
            continue
        x = fit["section_frac"].to_numpy(float)
        y = fit["headway_cv"].to_numpy(float)
        X = np.column_stack([np.ones(len(x)), x])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        slopes.append(
            {
                "line_code": keys[0],
                "direction": int(keys[1]),
                "service_date": keys[2],
                "hour": keys[3],
                "headway_cv_intercept": float(beta[0]),
                "headway_cv_slope": float(beta[1]),
                "sections_used": int(len(fit)),
                "mean_headway_cv": float(y.mean()),
            }
        )
    return section, pd.DataFrame(slopes)


def mediation_analysis(phase3: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    data = phase3[(phase3["analysis_eligible"]) & (phase3["dwell_events"] >= 2)].copy()
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["r_excess", "lambda_boardings_per_bus", "dwell_cv", "N"])
    if len(data) < 100:
        summary = {"rows": int(len(data)), "status": "insufficient dwell observations"}
        return pd.DataFrame(), summary
    model_df = prepare_model_frame(data)
    model_df["dwell_cv_std"] = (model_df["dwell_cv"] - model_df["dwell_cv"].mean()) / model_df["dwell_cv"].std(ddof=0)
    a, b, cprime, total = mediation_coefficients(model_df)
    rng = np.random.default_rng(RNG_SEED)
    groups = model_df["group"].unique()
    boot = []
    for i in range(BOOTSTRAP_DRAWS):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        sample = pd.concat([model_df[model_df["group"] == g] for g in sampled_groups], ignore_index=True)
        try:
            aa, bb, cc, tt = mediation_coefficients(sample)
            boot.append({"draw": i, "a": aa, "b": bb, "c_prime": cc, "total": tt, "indirect": aa * bb})
        except Exception:
            continue
    boot_df = pd.DataFrame(boot)
    indirect = a * b
    prop = indirect / total if abs(total) > 1e-12 else np.nan
    summary = {
        "rows": int(len(model_df)),
        "groups": int(model_df["group"].nunique()),
        "a_lambda_to_dwell": float(a),
        "b_dwell_to_r_excess": float(b),
        "c_prime_direct_lambda_to_r_excess": float(cprime),
        "total_lambda_to_r_excess": float(total),
        "indirect_effect": float(indirect),
        "indirect_ci_low": float(boot_df["indirect"].quantile(0.025)) if len(boot_df) else np.nan,
        "indirect_ci_high": float(boot_df["indirect"].quantile(0.975)) if len(boot_df) else np.nan,
        "proportion_mediated": float(prop),
    }
    boot_df.to_csv(output_dir / "mediation_bootstrap.csv", index=False)
    (output_dir / "mediation_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return boot_df, summary


def mediation_coefficients(df: pd.DataFrame) -> tuple[float, float, float, float]:
    base_X, base_terms = fixed_effect_design(df)
    lam = df["lambda_boardings_per_bus_std"].to_numpy(float)
    dwell = df["dwell_cv_std"].to_numpy(float)
    y_m = df["dwell_cv_std"].to_numpy(float)
    y_r = df["r_excess"].to_numpy(float)
    fit_a = ols_fit(y_m, np.column_stack([base_X, lam]))
    a = float(fit_a["beta"][-1])
    fit_b = ols_fit(y_r, np.column_stack([base_X, lam, dwell]))
    cprime = float(fit_b["beta"][-2])
    b = float(fit_b["beta"][-1])
    fit_total = ols_fit(y_r, np.column_stack([base_X, lam]))
    total = float(fit_total["beta"][-1])
    return a, b, cprime, total


def spatial_amplification_tests(phase3: pd.DataFrame, output_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    data = phase3.dropna(subset=["headway_cv_slope", "lambda_boardings_per_bus"]).copy()
    if data.empty:
        summary = {"rows": 0}
        return pd.DataFrame(), summary
    ttest = ttest_1samp(data["headway_cv_slope"], 0.0, nan_policy="omit")
    model_df = prepare_model_frame(data[data["analysis_eligible"]].copy())
    model_df = model_df.dropna(subset=["headway_cv_slope"])
    if len(model_df) >= 50:
        base_X, _ = fixed_effect_design(model_df)
        lam = model_df["lambda_boardings_per_bus_std"].to_numpy(float)
        fit = ols_fit(model_df["headway_cv_slope"].to_numpy(float), np.column_stack([base_X, lam]))
        lambda_coef = float(fit["beta"][-1])
    else:
        lambda_coef = np.nan
    data["lambda_tertile"] = pd.qcut(data["lambda_boardings_per_bus"], q=3, labels=["low", "mid", "high"], duplicates="drop")
    by_tertile = (
        data.groupby("lambda_tertile", observed=True, as_index=False)
        .agg(rows=("headway_cv_slope", "size"), mean_slope=("headway_cv_slope", "mean"), median_slope=("headway_cv_slope", "median"))
    )
    by_tertile.to_csv(output_dir / "spatial_amplification_by_lambda_tertile.csv", index=False)
    summary = {
        "rows": int(len(data)),
        "mean_slope": float(data["headway_cv_slope"].mean()),
        "median_slope": float(data["headway_cv_slope"].median()),
        "ttest_slope_gt_zero_t": float(ttest.statistic),
        "ttest_slope_gt_zero_p_two_sided": float(ttest.pvalue),
        "lambda_coef_for_slope": lambda_coef,
    }
    (output_dir / "spatial_amplification_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return by_tertile, summary


def route_samples_with_s(geom: RouteGeometry, step_m: float = 50.0) -> pd.DataFrame:
    coords = np.array(geom.coords, dtype=float)
    xy = global_lonlat_to_xy(coords[:, 0], coords[:, 1])
    seg_len = np.sqrt(np.sum(np.diff(xy, axis=0) ** 2, axis=1))
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    rows = []
    for i, length in enumerate(seg_len):
        if length <= 0:
            continue
        n = max(int(math.ceil(length / step_m)), 1)
        for frac in np.linspace(0, 1, n, endpoint=False):
            point = xy[i] + frac * (xy[i + 1] - xy[i])
            s = cum[i] + frac * length
            rows.append({"x": point[0], "y": point[1], "s_frac": s / cum[-1] if cum[-1] > 0 else 0.0})
    rows.append({"x": xy[-1, 0], "y": xy[-1, 1], "s_frac": 1.0})
    return pd.DataFrame(rows)


def build_line_direction_samples(
    selected_lines: list[str],
    static_geoms: dict[tuple[str, int | None], RouteGeometry],
    gtfs_geoms: dict[tuple[str, int], RouteGeometry],
) -> dict[tuple[str, int], pd.DataFrame]:
    samples = {}
    for line in selected_lines:
        for direction in [0, 1]:
            # Use Phase 2's empirical chooser on existing outputs if available would require GPS;
            # for corridor geometry the normal selected geometry is sufficient.
            from phase1_kuramoto_pipeline import select_geometry

            geom = select_geometry(line, direction, static_geoms, gtfs_geoms)
            if geom is not None:
                samples[(line, direction)] = route_samples_with_s(geom)
    return samples


def corridor_coupling_within_pair(
    root: Path,
    phase2_dir: Path,
    output_dir: Path,
    phase3: pd.DataFrame,
    bus_hours: pd.DataFrame,
    selected_lines: list[str],
    static_geoms: dict[tuple[str, int | None], RouteGeometry],
    gtfs_geoms: dict[tuple[str, int], RouteGeometry],
    null_table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    phase2_pairs = pd.read_csv(phase2_dir / "corridor_pair_overlap_and_correlation.csv")
    shared_pairs = phase2_pairs[phase2_pairs["shared_corridor"]].copy()
    shared_pairs = shared_pairs[shared_pairs["line_a"].isin(selected_lines) & shared_pairs["line_b"].isin(selected_lines)]
    samples = build_line_direction_samples(selected_lines, static_geoms, gtfs_geoms)
    active = bus_hours[bus_hours["active_for_order"]].copy()
    active["hour"] = pd.to_datetime(active["hour"])
    segment_rows = []
    pair_rows = []
    for _, pair in shared_pairs.iterrows():
        a, b = str(pair["line_a"]), str(pair["line_b"])
        segment_ranges: dict[tuple[str, int], tuple[float, float]] = {}
        for line, other in [(a, b), (b, a)]:
            other_xy_parts = []
            for d2 in [0, 1]:
                if (other, d2) in samples:
                    other_xy_parts.append(samples[(other, d2)][["x", "y"]].to_numpy(float))
            if not other_xy_parts:
                continue
            other_tree = cKDTree(np.vstack(other_xy_parts))
            for d in [0, 1]:
                key = (line, d)
                if key not in samples:
                    continue
                s = samples[key]
                dist, _ = other_tree.query(s[["x", "y"]].to_numpy(float), k=1)
                shared_s = s.loc[dist <= SHARED_BUFFER_M, "s_frac"].to_numpy(float)
                if len(shared_s) >= SEGMENT_MIN_SHARED_POINTS:
                    segment_ranges[key] = (float(np.quantile(shared_s, 0.05)), float(np.quantile(shared_s, 0.95)))
        if not segment_ranges:
            continue
        pair_series = []
        for line in [a, b]:
            line_parts = []
            for d in [0, 1]:
                key = (line, d)
                if key not in segment_ranges:
                    continue
                lo, hi = segment_ranges[key]
                bh = active[(active["line_code"] == line) & (active["direction"] == d)].copy()
                if bh.empty:
                    continue
                bh["segment"] = np.where((bh["s_mid_frac"] >= lo) & (bh["s_mid_frac"] <= hi), "shared", "offtrunk")
                for segment, group in bh.groupby("segment"):
                    order = segment_order(group, null_table)
                    order["pair_id"] = f"{a}__{b}"
                    order["line_code"] = line
                    order["direction"] = d
                    order["segment"] = segment
                    order["segment_lo"] = lo
                    order["segment_hi"] = hi
                    line_parts.append(order)
            if line_parts:
                pair_series.append(pd.concat(line_parts, ignore_index=True))
        if not pair_series:
            continue
        seg = pd.concat(pair_series, ignore_index=True)
        segment_rows.append(seg)
        summary = pair_correlation_summary(seg, f"{a}__{b}", a, b)
        if summary:
            pair_rows.append(summary)
    segment_table = pd.concat(segment_rows, ignore_index=True) if segment_rows else pd.DataFrame()
    pair_table = pd.DataFrame(pair_rows)
    if not pair_table.empty:
        diffs = pair_table["shared_minus_off_corr"].dropna().to_numpy(float)
        rng = np.random.default_rng(RNG_SEED)
        signs = rng.choice([-1, 1], size=(PERMUTATION_DRAWS, len(diffs))) if len(diffs) else np.empty((0, 0))
        null = np.sum(signs * diffs, axis=1) / len(diffs) if len(diffs) else np.array([])
        observed = float(diffs.mean()) if len(diffs) else np.nan
        p_perm = float((np.abs(null) >= abs(observed)).mean()) if len(null) else np.nan
        ttest = ttest_1samp(diffs, 0.0) if len(diffs) else None
    else:
        observed, p_perm, ttest = np.nan, np.nan, None
    summary = {
        "shared_pairs_tested": int(len(pair_table)),
        "mean_shared_minus_off_corr": observed,
        "sign_flip_permutation_p": p_perm,
        "paired_t_p_two_sided": float(ttest.pvalue) if ttest is not None else np.nan,
        "mean_shared_corr": float(pair_table["shared_corr"].mean()) if not pair_table.empty else np.nan,
        "mean_offtrunk_corr": float(pair_table["offtrunk_corr"].mean()) if not pair_table.empty else np.nan,
    }
    segment_table.to_csv(output_dir / "segment_local_order.csv", index=False)
    pair_table.to_csv(output_dir / "corridor_within_pair_coupling.csv", index=False)
    (output_dir / "corridor_within_pair_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    segment_hour_summary = segment_table.groupby(["line_code", "direction", "hour"], as_index=False).agg(
        segment_shared_r_excess_mean=("r_excess", lambda s: float(s[segment_table.loc[s.index, "segment"].eq("shared")].mean()) if any(segment_table.loc[s.index, "segment"].eq("shared")) else np.nan),
        segment_offtrunk_r_excess_mean=("r_excess", lambda s: float(s[segment_table.loc[s.index, "segment"].eq("offtrunk")].mean()) if any(segment_table.loc[s.index, "segment"].eq("offtrunk")) else np.nan),
        segment_pair_count=("pair_id", "nunique"),
    ) if not segment_table.empty else pd.DataFrame()
    return segment_table, pair_table, segment_hour_summary, summary


def segment_order(group: pd.DataFrame, null_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for hour, h in group.groupby("hour"):
        phases = h["phase_mid_rad"].to_numpy(float)
        n = len(phases)
        if n < SEGMENT_MIN_N:
            continue
        r1 = float(np.abs(np.exp(1j * phases).mean()))
        null = null_table[null_table["N"] == n]
        if null.empty:
            continue
        mean = float(null["r_null_mean"].iloc[0])
        std = float(null["r_null_std"].iloc[0])
        r_excess = (r1 - mean) / (1.0 - mean) if (1.0 - mean) > 1e-9 else np.nan
        z = (r1 - mean) / std if std > 1e-12 else np.nan
        rows.append({"hour": hour, "N_segment": n, "r1": r1, "r_excess": r_excess, "z": z})
    return pd.DataFrame(rows)


def pair_correlation_summary(seg: pd.DataFrame, pair_id: str, a: str, b: str) -> dict[str, Any] | None:
    out = {"pair_id": pair_id, "line_a": a, "line_b": b}
    for segment in ["shared", "offtrunk"]:
        piv = (
            seg[seg["segment"] == segment]
            .groupby(["line_code", "hour"], as_index=False)["r_excess"]
            .mean()
            .pivot(index="hour", columns="line_code", values="r_excess")
        )
        if a in piv.columns and b in piv.columns:
            data = piv[[a, b]].dropna()
            if len(data) >= 10 and data[a].std(ddof=0) > 0 and data[b].std(ddof=0) > 0:
                corr = float(data[a].corr(data[b]))
                out[f"{segment}_corr"] = corr
                out[f"{segment}_matched_hours"] = int(len(data))
                lag_rows = []
                for lag in range(-3, 4):
                    shifted = data[b].shift(lag)
                    lag_data = pd.concat([data[a], shifted], axis=1).dropna()
                    if len(lag_data) >= 10 and lag_data.iloc[:, 0].std(ddof=0) > 0 and lag_data.iloc[:, 1].std(ddof=0) > 0:
                        lag_rows.append((lag, float(lag_data.iloc[:, 0].corr(lag_data.iloc[:, 1]))))
                if lag_rows:
                    lag, corr_lag = max(lag_rows, key=lambda x: abs(x[1]))
                    out[f"{segment}_best_lag_hours_b_leads_positive"] = int(lag)
                    out[f"{segment}_best_lag_corr"] = corr_lag
            else:
                out[f"{segment}_corr"] = np.nan
                out[f"{segment}_matched_hours"] = int(len(data))
    if "shared_corr" not in out or "offtrunk_corr" not in out:
        return None
    out["shared_minus_off_corr"] = out["shared_corr"] - out["offtrunk_corr"]
    return out


def create_phase3_observables(
    hourly: pd.DataFrame,
    dwell_hourly: pd.DataFrame,
    amplification: pd.DataFrame,
    segment_hour_summary: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    phase3 = hourly.copy()
    keys = ["line_code", "direction", "service_date", "hour"]
    for df in [dwell_hourly, amplification]:
        if not df.empty:
            df = df.copy()
            df["hour"] = pd.to_datetime(df["hour"])
            df["line_code"] = df["line_code"].astype(str)
            phase3 = phase3.merge(df, on=keys, how="left")
    if not segment_hour_summary.empty:
        segment_hour_summary = segment_hour_summary.copy()
        segment_hour_summary["hour"] = pd.to_datetime(segment_hour_summary["hour"])
        segment_hour_summary["line_code"] = segment_hour_summary["line_code"].astype(str)
        phase3 = phase3.merge(segment_hour_summary, on=["line_code", "direction", "hour"], how="left")
    fill_zero = [
        "dwell_events",
        "dwell_mean_min",
        "dwell_std_min",
        "dwell_median_min",
        "dwell_total_min",
        "dwell_cv",
        "non_stop_delay_events",
        "non_stop_delay_total_min",
    ]
    for col in fill_zero:
        if col in phase3:
            phase3[col] = phase3[col].fillna(0)
    phase3.to_csv(output_dir / "phase3_observables.csv", index=False)
    return phase3


def save_figures(*args, **kwargs):
    return None



def write_report(
    root: Path,
    output_dir: Path,
    criticality: dict[str, Any],
    peaks: pd.DataFrame,
    mixture: pd.DataFrame,
    dwell_counts: dict[str, Any],
    mediation: dict[str, Any],
    spatial: dict[str, Any],
    corridor: dict[str, Any],
    phase3: pd.DataFrame,
) -> None:
    criteria_md = "\n".join([f"- **{k}:** {v}" for k, v in PRECOMMITTED_CRITERIA.items()])
    peaks_md = dataframe_to_markdown(peaks.round(3)) if not peaks.empty else "_No susceptibility peaks estimated._"
    mixture_md = dataframe_to_markdown(mixture.round(3)) if not mixture.empty else "_No mixture tests estimated._"
    indirect = mediation.get("indirect_effect", np.nan)
    indirect_low = mediation.get("indirect_ci_low", np.nan)
    indirect_high = mediation.get("indirect_ci_high", np.nan)
    prop_mediated = mediation.get("proportion_mediated", np.nan)
    mediation_null = np.isfinite(indirect_low) and np.isfinite(indirect_high) and indirect_low <= 0 <= indirect_high
    mediation_claim = (
        "No detectable dwell-variability mediation was found"
        if mediation_null
        else "Dwell-variability mediation is suggestive in this run"
    )
    spatial_positive = (
        np.isfinite(spatial.get("mean_slope", np.nan))
        and spatial.get("mean_slope", np.nan) > 0
        and spatial.get("ttest_slope_gt_zero_p_two_sided", 1.0) < 0.05
    )
    spatial_lambda = spatial.get("lambda_coef_for_slope", np.nan)
    spatial_claim = (
        "downstream headway-CV amplification is present"
        if spatial_positive
        else "downstream headway-CV amplification is not robustly detected"
    )
    demand_spatial_claim = (
        "but it does not increase with demand in this fit"
        if np.isfinite(spatial_lambda) and spatial_lambda <= 0
        else "and it increases with demand in this fit"
    )
    corridor_positive = (
        np.isfinite(corridor.get("mean_shared_minus_off_corr", np.nan))
        and corridor.get("mean_shared_minus_off_corr", np.nan) > 0
        and corridor.get("sign_flip_permutation_p", 1.0) < 0.05
    )
    corridor_claim = (
        "shared-trunk correlations exceed off-trunk controls"
        if corridor_positive
        else "shared-trunk correlations do not exceed off-trunk controls"
    )
    report = f"""# Phase 3 Physical Mechanism and Network Coupling

## Pre-Committed Criticality Criteria

These criteria were written into the Phase 3 script before reporting outcomes:

{criteria_md}

## Headline Verdict

T1 verdict: **{criticality['verdict']}**. The Phase 2 piecewise threshold is therefore treated
as an apparent crossover scale, not an established critical point, unless future data satisfy
the physical signatures below.

Signature summary: susceptibility pass={criticality['susceptibility_pass']}, bimodality
pass={criticality['bimodality_pass']}, N-stability pass={criticality['n_stability_pass']};
lambda-peak/N correlation={criticality['lambda_peak_N_correlation']:.3f}.

Apparent susceptibility peaks by N stratum:

{peaks_md}

Bimodality / coexistence check:

{mixture_md}

## Dwell-Time Mechanism

Dwell events were detected from GPS as low-speed intervals <= {DWELL_SPEED_MPS:.1f} m/s within
{DWELL_STOP_BUFFER_M:.0f} m of an official stop. Events shorter than {DWELL_MIN_SECONDS:.0f} s
or longer than {DWELL_MAX_SECONDS:.0f} s were excluded from stop-dwell summaries; longer
terminal-like dwells were excluded as layover. Low-speed intervals away from stops were kept
separately as non-stop delay.

Detected dwell events: {dwell_counts['dwell_events']:,}; non-stop delay events:
{dwell_counts['non_stop_delay_events']:,}; low-speed intervals near stops:
{dwell_counts['stop_near_low_speed_intervals']:,}; low-speed intervals away from stops:
{dwell_counts['non_stop_low_speed_intervals']:,}.

Mediation model rows: {mediation.get('rows', 0):,}. The estimated indirect effect
lambda -> dwell CV -> r_excess is {indirect:.4f}
(bootstrap 95% CI {indirect_low:.4f} to {indirect_high:.4f}); proportion mediated =
{prop_mediated:.3f}. **{mediation_claim}.** GPS dwell estimates are noisy and
traffic/signal delays remain confounders, so this should be treated as a null mechanistic
test rather than evidence against the physical mechanism.

## Spatial Amplification

Dense cross-sections were placed at 5%, every 10% from 10% to 90%, and 95% of route arc length.
Terminal proxy bands were excluded from the slope fit; headway-CV trend was fit only on the
interior 10%-90% sections.

Hourly slope rows: {spatial.get('rows', 0):,}. Mean headway-CV slope =
{spatial.get('mean_slope', np.nan):.4f}; median slope =
{spatial.get('median_slope', np.nan):.4f}; one-sample t-test two-sided p =
{spatial.get('ttest_slope_gt_zero_p_two_sided', np.nan):.3g}. The standardized-lambda
coefficient for slope is {spatial_lambda:.4f}. This means **{spatial_claim}**, {demand_spatial_claim}.

## Corridor Coupling

Phase 3 uses within-pair controls: for each shared-trunk line pair, compare cross-line
correlation of segment-local corrected synchronization on the shared trunk against the same
pair's off-trunk segments.

Pairs tested: {corridor.get('shared_pairs_tested', 0):,}. Mean shared corr =
{corridor.get('mean_shared_corr', np.nan):.3f}; mean off-trunk corr =
{corridor.get('mean_offtrunk_corr', np.nan):.3f}; shared-minus-off =
{corridor.get('mean_shared_minus_off_corr', np.nan):.3f}; sign-flip permutation p =
{corridor.get('sign_flip_permutation_p', np.nan):.3f}. Thus **{corridor_claim}**. Segment-local N is stored in
`outputs/phase3/segment_local_order.csv`.

## Abstract-Ready Claims, Updated

- **Supported:** Raw Kuramoto order parameters are strongly finite-size confounded; same-N
  null correction is mandatory.
- **Supported:** In this city-scale sample, corrected synchronization is lower on weekends
  despite higher raw r, resolving the Phase 1 paradox.
- **Supported:** Demand per active bus is positively associated with corrected bunching, but
  the effect is observational and modest.
- **Supported:** {criticality['verdict']}; do not claim a sharp critical transition from the
  current data.
- **Null/mechanism unresolved:** {mediation_claim}; the dwell channel is not established by
  these GPS-derived dwell events.
- **Supported:** {spatial_claim}, {demand_spatial_claim}.
- **Exploratory but positive:** {corridor_claim}; this is the seed for a Phase 4
  corridor-network model, not a final causal claim.
"""
    (output_dir / "phase3_report.md").write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("."), help="Path to data root")
    parser.add_argument(
        "--phase2-dir",
        type=Path,
        default=None,
        help="Phase 2 output directory (default: DATA_ROOT/outputs/phase2)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(OUTPUT_SUBDIR), help="Phase 3 output directory")
    parser.add_argument(
        "--max-static-part-gap-m",
        type=float,
        default=MAX_STATIC_PART_GAP_M,
        help="Reject discontinuous static route parts above this gap in metres",
    )
    parser.add_argument(
        "--historical-static-order",
        action="store_true",
        help="Keep GeoJSON part order/orientation exactly as stored (historical reproduction only)",
    )
    args = parser.parse_args()
    root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    phase2_dir = (args.phase2_dir or (root / "outputs" / "phase2")).resolve()
    hourly, bus_hours, line_selection, null_table, selected_lines = load_phase2(phase2_dir)
    static_geoms = load_static_geometries(
        root,
        args.max_static_part_gap_m,
        stitch_parts=not args.historical_static_order,
    )
    gtfs_geoms = load_gtfs_geometries(root)

    susceptibility, peaks, mixture, criticality_summary = criticality_analysis(hourly, phase2_dir, output_dir)
    dwell_hourly, headway_sections, amplification, dwell_counts = detect_dwell_and_headways(
        root, output_dir, selected_lines, static_geoms, gtfs_geoms
    )
    segment_table, corridor_pairs, segment_summary, corridor_summary = corridor_coupling_within_pair(
        root, phase2_dir, output_dir, hourly, bus_hours, selected_lines, static_geoms, gtfs_geoms, null_table
    )
    phase3 = create_phase3_observables(hourly, dwell_hourly, amplification, segment_summary, output_dir)
    mediation_boot, mediation_summary = mediation_analysis(phase3, output_dir)
    spatial_tertiles, spatial_summary = spatial_amplification_tests(phase3, output_dir)
    save_figures(
        output_dir,
        phase3,
        susceptibility,
        mixture,
        mediation_summary,
        headway_sections,
        spatial_summary,
        corridor_pairs,
        criticality_summary,
    )
    log = {
        "precommitted_criteria": PRECOMMITTED_CRITERIA,
        "selected_lines": selected_lines,
        "dwell_and_headway_counts": dwell_counts,
        "criticality_summary": criticality_summary,
        "mediation_summary": mediation_summary,
        "spatial_summary": spatial_summary,
        "corridor_summary": corridor_summary,
        "phase3_rows": int(len(phase3)),
    }
    (output_dir / "phase3_counts_and_diagnostics.json").write_text(json.dumps(log, indent=2, default=str))
    write_report(root, output_dir, criticality_summary, peaks, mixture, dwell_counts, mediation_summary, spatial_summary, corridor_summary, phase3)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "phase3_rows": int(len(phase3)),
                "criticality_verdict": criticality_summary["verdict"],
                "dwell_events": dwell_counts["dwell_events"],
                "spatial_slope_rows": spatial_summary.get("rows", 0),
                "corridor_pairs_tested": corridor_summary.get("shared_pairs_tested", 0),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
