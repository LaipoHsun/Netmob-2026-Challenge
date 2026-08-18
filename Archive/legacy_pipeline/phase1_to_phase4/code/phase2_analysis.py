#!/usr/bin/env python3
"""
Phase 2 finite-size-corrected synchronization analysis.

This script continues from the Phase 1 conventions but recomputes hourly observables over
the full mobility/ticketing overlap window. It intentionally avoids per-vehicle demand
because Phase 1 found 0% overlap between telemetry `id` and ticketing `vehicle_number`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.stats import norm, pearsonr

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from phase1_kuramoto_pipeline import (  # noqa: E402
    RouteGeometry,
    STATIC_GEOM_ALIASES,
    dataframe_to_markdown,
    load_gtfs_geometries,
    load_static_geometries,
    normalize_code,
    parse_brt_naive,
    select_geometry,
)


OVERLAP_START = "2026-03-11"
OVERLAP_END = "2026-03-31"
TOP_LINE_TARGET = 20
CANDIDATE_LINE_TARGET = 32
NULL_DRAWS = 5000
BOOTSTRAP_DRAWS = 300
CORRIDOR_PERMUTATIONS = 1000
RNG_SEED = 2602

TICKET_TO_CANONICAL = {"62B": "62"}
CANONICAL_TO_TICKET = {"62": ["62", "62B"]}
LAYOVER_TERMINAL_FRAC = 0.75
LAYOVER_MIN_DURATION_MIN = 10.0
LAYOVER_MAX_SPAN_M = 150.0
MIDPOINT_MAX_DISTANCE_MIN = 20.0
MAX_MEDIAN_RESIDUAL_M = 150.0
SHARED_BUFFER_M = 50.0
SHARED_TRUNK_MIN_M = 1000.0
SHARED_STOPS_MIN = 5


def service_days(root: Path) -> list[str]:
    days = []
    for p in sorted((root / "mobility_data").glob("2026-03-*.csv")):
        day = p.stem
        if OVERLAP_START <= day <= OVERLAP_END and (root / "ticket_data" / p.name).exists():
            days.append(day)
    return days


def canonical_ticket_code(value: object) -> str:
    code = normalize_code(value)
    return TICKET_TO_CANONICAL.get(code, code)


def ticket_aliases_for_line(line_code: str) -> list[str]:
    return CANONICAL_TO_TICKET.get(line_code, [line_code])


def parse_weather(root: Path) -> pd.DataFrame:
    weather = pd.read_csv(root / "auxiliar_data" / "meteorological_data.csv")
    utc = pd.to_datetime(weather["Timestamp (UTC)"], errors="coerce", utc=True)
    weather["hour"] = utc.dt.tz_convert("Etc/GMT+3").dt.tz_localize(None).dt.floor("h")
    weather["rain_mm"] = pd.to_numeric(weather["Rain (mm)"], errors="coerce").fillna(0.0)
    weather["rain_hour"] = weather["rain_mm"] > 0
    weather["temp_c"] = pd.to_numeric(weather["Temp. Ins. (C)"], errors="coerce")
    daily = weather.groupby(weather["hour"].dt.date)["rain_mm"].transform("sum")
    weather["rain_day"] = daily > 0
    keep = ["hour", "rain_mm", "rain_hour", "rain_day", "temp_c"]
    return weather[keep].dropna(subset=["hour"]).drop_duplicates("hour")


def select_top_lines(
    root: Path,
    days: list[str],
    static_geoms: dict[tuple[str, int | None], RouteGeometry],
    gtfs_geoms: dict[tuple[str, int], RouteGeometry],
) -> tuple[list[str], pd.DataFrame]:
    counts: dict[str, int] = {}
    for day in days:
        path = root / "ticket_data" / f"{day}.csv"
        for chunk in pd.read_csv(path, usecols=["route_name"], dtype={"route_name": "string"}, chunksize=500_000):
            codes = chunk["route_name"].dropna().map(canonical_ticket_code)
            vc = codes.value_counts()
            for code, count in vc.items():
                counts[code] = counts.get(code, 0) + int(count)
    rows = []
    selected: list[str] = []
    for line_code, boardings in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        has_dir0 = select_geometry(line_code, 0, static_geoms, gtfs_geoms) is not None
        has_dir1 = select_geometry(line_code, 1, static_geoms, gtfs_geoms) is not None
        usable = has_dir0 and has_dir1
        rank = len(rows) + 1
        include = usable and len(selected) < CANDIDATE_LINE_TARGET
        if include:
            selected.append(line_code)
        rows.append(
            {
                "rank_by_boardings": rank,
                "line_code": line_code,
                "ticket_codes": ",".join(ticket_aliases_for_line(line_code)),
                "boardings": boardings,
                "has_direction_0_geometry": has_dir0,
                "has_direction_1_geometry": has_dir1,
                "candidate": include,
                "selected": False,
            }
        )
    return selected, pd.DataFrame(rows)


def aggregate_demand(root: Path, days: list[str], selected_lines: list[str]) -> pd.DataFrame:
    selected = set(selected_lines)
    frames = []
    for day in days:
        path = root / "ticket_data" / f"{day}.csv"
        for chunk in pd.read_csv(
            path,
            dtype={"route_name": "string", "anon_user_id": "string"},
            chunksize=500_000,
            low_memory=False,
        ):
            chunk["line_code"] = chunk["route_name"].map(canonical_ticket_code)
            chunk = chunk[chunk["line_code"].isin(selected)].copy()
            if chunk.empty:
                continue
            chunk["timestamp"] = parse_brt_naive(chunk["transaction_date"])
            chunk = chunk.dropna(subset=["timestamp"])
            chunk["hour"] = chunk["timestamp"].dt.floor("h")
            chunk["service_date"] = day
            grouped = (
                chunk.groupby(["line_code", "service_date", "hour"], as_index=False)
                .agg(
                    boardings=("transaction_date", "size"),
                    distinct_registered_users=(
                        "anon_user_id",
                        lambda s: s[s.map(normalize_code) != "0"].nunique(),
                    ),
                    free_or_transfer_boardings=(
                        "debited_amount",
                        lambda s: int((pd.to_numeric(s, errors="coerce") == 0).sum()),
                    ),
                    day_type=("day_type", lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan),
                )
            )
            frames.append(grouped)
    if not frames:
        return pd.DataFrame()
    demand = pd.concat(frames, ignore_index=True)
    demand = (
        demand.groupby(["line_code", "service_date", "hour"], as_index=False)
        .agg(
            boardings=("boardings", "sum"),
            distinct_registered_users=("distinct_registered_users", "sum"),
            free_or_transfer_boardings=("free_or_transfer_boardings", "sum"),
            day_type=("day_type", lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan),
        )
        .sort_values(["line_code", "hour"])
    )
    demand["demand_join_level"] = "line_hour_repeated_for_each_direction"
    return demand


def process_mobility_hourly(
    root: Path,
    days: list[str],
    selected_lines: list[str],
    static_geoms: dict[tuple[str, int | None], RouteGeometry],
    gtfs_geoms: dict[tuple[str, int], RouteGeometry],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    selected = set(selected_lines)
    bus_hour_parts = []
    quality_arrays: dict[tuple[str, int, str], list[np.ndarray]] = {}
    counts: dict[str, Any] = {
        "days": days,
        "selected_lines": selected_lines,
        "raw_rows_by_day": {},
        "missing_service_rows_by_day": {},
        "selected_rows_by_day": {},
        "projected_rows_by_day": {},
        "missing_geometry_rows": [],
    }
    geom_cache: dict[tuple[str, int], RouteGeometry] = {}
    for day in days:
        path = root / "mobility_data" / f"{day}.csv"
        df = pd.read_csv(
            path,
            usecols=["id", "timestamp", "lat", "lng", "lineId", "direction"],
            dtype={"id": "string", "lineId": "string"},
            low_memory=False,
        )
        counts["raw_rows_by_day"][day] = int(len(df))
        df["line_code"] = df["lineId"].map(normalize_code)
        df["direction"] = pd.to_numeric(df["direction"], errors="coerce").astype("Int64")
        missing_service = df["line_code"].eq("") | df["direction"].isna()
        counts["missing_service_rows_by_day"][day] = int(missing_service.sum())
        df = df.loc[~missing_service & df["line_code"].isin(selected)].copy()
        counts["selected_rows_by_day"][day] = int(len(df))
        if df.empty:
            counts["projected_rows_by_day"][day] = 0
            continue
        df["timestamp"] = parse_brt_naive(df["timestamp"])
        df = df.dropna(subset=["timestamp", "lat", "lng"])
        df["service_date"] = day
        df["vehicle_id"] = df["id"].map(normalize_code)
        projected_rows = 0
        for (line_code, direction), group in df.groupby(["line_code", "direction"], sort=False):
            direction_int = int(direction)
            cache_key = (line_code, direction_int)
            geom = geom_cache.get(cache_key)
            if geom is None:
                geom = choose_best_geometry_for_group(line_code, direction_int, group, static_geoms, gtfs_geoms)
                if geom is not None:
                    geom_cache[cache_key] = geom
            if geom is None:
                counts["missing_geometry_rows"].append(
                    {"day": day, "line_code": line_code, "direction": direction_int, "rows": int(len(group))}
                )
                continue
            s_m, residual = geom.project_points(group["lng"].to_numpy(float), group["lat"].to_numpy(float))
            part = group[["line_code", "direction", "service_date", "vehicle_id", "timestamp"]].copy()
            part["direction"] = part["direction"].astype(int)
            part["s_m"] = s_m
            part["route_length_m"] = geom.total_length_m
            part["s_frac"] = part["s_m"] / geom.total_length_m
            part["phase_rad"] = (2.0 * np.pi * part["s_frac"]) % (2.0 * np.pi)
            part["residual_m"] = residual
            part["geometry_source"] = geom.source
            projected_rows += len(part)
            quality_arrays.setdefault((line_code, direction_int, geom.source), []).append(residual)
            bus_hour_parts.append(project_bus_hours(part))
        counts["projected_rows_by_day"][day] = int(projected_rows)
    bus_hours = pd.concat(bus_hour_parts, ignore_index=True) if bus_hour_parts else pd.DataFrame()
    if not bus_hours.empty:
        bus_hours["exclude_layover"] = (
            (bus_hours["samples"] >= 3)
            & (bus_hours["duration_min"] >= LAYOVER_MIN_DURATION_MIN)
            & (bus_hours["terminal_frac"] >= LAYOVER_TERMINAL_FRAC)
            & (bus_hours["s_span_m"] <= LAYOVER_MAX_SPAN_M)
        )
        bus_hours["exclude_sparse_midpoint"] = bus_hours["nearest_midpoint_min"] > MIDPOINT_MAX_DISTANCE_MIN
        bus_hours["exclude_bad_residual"] = bus_hours["median_residual_m"] > MAX_MEDIAN_RESIDUAL_M
        bus_hours["active_for_order"] = ~(
            bus_hours["exclude_layover"] | bus_hours["exclude_sparse_midpoint"] | bus_hours["exclude_bad_residual"]
        )
    quality_rows = []
    for (line_code, direction, source), arrays in quality_arrays.items():
        values = np.concatenate(arrays)
        quality_rows.append(
            {
                "line_code": line_code,
                "direction": direction,
                "geometry_source": source,
                "rows": int(len(values)),
                "residual_m_median": float(np.nanmedian(values)),
                "residual_m_p95": float(np.nanpercentile(values, 95)),
                "poor_match_flag": bool(np.nanpercentile(values, 95) > MAX_MEDIAN_RESIDUAL_M),
            }
        )
    quality = pd.DataFrame(quality_rows).sort_values(["line_code", "direction"])
    exclusion_summary = summarize_bus_hour_exclusions(bus_hours)
    counts["bus_hour_exclusion_summary"] = exclusion_summary
    return bus_hours, quality, pd.DataFrame(counts["missing_geometry_rows"]), counts


def geometry_candidates(
    line_code: str,
    direction: int,
    static_geoms: dict[tuple[str, int | None], RouteGeometry],
    gtfs_geoms: dict[tuple[str, int], RouteGeometry],
) -> list[RouteGeometry]:
    candidates: list[RouteGeometry] = []
    for key in [(line_code, direction), (line_code, None)]:
        geom = static_geoms.get(key)
        if geom is not None:
            candidates.append(geom)
    geom = gtfs_geoms.get((line_code, direction))
    if geom is not None:
        candidates.append(geom)
    for alias in STATIC_GEOM_ALIASES.get(line_code, []):
        for key in [(alias, direction), (alias, None)]:
            geom = static_geoms.get(key)
            if geom is not None:
                candidates.append(geom)
        geom = gtfs_geoms.get((alias, direction))
        if geom is not None:
            candidates.append(geom)
    fallback = select_geometry(line_code, direction, static_geoms, gtfs_geoms)
    if fallback is not None:
        candidates.append(fallback)

    unique = []
    seen = set()
    for geom in candidates:
        key = (geom.source, round(geom.total_length_m, 3))
        if key not in seen:
            seen.add(key)
            unique.append(geom)
    return unique


def choose_best_geometry_for_group(
    line_code: str,
    direction: int,
    group: pd.DataFrame,
    static_geoms: dict[tuple[str, int | None], RouteGeometry],
    gtfs_geoms: dict[tuple[str, int], RouteGeometry],
) -> RouteGeometry | None:
    candidates = geometry_candidates(line_code, direction, static_geoms, gtfs_geoms)
    if not candidates:
        return None
    sample = group
    if len(sample) > 5000:
        sample = sample.sample(5000, random_state=RNG_SEED)
    lon = sample["lng"].to_numpy(float)
    lat = sample["lat"].to_numpy(float)
    scored: list[tuple[float, float, RouteGeometry]] = []
    for geom in candidates:
        _, residual = geom.project_points(lon, lat)
        scored.append((float(np.nanpercentile(residual, 95)), float(np.nanmedian(residual)), geom))
    scored.sort(key=lambda item: (item[0], item[1]))
    return scored[0][2]


def project_bus_hours(part: pd.DataFrame) -> pd.DataFrame:
    part = part.copy()
    part["hour"] = part["timestamp"].dt.floor("h")
    midpoint = part["hour"] + pd.Timedelta(minutes=30)
    part["abs_midpoint_min"] = (part["timestamp"] - midpoint).abs().dt.total_seconds() / 60.0
    part["terminal_sample"] = (part["s_frac"] <= 0.05) | (part["s_frac"] >= 0.95)
    keys = ["line_code", "direction", "service_date", "vehicle_id", "hour"]
    representative = (
        part.sort_values(keys + ["abs_midpoint_min"])
        .drop_duplicates(keys)
        [keys + ["phase_rad", "s_m", "s_frac", "abs_midpoint_min"]]
        .rename(
            columns={
                "phase_rad": "phase_mid_rad",
                "s_m": "s_mid_m",
                "s_frac": "s_mid_frac",
                "abs_midpoint_min": "nearest_midpoint_min",
            }
        )
    )
    grouped = (
        part.groupby(keys, as_index=False)
        .agg(
            samples=("phase_rad", "size"),
            first_ts=("timestamp", "min"),
            last_ts=("timestamp", "max"),
            s_min_m=("s_m", "min"),
            s_max_m=("s_m", "max"),
            terminal_frac=("terminal_sample", "mean"),
            median_residual_m=("residual_m", "median"),
            route_length_m=("route_length_m", "first"),
            geometry_source=("geometry_source", "first"),
        )
        .merge(representative, on=keys, how="left")
    )
    grouped["duration_min"] = (grouped["last_ts"] - grouped["first_ts"]).dt.total_seconds() / 60.0
    grouped["s_span_m"] = grouped["s_max_m"] - grouped["s_min_m"]
    return grouped


def summarize_bus_hour_exclusions(bus_hours: pd.DataFrame) -> dict[str, int]:
    if bus_hours.empty:
        return {}
    return {
        "bus_hour_rows": int(len(bus_hours)),
        "layover_excluded_bus_hours": int(bus_hours["exclude_layover"].sum()),
        "sparse_midpoint_excluded_bus_hours": int(bus_hours["exclude_sparse_midpoint"].sum()),
        "bad_residual_excluded_bus_hours": int(bus_hours["exclude_bad_residual"].sum()),
        "active_bus_hour_rows": int(bus_hours["active_for_order"].sum()),
    }


def choose_final_lines_by_quality(line_selection: pd.DataFrame, quality: pd.DataFrame) -> list[str]:
    poor_by_line = quality.groupby("line_code")["poor_match_flag"].any().to_dict()
    directions_by_line = quality.groupby("line_code")["direction"].nunique().to_dict()
    final: list[str] = []
    candidates = line_selection[line_selection["candidate"]].sort_values("rank_by_boardings")
    for _, row in candidates.iterrows():
        line = str(row["line_code"])
        if poor_by_line.get(line, True):
            continue
        if directions_by_line.get(line, 0) < 2:
            continue
        final.append(line)
        if len(final) >= TOP_LINE_TARGET:
            break
    if len(final) < TOP_LINE_TARGET:
        for _, row in candidates.iterrows():
            line = str(row["line_code"])
            if line not in final and directions_by_line.get(line, 0) >= 2:
                final.append(line)
            if len(final) >= TOP_LINE_TARGET:
                break
    return final


def compute_hourly_order(bus_hours: pd.DataFrame) -> pd.DataFrame:
    active = bus_hours[bus_hours["active_for_order"]].copy()
    active["cos_phi"] = np.cos(active["phase_mid_rad"])
    active["sin_phi"] = np.sin(active["phase_mid_rad"])
    order = (
        active.groupby(["line_code", "direction", "service_date", "hour"], as_index=False)
        .agg(
            mean_cos=("cos_phi", "mean"),
            mean_sin=("sin_phi", "mean"),
            N=("vehicle_id", "nunique"),
            active_bus_hours=("vehicle_id", "size"),
            median_midpoint_gap_min=("nearest_midpoint_min", "median"),
            median_residual_m=("median_residual_m", "median"),
        )
    )
    order["r1"] = np.sqrt(order["mean_cos"] ** 2 + order["mean_sin"] ** 2)
    return order


def finite_size_null(max_n: int, draws: int = NULL_DRAWS, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for n in range(1, max_n + 1):
        if n == 1:
            values = np.ones(draws)
            splay_r = 1.0
        else:
            phases = rng.uniform(0.0, 2.0 * np.pi, size=(draws, n))
            values = np.abs(np.exp(1j * phases).mean(axis=1))
            splay = np.arange(n) * 2.0 * np.pi / n
            splay_r = float(np.abs(np.exp(1j * splay).mean()))
        rows.append(
            {
                "N": n,
                "draws": draws,
                "r_null_mean": float(values.mean()),
                "r_null_std": float(values.std(ddof=1)),
                "r_null_q025": float(np.quantile(values, 0.025)),
                "r_null_q975": float(np.quantile(values, 0.975)),
                "r_random_asymptotic_0_886_over_sqrtN": float(0.886226925 / math.sqrt(n)),
                "r_splay": splay_r,
            }
        )
    return pd.DataFrame(rows)


def assemble_hourly(
    order: pd.DataFrame,
    demand: pd.DataFrame,
    weather: pd.DataFrame,
    null_table: pd.DataFrame,
) -> pd.DataFrame:
    hourly = order.merge(demand, on=["line_code", "service_date", "hour"], how="left")
    hourly["boardings"] = hourly["boardings"].fillna(0).astype(int)
    hourly["distinct_registered_users"] = hourly["distinct_registered_users"].fillna(0).astype(int)
    hourly["free_or_transfer_boardings"] = hourly["free_or_transfer_boardings"].fillna(0).astype(int)
    hourly["day_category"] = np.where(pd.to_datetime(hourly["service_date"]).dt.dayofweek >= 5, "weekend", "weekday")
    hourly["hour_of_day"] = hourly["hour"].dt.hour
    hourly["is_peak"] = hourly["hour_of_day"].isin([7, 8, 17, 18])
    hourly["period"] = np.where(hourly["is_peak"], "peak", "off_peak")
    hourly["lambda_boardings_per_bus"] = hourly["boardings"] / hourly["N"].replace(0, np.nan)
    hourly = hourly.merge(weather, on="hour", how="left")
    hourly["rain_mm"] = hourly["rain_mm"].fillna(0.0)
    hourly["rain_hour"] = hourly["rain_hour"].fillna(False).astype(bool)
    hourly["rain_day"] = hourly["rain_day"].fillna(False).astype(bool)
    hourly = hourly.merge(null_table, on="N", how="left")
    denom = 1.0 - hourly["r_null_mean"]
    hourly["r_excess"] = np.where(denom > 1e-9, (hourly["r1"] - hourly["r_null_mean"]) / denom, np.nan)
    hourly["z"] = np.where(hourly["r_null_std"] > 1e-12, (hourly["r1"] - hourly["r_null_mean"]) / hourly["r_null_std"], np.nan)
    hourly["analysis_eligible"] = hourly["N"] >= 2
    return hourly.sort_values(["line_code", "direction", "hour"]).reset_index(drop=True)


def bootstrap_diff(
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    group_a: str,
    group_b: str,
    draws: int = 1000,
    seed: int = RNG_SEED,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    a = df.loc[df[group_col] == group_a, value_col].dropna().to_numpy()
    b = df.loc[df[group_col] == group_b, value_col].dropna().to_numpy()
    if len(a) == 0 or len(b) == 0:
        return {"diff": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    diffs = np.empty(draws)
    for i in range(draws):
        diffs[i] = rng.choice(b, size=len(b), replace=True).mean() - rng.choice(a, size=len(a), replace=True).mean()
    return {
        "diff": float(b.mean() - a.mean()),
        "ci_low": float(np.quantile(diffs, 0.025)),
        "ci_high": float(np.quantile(diffs, 0.975)),
    }


def corrected_comparison(hourly: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    eligible = hourly[hourly["analysis_eligible"]].copy()
    by_day = (
        eligible.groupby("day_category", as_index=False)
        .agg(
            rows=("r1", "size"),
            mean_N=("N", "mean"),
            mean_r1=("r1", "mean"),
            mean_r_excess=("r_excess", "mean"),
            mean_z=("z", "mean"),
            mean_lambda=("lambda_boardings_per_bus", "mean"),
        )
        .sort_values("day_category")
    )
    by_period = (
        eligible.groupby(["day_category", "period"], as_index=False)
        .agg(
            rows=("r1", "size"),
            mean_N=("N", "mean"),
            mean_r1=("r1", "mean"),
            mean_r_excess=("r_excess", "mean"),
            mean_z=("z", "mean"),
            mean_lambda=("lambda_boardings_per_bus", "mean"),
        )
        .sort_values(["day_category", "period"])
    )
    comparison = pd.concat([by_day.assign(comparison="weekday_weekend"), by_period.assign(comparison="day_period")])
    raw_diff = bootstrap_diff(eligible, "r1", "day_category", "weekday", "weekend")
    excess_diff = bootstrap_diff(eligible, "r_excess", "day_category", "weekday", "weekend")
    z_diff = bootstrap_diff(eligible, "z", "day_category", "weekday", "weekend")
    verdict = "ambiguous"
    if raw_diff["diff"] > 0 and excess_diff["diff"] < 0 and excess_diff["ci_high"] <= 0:
        verdict = "corrected synchronization is lower on weekends"
    elif raw_diff["diff"] > 0 and (excess_diff["ci_low"] <= 0 <= excess_diff["ci_high"] or abs(excess_diff["diff"]) < 0.02):
        verdict = "weekday/weekend gap disappears after finite-size correction"
    elif raw_diff["diff"] > 0 and excess_diff["diff"] > 0:
        verdict = "weekend remains higher after correction"
    return comparison, {"raw_diff": raw_diff, "r_excess_diff": excess_diff, "z_diff": z_diff, "verdict": verdict}


def prepare_model_frame(hourly: pd.DataFrame) -> pd.DataFrame:
    df = hourly[hourly["analysis_eligible"]].copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["r_excess", "z", "lambda_boardings_per_bus", "N", "rain_hour"])
    df["group"] = df["line_code"].astype(str) + "_d" + df["direction"].astype(str)
    df["weekend"] = (df["day_category"] == "weekend").astype(float)
    df["rain"] = df["rain_hour"].astype(float)
    df["hour_sin"] = np.sin(2.0 * np.pi * df["hour_of_day"] / 24.0)
    df["hour_cos"] = np.cos(2.0 * np.pi * df["hour_of_day"] / 24.0)
    for col in ["lambda_boardings_per_bus", "N"]:
        mean = df[col].mean()
        std = df[col].std(ddof=0)
        df[col + "_std"] = (df[col] - mean) / std if std > 0 else 0.0
        df[col + "_mean_for_standardization"] = mean
        df[col + "_std_for_standardization"] = std
    df["lambda_std_x_rain"] = df["lambda_boardings_per_bus_std"] * df["rain"]
    return df


def random_intercept_ml(y: np.ndarray, X: np.ndarray, groups: np.ndarray, names: list[str]) -> dict[str, Any]:
    group_codes, group_idx = np.unique(groups, return_inverse=True)
    group_members = [np.where(group_idx == g)[0] for g in range(len(group_codes))]
    n, p = X.shape

    def gls_components(log_vars: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, float]:
        sigma2 = float(np.exp(log_vars[0]))
        tau2 = float(np.exp(log_vars[1]))
        xt_v_x = np.zeros((p, p))
        xt_v_y = np.zeros(p)
        y_v_y = 0.0
        logdet = 0.0
        for idx in group_members:
            Xg = X[idx]
            yg = y[idx]
            ng = len(idx)
            sum_x = Xg.sum(axis=0)
            sum_y = yg.sum()
            alpha = tau2 / (sigma2 * (sigma2 + ng * tau2))
            xt_v_x += (Xg.T @ Xg) / sigma2 - alpha * np.outer(sum_x, sum_x)
            xt_v_y += (Xg.T @ yg) / sigma2 - alpha * sum_x * sum_y
            y_v_y += float((yg @ yg) / sigma2 - alpha * sum_y * sum_y)
            logdet += (ng - 1) * math.log(sigma2) + math.log(sigma2 + ng * tau2)
        beta = np.linalg.solve(xt_v_x, xt_v_y)
        quad = float(y_v_y - beta @ xt_v_y)
        ll = -0.5 * (n * math.log(2.0 * math.pi) + logdet + quad)
        return ll, beta, xt_v_x, quad

    y_var = np.var(y) if np.var(y) > 0 else 1.0

    def objective(log_vars: np.ndarray) -> float:
        ll, *_ = gls_components(log_vars)
        return -ll

    result = minimize(
        objective,
        x0=np.log([max(y_var * 0.8, 1e-6), max(y_var * 0.2, 1e-6)]),
        method="Nelder-Mead",
        options={"maxiter": 800, "xatol": 1e-8, "fatol": 1e-8},
    )
    ll, beta, xt_v_x, _ = gls_components(result.x)
    cov = np.linalg.inv(xt_v_x)
    se = np.sqrt(np.diag(cov))
    zstat = beta / se
    pvals = 2.0 * (1.0 - norm.cdf(np.abs(zstat)))
    sigma2, tau2 = np.exp(result.x)
    coef = pd.DataFrame(
        {
            "term": names,
            "estimate": beta,
            "std_error": se,
            "z": zstat,
            "p_value": pvals,
        }
    )
    return {
        "coef": coef,
        "log_likelihood": float(ll),
        "aic": float(2 * (p + 2) - 2 * ll),
        "sigma2": float(sigma2),
        "tau2": float(tau2),
        "n": int(n),
        "groups": int(len(group_codes)),
        "optimizer_success": bool(result.success),
    }


def fit_main_mixed_models(model_df: pd.DataFrame) -> dict[str, Any]:
    terms = [
        "intercept",
        "lambda_boardings_per_bus_std",
        "N_std",
        "hour_sin",
        "hour_cos",
        "weekend",
        "rain",
        "lambda_std_x_rain",
    ]
    X = np.column_stack(
        [
            np.ones(len(model_df)),
            model_df["lambda_boardings_per_bus_std"].to_numpy(float),
            model_df["N_std"].to_numpy(float),
            model_df["hour_sin"].to_numpy(float),
            model_df["hour_cos"].to_numpy(float),
            model_df["weekend"].to_numpy(float),
            model_df["rain"].to_numpy(float),
            model_df["lambda_std_x_rain"].to_numpy(float),
        ]
    )
    groups = model_df["group"].to_numpy(str)
    return {
        "r_excess": random_intercept_ml(model_df["r_excess"].to_numpy(float), X, groups, terms),
        "z": random_intercept_ml(model_df["z"].to_numpy(float), X, groups, terms),
    }


def fixed_effect_design(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    groups = pd.get_dummies(df["group"], prefix="group", drop_first=True, dtype=float)
    controls = pd.DataFrame(
        {
            "intercept": 1.0,
            "N_std": df["N_std"].to_numpy(float),
            "hour_sin": df["hour_sin"].to_numpy(float),
            "hour_cos": df["hour_cos"].to_numpy(float),
            "weekend": df["weekend"].to_numpy(float),
            "rain": df["rain"].to_numpy(float),
        },
        index=df.index,
    )
    design = pd.concat([controls, groups], axis=1)
    return design.to_numpy(float), list(design.columns)


def ols_fit(y: np.ndarray, X: np.ndarray, k_extra: int = 0) -> dict[str, Any]:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n = len(y)
    rss = float(resid @ resid)
    sigma2 = max(rss / n, 1e-12)
    k = X.shape[1] + k_extra
    ll = -0.5 * n * (math.log(2.0 * math.pi * sigma2) + 1.0)
    return {"beta": beta, "rss": rss, "log_likelihood": ll, "aic": float(2 * k - 2 * ll), "sigma2": sigma2}


def fit_transition_models(model_df: pd.DataFrame, seed: int = RNG_SEED) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    y = model_df["r_excess"].to_numpy(float)
    base_X, base_terms = fixed_effect_design(model_df)
    lam = model_df["lambda_boardings_per_bus"].to_numpy(float)
    lam_std = model_df["lambda_boardings_per_bus_std"].to_numpy(float)
    quantiles = np.unique(np.quantile(lam, np.linspace(0.05, 0.95, 61)))
    iqr = max(np.quantile(lam, 0.75) - np.quantile(lam, 0.25), 1.0)

    linear_X = np.column_stack([base_X, lam_std])
    linear = ols_fit(y, linear_X)
    linear.update({"model": "linear", "lambda_c": np.nan, "scale": np.nan})

    best_piece = None
    for c in quantiles:
        hinge = np.maximum(0.0, lam - c)
        hinge_std = (hinge - hinge.mean()) / hinge.std(ddof=0) if hinge.std(ddof=0) > 0 else hinge
        X = np.column_stack([base_X, lam_std, hinge_std])
        fit = ols_fit(y, X, k_extra=1)
        fit.update({"model": "piecewise", "lambda_c": float(c), "scale": np.nan})
        if best_piece is None or fit["aic"] < best_piece["aic"]:
            best_piece = fit

    best_logistic = None
    scales = np.unique(np.maximum(np.array([iqr / 20, iqr / 12, iqr / 8, iqr / 5, iqr / 3, iqr / 2]), 0.5))
    c_grid = np.unique(np.quantile(lam, np.linspace(0.10, 0.90, 41)))
    for c in c_grid:
        for scale in scales:
            sigmoid = 1.0 / (1.0 + np.exp(-(lam - c) / scale))
            sigmoid_std = (sigmoid - sigmoid.mean()) / sigmoid.std(ddof=0) if sigmoid.std(ddof=0) > 0 else sigmoid
            X = np.column_stack([base_X, sigmoid_std])
            fit = ols_fit(y, X, k_extra=2)
            fit.update({"model": "sigmoid", "lambda_c": float(c), "scale": float(scale)})
            if best_logistic is None or fit["aic"] < best_logistic["aic"]:
                best_logistic = fit

    fits = [linear, best_piece, best_logistic]
    table = pd.DataFrame(
        [
            {
                "model": f["model"],
                "aic": f["aic"],
                "delta_aic": f["aic"] - min(x["aic"] for x in fits),
                "rss": f["rss"],
                "lambda_c": f["lambda_c"],
                "scale": f["scale"],
            }
            for f in fits
        ]
    ).sort_values("aic")
    boot = bootstrap_piecewise_change_point(model_df, quantiles=quantiles, draws=BOOTSTRAP_DRAWS, seed=seed)
    details = {
        "best_model": str(table.iloc[0]["model"]),
        "linear": linear,
        "piecewise": best_piece,
        "sigmoid": best_logistic,
        "base_terms": base_terms,
    }
    return table, details, boot


def bootstrap_piecewise_change_point(
    model_df: pd.DataFrame,
    quantiles: np.ndarray,
    draws: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    n = len(model_df)
    for i in range(draws):
        sample_idx = rng.integers(0, n, size=n)
        sample = model_df.iloc[sample_idx].reset_index(drop=True)
        y = sample["r_excess"].to_numpy(float)
        base_X, _ = fixed_effect_design(sample)
        lam = sample["lambda_boardings_per_bus"].to_numpy(float)
        lam_std = sample["lambda_boardings_per_bus_std"].to_numpy(float)
        best_aic = np.inf
        best_c = np.nan
        for c in quantiles:
            hinge = np.maximum(0.0, lam - c)
            hstd = hinge.std(ddof=0)
            hinge_std = (hinge - hinge.mean()) / hstd if hstd > 0 else hinge
            X = np.column_stack([base_X, lam_std, hinge_std])
            fit = ols_fit(y, X, k_extra=1)
            if fit["aic"] < best_aic:
                best_aic = fit["aic"]
                best_c = c
        rows.append({"draw": i, "lambda_c": float(best_c), "aic": float(best_aic)})
    return pd.DataFrame(rows)


def weather_matched_comparison(model_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    grouped = (
        model_df.groupby(["group", "hour_of_day", "rain"], as_index=False)
        .agg(mean_r_excess=("r_excess", "mean"), mean_z=("z", "mean"), rows=("r_excess", "size"))
    )
    pivot = grouped.pivot_table(index=["group", "hour_of_day"], columns="rain", values="mean_r_excess")
    if 0.0 in pivot.columns and 1.0 in pivot.columns:
        paired = (pivot[1.0] - pivot[0.0]).dropna().reset_index(name="rain_minus_dry_r_excess")
    else:
        paired = pd.DataFrame(columns=["group", "hour_of_day", "rain_minus_dry_r_excess"])
    values = paired["rain_minus_dry_r_excess"].to_numpy(float)
    if len(values):
        rng = np.random.default_rng(RNG_SEED)
        boots = np.array([rng.choice(values, size=len(values), replace=True).mean() for _ in range(1000)])
        summary = {
            "paired_cells": int(len(values)),
            "mean_rain_minus_dry_r_excess": float(values.mean()),
            "ci_low": float(np.quantile(boots, 0.025)),
            "ci_high": float(np.quantile(boots, 0.975)),
        }
    else:
        summary = {"paired_cells": 0, "mean_rain_minus_dry_r_excess": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    return paired, summary


def global_lonlat_to_xy(lon: np.ndarray, lat: np.ndarray, lon0: float = -43.08, lat0: float = -22.88) -> np.ndarray:
    radius = 6_371_000.0
    x = radius * np.deg2rad(lon - lon0) * math.cos(math.radians(lat0))
    y = radius * np.deg2rad(lat - lat0)
    return np.column_stack([x, y])


def sample_route_xy(geom: RouteGeometry, step_m: float = 50.0) -> np.ndarray:
    coords = np.array(geom.coords, dtype=float)
    xy = global_lonlat_to_xy(coords[:, 0], coords[:, 1])
    points = []
    for a, b in zip(xy[:-1], xy[1:]):
        vec = b - a
        length = float(np.linalg.norm(vec))
        if length <= 0:
            continue
        n = max(int(math.ceil(length / step_m)), 1)
        for t in np.linspace(0.0, 1.0, n, endpoint=False):
            points.append(a + t * vec)
    points.append(xy[-1])
    return np.array(points)


def route_samples_for_lines(
    selected_lines: list[str],
    static_geoms: dict[tuple[str, int | None], RouteGeometry],
    gtfs_geoms: dict[tuple[str, int], RouteGeometry],
) -> dict[str, np.ndarray]:
    samples = {}
    for line in selected_lines:
        chunks = []
        for direction in [0, 1]:
            geom = select_geometry(line, direction, static_geoms, gtfs_geoms)
            if geom is not None:
                chunks.append(sample_route_xy(geom, step_m=50.0))
        if chunks:
            samples[line] = np.vstack(chunks)
    return samples


def load_stop_xy(root: Path) -> np.ndarray:
    data = json.loads((root / "auxiliar_data" / "stops.json").read_text())
    coords = []
    for feature in data.get("features", []):
        if feature.get("geometry", {}).get("type") == "Point":
            lon, lat = feature["geometry"]["coordinates"][:2]
            coords.append((float(lon), float(lat)))
    arr = np.array(coords, dtype=float)
    return global_lonlat_to_xy(arr[:, 0], arr[:, 1]) if len(arr) else np.empty((0, 2))


def corridor_and_correlation(
    root: Path,
    selected_lines: list[str],
    hourly: pd.DataFrame,
    static_geoms: dict[tuple[str, int | None], RouteGeometry],
    gtfs_geoms: dict[tuple[str, int], RouteGeometry],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    samples = route_samples_for_lines(selected_lines, static_geoms, gtfs_geoms)
    stop_xy = load_stop_xy(root)
    stop_sets: dict[str, set[int]] = {}
    for line, xy in samples.items():
        tree = cKDTree(xy)
        dist, _ = tree.query(stop_xy, k=1)
        stop_sets[line] = set(np.where(dist <= SHARED_BUFFER_M)[0].tolist())

    line_hour = (
        hourly[hourly["analysis_eligible"]]
        .groupby(["line_code", "hour"], as_index=False)
        .agg(r_excess=("r_excess", "mean"), z=("z", "mean"), N=("N", "sum"))
    )
    pivot = line_hour.pivot(index="hour", columns="line_code", values="r_excess").sort_index()
    rows = []
    lines = [line for line in selected_lines if line in samples]
    rng = np.random.default_rng(RNG_SEED)
    for i, a in enumerate(lines):
        tree_a = cKDTree(samples[a])
        for b in lines[i + 1 :]:
            tree_b = cKDTree(samples[b])
            dist_ab, _ = tree_b.query(samples[a], k=1)
            dist_ba, _ = tree_a.query(samples[b], k=1)
            overlap_m = 0.5 * (float((dist_ab <= SHARED_BUFFER_M).sum()) + float((dist_ba <= SHARED_BUFFER_M).sum())) * 50.0
            shared_stops = len(stop_sets.get(a, set()) & stop_sets.get(b, set()))
            substantial = (overlap_m >= SHARED_TRUNK_MIN_M) or (shared_stops >= SHARED_STOPS_MIN)
            pair = pivot[[a, b]].dropna()
            if len(pair) >= 20 and pair[a].std(ddof=0) > 0 and pair[b].std(ddof=0) > 0:
                corr = float(pair[a].corr(pair[b]))
                null = np.empty(500)
                bvals = pair[b].to_numpy(float)
                avals = pair[a].to_numpy(float)
                for j in range(len(null)):
                    null[j] = np.corrcoef(avals, rng.permutation(bvals))[0, 1]
                p_perm = float((np.abs(null) >= abs(corr)).mean())
            else:
                corr = np.nan
                p_perm = np.nan
            rows.append(
                {
                    "line_a": a,
                    "line_b": b,
                    "overlap_m_50m_buffer_approx": overlap_m,
                    "shared_stops_50m": shared_stops,
                    "shared_corridor": substantial,
                    "matched_hours": int(len(pair)),
                    "r_excess_corr": corr,
                    "pair_permutation_p": p_perm,
                }
            )
    pairs = pd.DataFrame(rows)
    corr_matrix = pivot.corr(min_periods=20).reindex(index=selected_lines, columns=selected_lines)
    observed = pairs.dropna(subset=["r_excess_corr"]).copy()
    shared = observed[observed["shared_corridor"]]["r_excess_corr"].to_numpy(float)
    non = observed[~observed["shared_corridor"]]["r_excess_corr"].to_numpy(float)
    if len(shared) and len(non):
        diff = float(shared.mean() - non.mean())
        labels = observed["shared_corridor"].to_numpy(bool)
        vals = observed["r_excess_corr"].to_numpy(float)
        null = np.empty(CORRIDOR_PERMUTATIONS)
        for i in range(CORRIDOR_PERMUTATIONS):
            perm = rng.permutation(labels)
            null[i] = vals[perm].mean() - vals[~perm].mean()
        p = float((np.abs(null) >= abs(diff)).mean())
    else:
        diff = np.nan
        p = np.nan
    summary = {
        "shared_pairs": int(pairs["shared_corridor"].sum()) if not pairs.empty else 0,
        "nonshared_pairs": int((~pairs["shared_corridor"]).sum()) if not pairs.empty else 0,
        "mean_corr_shared": float(np.nanmean(shared)) if len(shared) else np.nan,
        "mean_corr_nonshared": float(np.nanmean(non)) if len(non) else np.nan,
        "shared_minus_nonshared_corr": diff,
        "label_permutation_p": p,
    }
    return pairs, corr_matrix, summary


def mixed_model_summary_to_json(result: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for key, value in result.items():
        out[key] = {
            "coef": value["coef"].to_dict(orient="records"),
            "log_likelihood": value["log_likelihood"],
            "aic": value["aic"],
            "sigma2": value["sigma2"],
            "tau2": value["tau2"],
            "n": value["n"],
            "groups": value["groups"],
            "optimizer_success": value["optimizer_success"],
        }
    return out


def save_phase2_figures(
    output_dir: Path,
    hourly: pd.DataFrame,
    null_table: pd.DataFrame,
    comparison: pd.DataFrame,
    model_df: pd.DataFrame,
    transition_table: pd.DataFrame,
    transition_details: dict[str, Any],
    weather_pairs: pd.DataFrame,
    corridor_pairs: pd.DataFrame,
    corr_matrix: pd.DataFrame,
    selected_lines: list[str],
    static_geoms: dict[tuple[str, int | None], RouteGeometry],
    gtfs_geoms: dict[tuple[str, int], RouteGeometry],
) -> None:
    eligible = hourly[hourly["analysis_eligible"]].copy()
    eligible[["line_code", "direction", "hour", "N", "r1", "r_null_mean", "r_null_q025", "r_null_q975", "r_excess", "z"]].to_csv(
        output_dir / "p1_finite_size_validation_data.csv", index=False
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    rng = np.random.default_rng(RNG_SEED)
    plot_data = eligible.sample(min(12000, len(eligible)), random_state=RNG_SEED)
    ax.scatter(plot_data["N"], plot_data["r1"], s=8, alpha=0.20, color="#2b6cb0", label="observed hourly r1")
    ax.plot(null_table["N"], null_table["r_null_mean"], color="black", linewidth=2, label="random-phase null mean")
    ax.fill_between(
        null_table["N"].to_numpy(float),
        null_table["r_null_q025"].to_numpy(float),
        null_table["r_null_q975"].to_numpy(float),
        color="gray",
        alpha=0.25,
        label="random-phase 95% band",
    )
    ax.plot(
        null_table["N"],
        null_table["r_random_asymptotic_0_886_over_sqrtN"],
        color="#d62728",
        linestyle="--",
        label="0.886/sqrt(N)",
    )
    ax.set_xlabel("Active buses N")
    ax.set_ylabel("Raw order parameter r1")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "p1_finite_size_validation.png", dpi=200)
    plt.close(fig)

    comparison.to_csv(output_dir / "p2_weekday_weekend_peak_comparison.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, value, title in zip(axes, ["mean_r_excess", "mean_z"], ["r_excess", "z"]):
        data = comparison[comparison["comparison"] == "day_period"].copy()
        labels = data["day_category"] + "\n" + data["period"]
        ax.bar(labels, data[value], color=["#4c78a8" if d == "weekday" else "#f58518" for d in data["day_category"]])
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.set_ylabel(value)
        ax.tick_params(axis="x", rotation=0)
        ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "p2_corrected_weekday_weekend_peak.png", dpi=200)
    plt.close(fig)

    binned = model_df.copy()
    binned["lambda_bin"] = pd.qcut(binned["lambda_boardings_per_bus"], q=30, duplicates="drop")
    p3_data = (
        binned.groupby("lambda_bin", observed=True, as_index=False)
        .agg(
            lambda_mid=("lambda_boardings_per_bus", "median"),
            mean_r_excess=("r_excess", "mean"),
            mean_z=("z", "mean"),
            rows=("r_excess", "size"),
        )
        .sort_values("lambda_mid")
    )
    p3_data.to_csv(output_dir / "p3_transition_plot_data.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(
        p3_data["lambda_mid"],
        p3_data["mean_r_excess"],
        s=np.clip(p3_data["rows"], 10, 200),
        alpha=0.75,
        color="#2b6cb0",
        label="hourly bins",
    )
    best = transition_table.iloc[0]["model"]
    xgrid = np.linspace(model_df["lambda_boardings_per_bus"].quantile(0.01), model_df["lambda_boardings_per_bus"].quantile(0.99), 250)
    ycurve = transition_curve_reference(model_df, transition_details, str(best), xgrid)
    ax.plot(xgrid, ycurve, color="#d62728", linewidth=2.2, label=f"best fit: {best}")
    if str(best) == "piecewise" and np.isfinite(transition_details["piecewise"]["lambda_c"]):
        ax.axvline(transition_details["piecewise"]["lambda_c"], color="#d62728", linestyle="--", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Load per bus lambda = boardings / N")
    ax.set_ylabel("Corrected synchronization r_excess")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "p3_transition_lambda_sync.png", dpi=200)
    plt.close(fig)

    weather_plot = model_df.copy()
    weather_plot["rain_label"] = np.where(weather_plot["rain"] > 0, "rain", "dry")
    weather_plot["lambda_bin"] = pd.qcut(weather_plot["lambda_boardings_per_bus"], q=16, duplicates="drop")
    p4_data = (
        weather_plot.groupby(["rain_label", "lambda_bin"], observed=True, as_index=False)
        .agg(lambda_mid=("lambda_boardings_per_bus", "median"), mean_r_excess=("r_excess", "mean"), rows=("r_excess", "size"))
    )
    p4_data.to_csv(output_dir / "p4_weather_plot_data.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 6))
    for label, color in [("dry", "#4c78a8"), ("rain", "#f58518")]:
        group = p4_data[p4_data["rain_label"] == label].sort_values("lambda_mid")
        ax.plot(group["lambda_mid"], group["mean_r_excess"], marker="o", color=color, label=label)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Load per bus lambda")
    ax.set_ylabel("Corrected synchronization r_excess")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "p4_weather_modifier.png", dpi=200)
    plt.close(fig)

    samples = route_samples_for_lines(selected_lines, static_geoms, gtfs_geoms)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    ax = axes[0]
    for line, xy in samples.items():
        ax.plot(xy[:, 0], xy[:, 1], color="lightgray", linewidth=0.7, alpha=0.65)
    graph = nx.Graph()
    for line in selected_lines:
        graph.add_node(line)
    shared = corridor_pairs[corridor_pairs["shared_corridor"]].copy()
    for _, row in shared.iterrows():
        a, b = row["line_a"], row["line_b"]
        if a in samples and b in samples:
            graph.add_edge(a, b, weight=row["overlap_m_50m_buffer_approx"])
            ax.plot(samples[a][:, 0], samples[a][:, 1], linewidth=1.2, alpha=0.35, color="#d62728")
            ax.plot(samples[b][:, 0], samples[b][:, 1], linewidth=1.2, alpha=0.35, color="#d62728")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Approximate shared trunks")
    ax.set_xticks([])
    ax.set_yticks([])

    mat = corr_matrix.to_numpy(float)
    im = axes[1].imshow(mat, vmin=-1, vmax=1, cmap="coolwarm")
    axes[1].set_xticks(np.arange(len(corr_matrix.columns)))
    axes[1].set_xticklabels(corr_matrix.columns, rotation=90, fontsize=8)
    axes[1].set_yticks(np.arange(len(corr_matrix.index)))
    axes[1].set_yticklabels(corr_matrix.index, fontsize=8)
    axes[1].set_title("Hourly r_excess correlation")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_dir / "p5_corridor_coupling.png", dpi=200)
    plt.close(fig)


def transition_curve_reference(model_df: pd.DataFrame, details: dict[str, Any], model: str, xgrid: np.ndarray) -> np.ndarray:
    reference = model_df.iloc[[0]].copy()
    for col in ["N_std", "hour_sin", "hour_cos", "weekend", "rain"]:
        reference[col] = 0.0
    base_X, _ = fixed_effect_design(pd.concat([reference] * len(xgrid), ignore_index=True))
    # fixed_effect_design with one group has no group columns; use only control coefficients.
    fit = details[model]
    base_cols = base_X.shape[1]
    if model == "linear":
        lam_mean = model_df["lambda_boardings_per_bus_mean_for_standardization"].iloc[0]
        lam_std = model_df["lambda_boardings_per_bus_std_for_standardization"].iloc[0]
        shape = ((xgrid - lam_mean) / lam_std)[:, None]
        beta = fit["beta"][: base_cols + 1]
        X = np.column_stack([base_X, shape])
        return X @ beta
    if model == "piecewise":
        lam_mean = model_df["lambda_boardings_per_bus_mean_for_standardization"].iloc[0]
        lam_std = model_df["lambda_boardings_per_bus_std_for_standardization"].iloc[0]
        shape1 = ((xgrid - lam_mean) / lam_std)
        hinge = np.maximum(0.0, xgrid - fit["lambda_c"])
        hstd = hinge.std(ddof=0)
        shape2 = (hinge - hinge.mean()) / hstd if hstd > 0 else hinge
        beta = fit["beta"][: base_cols + 2]
        X = np.column_stack([base_X, shape1, shape2])
        return X @ beta
    sigmoid = 1.0 / (1.0 + np.exp(-(xgrid - fit["lambda_c"]) / fit["scale"]))
    sigmoid = (sigmoid - sigmoid.mean()) / sigmoid.std(ddof=0)
    beta = fit["beta"][: base_cols + 1]
    X = np.column_stack([base_X, sigmoid])
    return X @ beta


def write_report(
    root: Path,
    output_dir: Path,
    selected_lines: list[str],
    line_selection: pd.DataFrame,
    quality: pd.DataFrame,
    counts: dict[str, Any],
    hourly: pd.DataFrame,
    null_table: pd.DataFrame,
    comparison: pd.DataFrame,
    comparison_stats: dict[str, Any],
    mixed: dict[str, Any],
    transition_table: pd.DataFrame,
    boot_cp: pd.DataFrame,
    weather_summary: dict[str, float],
    corridor_summary: dict[str, float],
) -> None:
    eligible = hourly[hourly["analysis_eligible"]].copy()
    rain_hours = hourly.loc[hourly["rain_hour"], "hour"].dt.date.astype(str).value_counts().sort_index()
    selected_md = dataframe_to_markdown(
        line_selection[line_selection["selected"]][["rank_by_boardings", "line_code", "ticket_codes", "boardings"]]
    )
    quality_md = dataframe_to_markdown(
        quality[["line_code", "direction", "rows", "geometry_source", "residual_m_median", "residual_m_p95", "poor_match_flag"]]
    )
    comparison_md = dataframe_to_markdown(comparison.round(3))
    transition_md = dataframe_to_markdown(transition_table.round(3))
    cp_ci = {
        "median": float(boot_cp["lambda_c"].median()),
        "ci_low": float(boot_cp["lambda_c"].quantile(0.025)),
        "ci_high": float(boot_cp["lambda_c"].quantile(0.975)),
    }
    mixed_coef = mixed["r_excess"]["coef"]
    lambda_row = mixed_coef[mixed_coef["term"] == "lambda_boardings_per_bus_std"].iloc[0].to_dict()
    rain_interaction = mixed_coef[mixed_coef["term"] == "lambda_std_x_rain"].iloc[0].to_dict()
    best_model = str(transition_table.iloc[0]["model"])
    best_delta = float(transition_table.iloc[1]["delta_aic"]) if len(transition_table) > 1 else np.nan
    best_lambda_c = float(transition_table.iloc[0]["lambda_c"]) if pd.notna(transition_table.iloc[0]["lambda_c"]) else np.nan
    cp_width = cp_ci["ci_high"] - cp_ci["ci_low"]
    cp_far_from_best = np.isfinite(best_lambda_c) and abs(cp_ci["median"] - best_lambda_c) > max(10.0, 0.2 * best_lambda_c)
    cp_degenerate = cp_width < 1.0
    if best_model == "piecewise" and best_delta >= 4 and not cp_far_from_best and not cp_degenerate:
        transition_claim = (
            f"The piecewise model is selected by AIC; candidate critical load "
            f"lambda_c={transition_table.iloc[0]['lambda_c']:.2f} boardings/bus "
            f"(bootstrap median {cp_ci['median']:.2f}, 95% CI {cp_ci['ci_low']:.2f}-{cp_ci['ci_high']:.2f})."
        )
    elif best_model == "piecewise" and best_delta >= 4:
        transition_claim = (
            f"The piecewise model has the best AIC with candidate lambda_c={best_lambda_c:.2f} "
            f"boardings/bus, but bootstrap breakpoints are unstable "
            f"(median {cp_ci['median']:.2f}, 95% CI {cp_ci['ci_low']:.2f}-{cp_ci['ci_high']:.2f}); "
            "this supports a nonlinear association, not a robust critical threshold claim."
        )
    else:
        transition_claim = (
            f"The best AIC model is `{best_model}`, and the piecewise threshold is not decisively selected; "
            "the data do not support a sharp critical transition in this Phase 2 fit."
        )
    report = f"""# Phase 2 Finite-Size-Corrected Synchronization Analysis

## Headline Verdict

Phase 2 resolves the Phase 1 finite-size concern by comparing every hourly raw order parameter
to a same-`N` random-phase null with {NULL_DRAWS:,} simulations per `N`. The weekday/weekend
result after correction is: **{comparison_stats['verdict']}**.

Raw weekend-minus-weekday `r1` difference is {comparison_stats['raw_diff']['diff']:.3f}
(95% bootstrap CI {comparison_stats['raw_diff']['ci_low']:.3f} to
{comparison_stats['raw_diff']['ci_high']:.3f}). Corrected weekend-minus-weekday
`r_excess` difference is {comparison_stats['r_excess_diff']['diff']:.3f}
(95% CI {comparison_stats['r_excess_diff']['ci_low']:.3f} to
{comparison_stats['r_excess_diff']['ci_high']:.3f}); `z` difference is
{comparison_stats['z_diff']['diff']:.3f} (95% CI {comparison_stats['z_diff']['ci_low']:.3f}
to {comparison_stats['z_diff']['ci_high']:.3f}). N is reported alongside every corrected
summary in `outputs/phase2/p2_weekday_weekend_peak_comparison.csv`.

## Scope and Exclusions

This run covers the mobility/ticketing overlap days with mobility files present:
{', '.join(counts['days'])}. The analysis unit is `(line, direction, hour)`.
Demand remains line-hour level and is repeated across directions because Phase 1 found 0%
vehicle-id overlap.

Final quality-approved top ridership lines with usable direction-0 and direction-1 geometry:

{selected_md}

Bus-hour active oscillator exclusions are logged in
`outputs/phase2/filtering_and_exclusion_log.json`. Terminal layover is defined as at least
{LAYOVER_MIN_DURATION_MIN:.0f} minutes in a bus-hour, at least {LAYOVER_TERMINAL_FRAC:.0%}
of samples within 5% of either route endpoint, and arc-length span <= {LAYOVER_MAX_SPAN_M:.0f} m.
This removed {counts['bus_hour_exclusion_summary']['layover_excluded_bus_hours']:,} bus-hours.
Sparse midpoint sampling removed {counts['bus_hour_exclusion_summary']['sparse_midpoint_excluded_bus_hours']:,}
bus-hours, and high median residual removed
{counts['bus_hour_exclusion_summary']['bad_residual_excluded_bus_hours']:,}. Active bus-hours
retained: {counts['bus_hour_exclusion_summary']['active_bus_hour_rows']:,}.

Map-matching quality for selected lines:

{quality_md}

## Finite-Size Correction

For each `N`, the null distribution is generated from uniform random phases. The perfect splay
state is also recorded (`r_splay`, zero up to numerical precision for N>=2). We use:

`r_excess = (r1 - E_null[r|N]) / (1 - E_null[r|N])`

and

`z = (r1 - E_null[r|N]) / sd_null[r|N]`.

Rows with `N=1` are retained in the file but excluded from corrected modeling because the
random null has `r=1` and zero variance. The final hourly table has {len(hourly):,} rows,
of which {len(eligible):,} have `N>=2`.

Corrected weekday/weekend and peak/off-peak table:

{comparison_md}

## Demand-Coupling and Transition Tests

Primary load is `lambda = boardings / N`. The main association model is a Gaussian random-
intercept model with random intercepts for line-direction groups and fixed effects for
standardized lambda, N, hour-of-day sine/cosine, weekend, rain, and lambda x rain. It is
implemented directly with `numpy`/`scipy` maximum likelihood because `statsmodels` is not
installed in this environment.

For `r_excess`, the standardized-lambda coefficient is {lambda_row['estimate']:.3f}
(SE {lambda_row['std_error']:.3f}, p={lambda_row['p_value']:.3g}). The lambda x rain
interaction is {rain_interaction['estimate']:.3f} (SE {rain_interaction['std_error']:.3f},
p={rain_interaction['p_value']:.3g}).

Transition-shape comparison, using line-direction fixed effects plus the same controls:

{transition_md}

{transition_claim}

These are observational associations, not causal estimates.

## Weather Modifier

Weather is converted from the `Timestamp (UTC)` column to local BRT hours and merged by hour.
Rain hours in the mobility window occur on: {', '.join(rain_hours.index.tolist()) if len(rain_hours) else 'none'}.
The matched dry-hour comparison pairs line-direction/hour-of-day cells observed under both rain
and dry conditions. Rain-minus-dry mean `r_excess` is
{weather_summary['mean_rain_minus_dry_r_excess']:.3f} across
{weather_summary['paired_cells']} paired cells (95% CI {weather_summary['ci_low']:.3f} to
{weather_summary['ci_high']:.3f}). This is the cleanest quasi-experimental check, but it still
requires the assumption that rain affects bunching mainly through dwell/service variability
conditional on observed load.

## Corridor Coupling

Shared trunks are estimated without GIS dependencies by sampling each route every 50 m and
counting route-sample overlap within a 50 m buffer; shared stops are official stops within
50 m of both sampled routes. A pair is flagged as sharing a corridor if overlap is at least
{SHARED_TRUNK_MIN_M:.0f} m or shared stops >= {SHARED_STOPS_MIN}.

Shared-corridor pairs: {corridor_summary['shared_pairs']}; non-sharing pairs:
{corridor_summary['nonshared_pairs']}. Mean hourly `r_excess` correlation is
{corridor_summary['mean_corr_shared']:.3f} for shared pairs vs
{corridor_summary['mean_corr_nonshared']:.3f} for non-shared pairs; label-permutation p-value
is {corridor_summary['label_permutation_p']:.3f}. This is descriptive evidence for or against
shared-medium coupling, not a full coupled-oscillator network model.

## Decisions for Phase 3 and Abstract-Ready Claims

- Supported: "Raw Kuramoto order parameters in bus data are strongly confounded by active
  fleet size; same-`N` null correction is necessary before interpreting bunching."
- Supported: "{comparison_stats['verdict']}."
- Supported with observational caveat: "Corrected synchronization has the reported association
  with load per bus after line-direction random intercepts and time/weather controls."
- Not yet causal: "Passenger demand causes synchronization." Weather provides a useful
  exogenous modifier but does not by itself prove the demand mechanism.
- Transition claim: "{transition_claim}"
- Exploratory only: "Shared corridors show the reported cross-line synchronization correlation;
  Phase 3 should model corridor-level coupling explicitly."
"""
    (root / "phase2_report.md").write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("."), help="Path to data root")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/phase2"), help="Phase 2 output directory")
    args = parser.parse_args()
    root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    static_geoms = load_static_geometries(root)
    gtfs_geoms = load_gtfs_geometries(root)
    days = service_days(root)

    candidate_lines, line_selection = select_top_lines(root, days, static_geoms, gtfs_geoms)

    candidate_demand = aggregate_demand(root, days, candidate_lines)
    candidate_demand.to_csv(output_dir / "candidate_line_hour_demand.csv", index=False)

    candidate_bus_hours, candidate_quality, missing_geometry, counts = process_mobility_hourly(
        root, days, candidate_lines, static_geoms, gtfs_geoms
    )
    candidate_bus_hours.to_csv(output_dir / "candidate_bus_hour_phase_candidates.csv", index=False)
    candidate_quality.to_csv(output_dir / "candidate_map_matching_quality_phase2.csv", index=False)
    missing_geometry.to_csv(output_dir / "missing_geometry_phase2.csv", index=False)

    selected_lines = choose_final_lines_by_quality(line_selection, candidate_quality)
    line_selection["selected"] = line_selection["line_code"].isin(selected_lines)
    line_selection.to_csv(output_dir / "line_selection.csv", index=False)

    bus_hours = candidate_bus_hours[candidate_bus_hours["line_code"].isin(selected_lines)].copy()
    quality = candidate_quality[candidate_quality["line_code"].isin(selected_lines)].copy()
    demand = candidate_demand[candidate_demand["line_code"].isin(selected_lines)].copy()
    bus_hours.to_csv(output_dir / "bus_hour_phase_candidates.csv", index=False)
    quality.to_csv(output_dir / "map_matching_quality_phase2.csv", index=False)
    demand.to_csv(output_dir / "line_hour_demand.csv", index=False)

    counts["candidate_lines"] = candidate_lines
    counts["final_selected_lines"] = selected_lines
    poor_by_line = candidate_quality.groupby("line_code")["poor_match_flag"].any().to_dict()
    counts["quality_excluded_candidate_lines"] = [
        line for line in candidate_lines if line not in selected_lines and poor_by_line.get(line, False)
    ]
    counts["candidate_lines_not_in_final_selection"] = [
        line for line in candidate_lines if line not in selected_lines
    ]
    counts["bus_hour_exclusion_summary"] = summarize_bus_hour_exclusions(bus_hours)

    order = compute_hourly_order(bus_hours)
    order.to_csv(output_dir / "hourly_order_raw.csv", index=False)
    null_table = finite_size_null(int(order["N"].max()), draws=NULL_DRAWS)
    null_table.to_csv(output_dir / "finite_size_null_by_N.csv", index=False)

    weather = parse_weather(root)
    weather.to_csv(output_dir / "weather_hourly_local_brt.csv", index=False)
    hourly = assemble_hourly(order, demand, weather, null_table)
    hourly.to_csv(output_dir / "phase2_hourly_observables.csv", index=False)

    comparison, comparison_stats = corrected_comparison(hourly)
    model_df = prepare_model_frame(hourly)
    model_df.to_csv(output_dir / "model_frame.csv", index=False)
    mixed = fit_main_mixed_models(model_df)
    for response, result in mixed.items():
        result["coef"].to_csv(output_dir / f"mixed_model_{response}_coefficients.csv", index=False)
    (output_dir / "mixed_model_summary.json").write_text(json.dumps(mixed_model_summary_to_json(mixed), indent=2))

    transition_table, transition_details, boot_cp = fit_transition_models(model_df)
    transition_table.to_csv(output_dir / "transition_model_comparison.csv", index=False)
    boot_cp.to_csv(output_dir / "piecewise_change_point_bootstrap.csv", index=False)

    weather_pairs, weather_summary = weather_matched_comparison(model_df)
    weather_pairs.to_csv(output_dir / "weather_matched_dry_rain_pairs.csv", index=False)
    (output_dir / "weather_summary.json").write_text(json.dumps(weather_summary, indent=2))

    corridor_pairs, corr_matrix, corridor_summary = corridor_and_correlation(
        root, selected_lines, hourly, static_geoms, gtfs_geoms
    )
    corridor_pairs.to_csv(output_dir / "corridor_pair_overlap_and_correlation.csv", index=False)
    corr_matrix.to_csv(output_dir / "corridor_hourly_correlation_matrix.csv")
    (output_dir / "corridor_summary.json").write_text(json.dumps(corridor_summary, indent=2))

    counts["selected_line_count"] = len(selected_lines)
    counts["hourly_rows"] = int(len(hourly))
    counts["model_rows"] = int(len(model_df))
    (output_dir / "filtering_and_exclusion_log.json").write_text(json.dumps(counts, indent=2, default=str))

    save_phase2_figures(
        output_dir,
        hourly,
        null_table,
        comparison,
        model_df,
        transition_table,
        transition_details,
        weather_pairs,
        corridor_pairs,
        corr_matrix,
        selected_lines,
        static_geoms,
        gtfs_geoms,
    )

    write_report(
        root,
        output_dir,
        selected_lines,
        line_selection,
        quality,
        counts,
        hourly,
        null_table,
        comparison,
        comparison_stats,
        mixed,
        transition_table,
        boot_cp,
        weather_summary,
        corridor_summary,
    )

    summary = {
        "output_dir": str(output_dir),
        "selected_lines": selected_lines,
        "hourly_rows": int(len(hourly)),
        "model_rows": int(len(model_df)),
        "layover_bus_hours_removed": counts["bus_hour_exclusion_summary"]["layover_excluded_bus_hours"],
        "weekday_weekend_verdict": comparison_stats["verdict"],
        "best_transition_model": str(transition_table.iloc[0]["model"]),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
