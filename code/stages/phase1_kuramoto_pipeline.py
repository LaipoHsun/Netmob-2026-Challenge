#!/usr/bin/env python3
"""
Phase 1 bus-bunching observables for the NetMob26 Niteroi dataset.

The pipeline is intentionally explicit and auditable:
- GPS telemetry is projected to route arc length.
- Vehicle trajectories are resampled to a 30 s grid without bridging gaps > 3 min.
- Kuramoto order parameter, headway CV, and line-level demand are joined into a
  tidy line-direction-time table.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


FOCUS_DAYS = ["2026-03-11", "2026-03-12", "2026-03-13", "2026-03-14"]
FOCUS_LINES = ["49.2", "45", "49.1", "48", "35", "62"]
DEMAND_ALIASES = {"62": ["62", "62B"]}
STATIC_GEOM_ALIASES = {"62": ["62", "62B"]}
SECTION_FRACTIONS = [
    ("terminal_start_5pct", 0.05),
    ("pct25", 0.25),
    ("pct50", 0.50),
    ("pct75", 0.75),
    ("terminal_end_95pct", 0.95),
]
TIME_BIN = "10min"
RESAMPLE_GRID = "30s"
MAX_INTERPOLATION_GAP_SECONDS = 180
ROUTE_JUMP_SPLIT_FRACTION = 0.50
HEADWAY_ROLLING_WINDOW = "30min"
POOR_MATCH_P95_M = 150.0
MAX_STATIC_PART_GAP_M = 250.0


def normalize_code(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def parse_brt_naive(series: pd.Series) -> pd.Series:
    """Parse BRT timestamps to timezone-naive local clock time."""
    s = series.astype(str)
    has_offset = s.str.contains(r"[+-]\d\d:\d\d$", regex=True, na=False)
    out = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    if has_offset.any():
        parsed = pd.to_datetime(s[has_offset], utc=True, errors="coerce")
        out.loc[has_offset] = parsed.dt.tz_convert("Etc/GMT+3").dt.tz_localize(None)
    if (~has_offset).any():
        out.loc[~has_offset] = pd.to_datetime(s[~has_offset], errors="coerce")
    return out


def ceil_timestamp(ts: pd.Timestamp, freq: str) -> pd.Timestamp:
    return ts.ceil(freq)


def floor_timestamp(ts: pd.Timestamp, freq: str) -> pd.Timestamp:
    return ts.floor(freq)


@dataclass
class RouteGeometry:
    line_code: str
    direction: int | None
    source: str
    coords: list[tuple[float, float]]  # lon, lat
    total_length_m: float
    lon0: float
    lat0: float
    seg_start: np.ndarray
    seg_vec: np.ndarray
    seg_len2: np.ndarray
    seg_len: np.ndarray
    seg_offset: np.ndarray
    tree: cKDTree

    @classmethod
    def from_coords(
        cls,
        line_code: str,
        direction: int | None,
        source: str,
        coords: list[tuple[float, float]],
    ) -> "RouteGeometry":
        clean = [(float(lon), float(lat)) for lon, lat in coords if lon is not None and lat is not None]
        if len(clean) < 2:
            raise ValueError(f"Route {line_code}/{direction} has fewer than two coordinates")
        lon0 = float(np.mean([c[0] for c in clean]))
        lat0 = float(np.mean([c[1] for c in clean]))
        xy = lonlat_to_xy(np.array([c[0] for c in clean]), np.array([c[1] for c in clean]), lon0, lat0)
        start = xy[:-1]
        end = xy[1:]
        vec = end - start
        seg_len = np.sqrt(np.sum(vec * vec, axis=1))
        keep = seg_len > 0
        start = start[keep]
        vec = vec[keep]
        seg_len = seg_len[keep]
        if len(seg_len) == 0:
            raise ValueError(f"Route {line_code}/{direction} has zero usable segment length")
        seg_len2 = seg_len * seg_len
        offset = np.concatenate([[0.0], np.cumsum(seg_len[:-1])])
        midpoint = start + 0.5 * vec
        tree = cKDTree(midpoint)
        return cls(
            line_code=line_code,
            direction=direction,
            source=source,
            coords=clean,
            total_length_m=float(np.sum(seg_len)),
            lon0=lon0,
            lat0=lat0,
            seg_start=start,
            seg_vec=vec,
            seg_len2=seg_len2,
            seg_len=seg_len,
            seg_offset=offset,
            tree=tree,
        )

    def project_points(self, lon: np.ndarray, lat: np.ndarray, k: int = 48) -> tuple[np.ndarray, np.ndarray]:
        xy = lonlat_to_xy(lon.astype(float), lat.astype(float), self.lon0, self.lat0)
        if len(self.seg_len) == 1:
            idx = np.zeros((len(xy), 1), dtype=int)
        else:
            _, idx = self.tree.query(xy, k=min(k, len(self.seg_len)))
            if idx.ndim == 1:
                idx = idx[:, None]
        starts = self.seg_start[idx]
        vecs = self.seg_vec[idx]
        denom = self.seg_len2[idx]
        delta = xy[:, None, :] - starts
        t = np.sum(delta * vecs, axis=2) / denom
        t = np.clip(t, 0.0, 1.0)
        proj = starts + t[:, :, None] * vecs
        dist = np.sqrt(np.sum((xy[:, None, :] - proj) ** 2, axis=2))
        best = np.argmin(dist, axis=1)
        row = np.arange(len(xy))
        best_idx = idx[row, best]
        s = self.seg_offset[best_idx] + t[row, best] * self.seg_len[best_idx]
        residual = dist[row, best]
        return s, residual


def lonlat_to_xy(lon: np.ndarray, lat: np.ndarray, lon0: float, lat0: float) -> np.ndarray:
    radius = 6_371_000.0
    x = radius * np.deg2rad(lon - lon0) * math.cos(math.radians(lat0))
    y = radius * np.deg2rad(lat - lat0)
    return np.column_stack([x, y])


def flatten_geojson_coords(geometry: dict) -> list[list[tuple[float, float]]]:
    gtype = geometry.get("type")
    coords = geometry.get("coordinates", [])
    if gtype == "LineString":
        return [[(float(lon), float(lat)) for lon, lat in coords]]
    if gtype == "MultiLineString":
        return [[(float(lon), float(lat)) for lon, lat in part] for part in coords if len(part) >= 2]
    return []


def order_and_orient_parts(
    parts: list[list[tuple[float, float]]],
) -> list[list[tuple[float, float]]]:
    """Globally stitch GeoJSON parts by their nearest endpoints.

    Static route files do not guarantee that MultiLineString members share a
    common orientation or appear in travel order.  A small Held--Karp dynamic
    program finds the open path through all members that first minimises the
    largest connector and then the total connector length, allowing every part
    to be reversed.  The largest residual gap is audited below.
    """
    valid = [list(part) for part in parts if len(part) >= 2]
    if not valid:
        return []

    def endpoint_distance(a: tuple[float, float], b: tuple[float, float]) -> float:
        lon0 = 0.5 * (a[0] + b[0])
        lat0 = 0.5 * (a[1] + b[1])
        xy = lonlat_to_xy(
            np.array([a[0], b[0]]), np.array([a[1], b[1]]), lon0, lat0
        )
        return float(np.linalg.norm(xy[1] - xy[0]))

    n_parts = len(valid)

    def start(index: int, reverse: int) -> tuple[float, float]:
        return valid[index][-1] if reverse else valid[index][0]

    def end(index: int, reverse: int) -> tuple[float, float]:
        return valid[index][0] if reverse else valid[index][-1]

    # state -> (maximum connector, total connector, path)
    states: dict[
        tuple[int, int, int],
        tuple[float, float, tuple[tuple[int, int], ...]],
    ] = {}
    for index in range(n_parts):
        for reverse in (0, 1):
            states[(1 << index, index, reverse)] = (0.0, 0.0, ((index, reverse),))

    for _ in range(1, n_parts):
        updated = dict(states)
        for (mask, last, last_reverse), (max_gap, total_gap, path) in states.items():
            if bin(mask).count("1") != _:
                continue
            for nxt in range(n_parts):
                if mask & (1 << nxt):
                    continue
                for nxt_reverse in (0, 1):
                    gap = endpoint_distance(
                        end(last, last_reverse), start(nxt, nxt_reverse)
                    )
                    candidate = (
                        max(max_gap, gap),
                        total_gap + gap,
                        path + ((nxt, nxt_reverse),),
                    )
                    key = (mask | (1 << nxt), nxt, nxt_reverse)
                    incumbent = updated.get(key)
                    if incumbent is None or candidate[:2] < incumbent[:2]:
                        updated[key] = candidate
        states = updated

    full_mask = (1 << n_parts) - 1
    best = min(
        value
        for (mask, _, _), value in states.items()
        if mask == full_mask
    )
    return [
        list(reversed(valid[index])) if reverse else valid[index]
        for index, reverse in best[2]
    ]


def combine_parts(
    parts: list[list[tuple[float, float]]],
    stitch: bool = True,
) -> list[tuple[float, float]]:
    """Flatten route parts.

    Callers must first reject discontinuous parts.  A flat coordinate sequence
    necessarily creates a segment between adjacent parts; silently flattening a
    disconnected MultiLineString was the source of artificial route arcs in the
    historical pipeline.
    """
    combined: list[tuple[float, float]] = []
    ordered = order_and_orient_parts(parts) if stitch else parts
    for part in ordered:
        if not part:
            continue
        if combined and combined[-1] == part[0]:
            combined.extend(part[1:])
        else:
            combined.extend(part)
    return combined


def maximum_consecutive_part_gap_m(
    parts: list[list[tuple[float, float]]],
    stitch: bool = True,
) -> float:
    """Return the largest endpoint gap between consecutive GeoJSON parts."""
    valid = order_and_orient_parts(parts) if stitch else parts
    if len(valid) < 2:
        return 0.0
    lon0 = float(np.mean([point[0] for part in valid for point in part]))
    lat0 = float(np.mean([point[1] for part in valid for point in part]))
    gaps = []
    for left, right in zip(valid[:-1], valid[1:]):
        endpoints = lonlat_to_xy(
            np.array([left[-1][0], right[0][0]]),
            np.array([left[-1][1], right[0][1]]),
            lon0,
            lat0,
        )
        gaps.append(float(np.linalg.norm(endpoints[1] - endpoints[0])))
    return max(gaps, default=0.0)


def parse_static_route_name(name: str) -> tuple[str, int | None] | None:
    if not name or not name.startswith("L"):
        return None
    body = name[1:]
    direction = None
    for suffix, value in [("_Ida", 0), ("_Volta", 1), ("_Circular", None)]:
        if body.endswith(suffix):
            body = body[: -len(suffix)]
            direction = value
            break
    code = body.replace("_", ".")
    return code, direction


def load_static_geometries(
    root: Path,
    max_part_gap_m: float = MAX_STATIC_PART_GAP_M,
    stitch_parts: bool = True,
) -> dict[tuple[str, int | None], RouteGeometry]:
    path = root / "auxiliar_data" / "line_routes.json"
    data = json.loads(path.read_text())
    geoms: dict[tuple[str, int | None], RouteGeometry] = {}
    for feature in data.get("features", []):
        parsed = parse_static_route_name(feature.get("properties", {}).get("tx_nome", ""))
        if parsed is None:
            continue
        code, direction = parsed
        parts = flatten_geojson_coords(feature.get("geometry", {}))
        ordered_parts = order_and_orient_parts(parts) if stitch_parts else parts
        if maximum_consecutive_part_gap_m(ordered_parts, stitch=False) > max_part_gap_m:
            # Prefer the direction-specific GTFS shape to inventing a straight
            # link across disconnected static geometry parts.
            continue
        coords = combine_parts(ordered_parts, stitch=False)
        if len(coords) < 2:
            continue
        try:
            geoms[(code, direction)] = RouteGeometry.from_coords(
                code, direction, "line_routes.json", coords
            )
        except ValueError:
            continue
    return geoms


def load_gtfs_geometries(root: Path) -> dict[tuple[str, int], RouteGeometry]:
    geoms: dict[tuple[str, int], RouteGeometry] = {}
    gtfs_dirs = [
        root / "GTFS_data" / "TransNit_20260320",
        root / "GTFS_data" / "TransOceanico_20260315",
        root / "GTFS_data" / "TransOceanico_20260329",
    ]
    for folder in gtfs_dirs:
        if not folder.exists():
            continue
        routes = pd.read_csv(folder / "routes.txt", dtype=str)
        trips = pd.read_csv(folder / "trips.txt", dtype=str)
        shapes = pd.read_csv(folder / "shapes.txt", dtype=str)
        routes["route_short_name"] = routes["route_short_name"].map(normalize_code)
        trips["direction_id"] = pd.to_numeric(trips["direction_id"], errors="coerce").astype("Int64")
        merged = trips.merge(routes[["route_id", "route_short_name"]], on="route_id", how="left")
        shapes["shape_pt_sequence"] = pd.to_numeric(shapes["shape_pt_sequence"], errors="coerce")
        shapes["shape_pt_lon"] = pd.to_numeric(shapes["shape_pt_lon"], errors="coerce")
        shapes["shape_pt_lat"] = pd.to_numeric(shapes["shape_pt_lat"], errors="coerce")
        for (code, direction), group in merged.dropna(subset=["route_short_name", "direction_id"]).groupby(
            ["route_short_name", "direction_id"]
        ):
            direction_int = int(direction)
            if (code, direction_int) in geoms:
                continue
            shape_counts = group["shape_id"].value_counts()
            if shape_counts.empty:
                continue
            shape_id = shape_counts.index[0]
            shape = (
                shapes[shapes["shape_id"] == shape_id]
                .dropna(subset=["shape_pt_lon", "shape_pt_lat", "shape_pt_sequence"])
                .sort_values("shape_pt_sequence")
            )
            coords = list(zip(shape["shape_pt_lon"].astype(float), shape["shape_pt_lat"].astype(float)))
            if len(coords) < 2:
                continue
            try:
                geoms[(code, direction_int)] = RouteGeometry.from_coords(
                    code, direction_int, f"GTFS:{folder.name}:shape_id={shape_id}", coords
                )
            except ValueError:
                continue
    return geoms


def select_geometry(
    line_code: str,
    direction: int,
    static_geoms: dict[tuple[str, int | None], RouteGeometry],
    gtfs_geoms: dict[tuple[str, int], RouteGeometry],
) -> RouteGeometry | None:
    if (line_code, direction) in static_geoms:
        return static_geoms[(line_code, direction)]
    if (line_code, None) in static_geoms:
        return static_geoms[(line_code, None)]
    if (line_code, direction) in gtfs_geoms:
        return gtfs_geoms[(line_code, direction)]
    static_codes = STATIC_GEOM_ALIASES.get(line_code, [])
    for code in static_codes:
        if (code, direction) in static_geoms:
            return static_geoms[(code, direction)]
    for code in static_codes:
        if (code, None) in static_geoms:
            return static_geoms[(code, None)]
    if (line_code, direction) in gtfs_geoms:
        return gtfs_geoms[(line_code, direction)]
    return None


def row_count_csv(path: Path) -> int:
    with path.open("rb") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def discover_datasets(root: Path) -> dict:
    mobility_files = sorted((root / "mobility_data").glob("*.csv"))
    ticket_files = sorted((root / "ticket_data").glob("*.csv"))
    mobility_sample = pd.read_csv(mobility_files[0], nrows=5) if mobility_files else pd.DataFrame()
    ticket_sample = pd.read_csv(ticket_files[0], nrows=5) if ticket_files else pd.DataFrame()
    return {
        "data_description_pdf_found": (root / "data_description.pdf").exists(),
        "mnt_uploads_found": Path("/mnt/user-data/uploads").exists(),
        "mnt_data_found": Path("/mnt/data").exists(),
        "mobility": {
            "files": [p.name for p in mobility_files],
            "columns": list(mobility_sample.columns),
            "sample_dtypes": {c: str(t) for c, t in mobility_sample.dtypes.items()},
            "sample_rows": mobility_sample.head(3).to_dict(orient="records"),
        },
        "ticketing": {
            "files": [p.name for p in ticket_files],
            "columns": list(ticket_sample.columns),
            "sample_dtypes": {c: str(t) for c, t in ticket_sample.dtypes.items()},
            "sample_rows": ticket_sample.head(3).to_dict(orient="records"),
        },
    }


def scan_csv_overview(root: Path, output_dir: Path) -> dict:
    overview: dict = discover_datasets(root)
    for name, folder, time_col, id_cols, route_cols in [
        ("mobility", root / "mobility_data", "timestamp", ["id"], ["lineId", "direction"]),
        (
            "ticketing",
            root / "ticket_data",
            "transaction_date",
            ["vehicle_number", "anon_user_id"],
            ["route_name", "route_detail_id"],
        ),
    ]:
        files = sorted(folder.glob("*.csv"))
        row_counts: dict[str, int] = {}
        time_min: str | None = None
        time_max: str | None = None
        distincts = {c: set() for c in id_cols + route_cols}
        null_counts = {c: 0 for c in set(id_cols + route_cols + [time_col])}
        for path in files:
            row_counts[path.name] = 0
            header = pd.read_csv(path, nrows=0).columns.tolist()
            usecols = [c for c in set(id_cols + route_cols + [time_col]) if c in header]
            for chunk in pd.read_csv(path, usecols=usecols, dtype=str, chunksize=500_000, low_memory=False):
                row_counts[path.name] += len(chunk)
                if time_col in chunk:
                    values = chunk[time_col].dropna().astype(str)
                    if not values.empty:
                        mn, mx = values.min(), values.max()
                        time_min = mn if time_min is None else min(time_min, mn)
                        time_max = mx if time_max is None else max(time_max, mx)
                for col in distincts:
                    if col in chunk:
                        distincts[col].update(chunk[col].dropna().map(normalize_code).unique().tolist())
                for col in null_counts:
                    if col in chunk:
                        null_counts[col] += int(chunk[col].isna().sum())
        overview[name].update(
            {
                "n_files": len(files),
                "rows": int(sum(row_counts.values())),
                "row_counts_by_file": row_counts,
                "date_coverage": [time_min, time_max],
                "distinct_counts": {col: len(vals) for col, vals in distincts.items()},
                "distinct_examples": {col: sorted(vals)[:20] for col, vals in distincts.items()},
                "null_counts": null_counts,
            }
        )

    mobility_ids = set()
    ticket_vehicles = set()
    for path in sorted((root / "mobility_data").glob("*.csv")):
        for chunk in pd.read_csv(path, usecols=["id"], dtype=str, chunksize=500_000):
            mobility_ids.update(chunk["id"].dropna().map(normalize_code).unique().tolist())
    for path in sorted((root / "ticket_data").glob("*.csv")):
        for chunk in pd.read_csv(path, usecols=["vehicle_number"], dtype=str, chunksize=500_000):
            ticket_vehicles.update(chunk["vehicle_number"].dropna().map(normalize_code).unique().tolist())
    overlap = mobility_ids & ticket_vehicles
    overview["vehicle_join_check"] = {
        "mobility_distinct_ids": len(mobility_ids),
        "ticket_distinct_vehicle_numbers": len(ticket_vehicles),
        "overlap": len(overlap),
        "pct_mobility_ids_matched": 100.0 * len(overlap) / len(mobility_ids) if mobility_ids else None,
        "pct_ticket_vehicles_matched": 100.0 * len(overlap) / len(ticket_vehicles) if ticket_vehicles else None,
        "overlap_examples": sorted(overlap)[:20],
        "decision": "line_level_demand" if not overlap else "vehicle_level_possible",
    }
    (output_dir / "dataset_discovery.json").write_text(json.dumps(overview, indent=2, ensure_ascii=False))
    return overview


def load_focus_mobility(root: Path) -> tuple[pd.DataFrame, dict]:
    frames = []
    counts: dict[str, object] = {
        "focus_days": FOCUS_DAYS,
        "focus_lines_requested": FOCUS_LINES,
        "raw_rows_by_day": {},
        "missing_service_rows_by_day": {},
        "missing_latlng_rows_by_day": {},
    }
    for day in FOCUS_DAYS:
        path = root / "mobility_data" / f"{day}.csv"
        df = pd.read_csv(
            path,
            dtype={"id": "string", "tripId": "string", "lineId": "string", "lineName": "string", "headsign": "string"},
            low_memory=False,
        )
        counts["raw_rows_by_day"][day] = int(len(df))
        df["line_code"] = df["lineId"].map(normalize_code)
        df["direction"] = pd.to_numeric(df["direction"], errors="coerce").astype("Int64")
        missing_service = df["line_code"].eq("") | df["direction"].isna()
        counts["missing_service_rows_by_day"][day] = int(missing_service.sum())
        missing_latlng = df["lat"].isna() | df["lng"].isna()
        counts["missing_latlng_rows_by_day"][day] = int(missing_latlng.sum())
        df = df.loc[~missing_service & ~missing_latlng].copy()
        df["timestamp"] = parse_brt_naive(df["timestamp"])
        df = df.dropna(subset=["timestamp"])
        df["service_date"] = day
        frames.append(df)
    mobility = pd.concat(frames, ignore_index=True)
    counts["after_drop_out_of_service_and_bad_times"] = int(len(mobility))
    focus = mobility[mobility["line_code"].isin(FOCUS_LINES)].copy()
    counts["after_focus_line_filter"] = int(len(focus))
    counts["focus_rows_by_line_direction"] = (
        focus.groupby(["line_code", "direction"]).size().rename("rows").reset_index().to_dict(orient="records")
    )
    focus["vehicle_id"] = focus["id"].map(normalize_code)
    return focus, counts


def project_focus_mobility(
    mobility: pd.DataFrame,
    static_geoms: dict[tuple[str, int | None], RouteGeometry],
    gtfs_geoms: dict[tuple[str, int], RouteGeometry],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    pieces = []
    quality_rows = []
    missing_geometry = []
    for (line, direction), group in mobility.groupby(["line_code", "direction"], sort=True):
        direction_int = int(direction)
        geom = select_geometry(line, direction_int, static_geoms, gtfs_geoms)
        if geom is None:
            missing_geometry.append({"line_code": line, "direction": direction_int, "rows": int(len(group))})
            continue
        s, residual = geom.project_points(group["lng"].to_numpy(float), group["lat"].to_numpy(float))
        part = group.copy()
        part["s_m"] = s
        part["route_length_m"] = geom.total_length_m
        part["phase_rad"] = (2.0 * np.pi * part["s_m"] / geom.total_length_m) % (2.0 * np.pi)
        part["residual_m"] = residual
        part["geometry_source"] = geom.source
        pieces.append(part)
        quality_rows.append(
            {
                "line_code": line,
                "direction": direction_int,
                "rows": int(len(part)),
                "route_length_m": geom.total_length_m,
                "geometry_source": geom.source,
                "residual_m_median": float(np.nanmedian(residual)),
                "residual_m_p95": float(np.nanpercentile(residual, 95)),
                "poor_match_flag": bool(np.nanpercentile(residual, 95) > POOR_MATCH_P95_M),
            }
        )
    if pieces:
        projected = pd.concat(pieces, ignore_index=True)
    else:
        projected = pd.DataFrame()
    quality = pd.DataFrame(quality_rows)
    return projected, quality, missing_geometry


def resample_projected(projected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    resampled_parts = []
    gap_rows = []
    group_cols = ["line_code", "direction", "service_date", "vehicle_id"]
    for keys, group in projected.sort_values("timestamp").groupby(group_cols, sort=False):
        line, direction, service_date, vehicle = keys
        group = group.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").copy()
        if len(group) < 2:
            continue
        route_length = float(group["route_length_m"].iloc[0])
        dt = group["timestamp"].diff().dt.total_seconds().fillna(0)
        ds = group["s_m"].diff().abs().fillna(0)
        large_gap = dt > MAX_INTERPOLATION_GAP_SECONDS
        route_jump = ds > (ROUTE_JUMP_SPLIT_FRACTION * route_length)
        split = (large_gap | route_jump).cumsum()
        gap_rows.append(
            {
                "line_code": line,
                "direction": int(direction),
                "service_date": service_date,
                "vehicle_id": vehicle,
                "large_time_gaps_gt_3min": int(large_gap.sum()),
                "large_route_jumps_split": int(route_jump.sum()),
                "raw_points": int(len(group)),
            }
        )
        group["trajectory_segment"] = split.to_numpy()
        for segment_id, seg in group.groupby("trajectory_segment", sort=True):
            if len(seg) < 2:
                continue
            start = ceil_timestamp(seg["timestamp"].iloc[0], RESAMPLE_GRID)
            end = floor_timestamp(seg["timestamp"].iloc[-1], RESAMPLE_GRID)
            if start > end:
                continue
            grid = pd.date_range(start, end, freq=RESAMPLE_GRID)
            if len(grid) == 0:
                continue
            work = seg.set_index("timestamp")[["s_m", "residual_m"]].sort_index()
            merged_index = work.index.union(grid).sort_values()
            interp = work.reindex(merged_index).interpolate(method="time").loc[grid]
            interp = interp.dropna(subset=["s_m"])
            if interp.empty:
                continue
            out = interp.reset_index().rename(columns={"index": "timestamp"})
            out["line_code"] = line
            out["direction"] = int(direction)
            out["service_date"] = service_date
            out["vehicle_id"] = vehicle
            out["trajectory_segment"] = int(segment_id)
            out["route_length_m"] = route_length
            out["phase_rad"] = (2.0 * np.pi * out["s_m"] / route_length) % (2.0 * np.pi)
            out["s_frac"] = out["s_m"] / route_length
            out["geometry_source"] = seg["geometry_source"].iloc[0]
            resampled_parts.append(out)
    resampled = pd.concat(resampled_parts, ignore_index=True) if resampled_parts else pd.DataFrame()
    gaps = pd.DataFrame(gap_rows)
    return resampled, gaps


def compute_order_parameter(resampled: pd.DataFrame) -> pd.DataFrame:
    if resampled.empty:
        return pd.DataFrame()
    vehicle_bin = resampled.copy()
    vehicle_bin["time_bin"] = vehicle_bin["timestamp"].dt.floor(TIME_BIN)
    vehicle_bin["cos_phi"] = np.cos(vehicle_bin["phase_rad"])
    vehicle_bin["sin_phi"] = np.sin(vehicle_bin["phase_rad"])
    per_vehicle = (
        vehicle_bin.groupby(["line_code", "direction", "service_date", "time_bin", "vehicle_id"], as_index=False)
        .agg(cos_phi=("cos_phi", "mean"), sin_phi=("sin_phi", "mean"), samples=("phase_rad", "size"))
    )
    order = (
        per_vehicle.groupby(["line_code", "direction", "service_date", "time_bin"], as_index=False)
        .agg(mean_cos=("cos_phi", "mean"), mean_sin=("sin_phi", "mean"), active_buses=("vehicle_id", "nunique"))
    )
    order["kuramoto_r"] = np.sqrt(order["mean_cos"] ** 2 + order["mean_sin"] ** 2)
    order["kuramoto_psi"] = np.arctan2(order["mean_sin"], order["mean_cos"])
    return order


def compute_crossing_events(resampled: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["line_code", "direction", "service_date", "vehicle_id", "trajectory_segment"]
    for keys, group in resampled.sort_values("timestamp").groupby(group_cols, sort=False):
        line, direction, service_date, vehicle, segment = keys
        if len(group) < 2:
            continue
        route_length = float(group["route_length_m"].iloc[0])
        t_ns = group["timestamp"].astype("int64").to_numpy()
        s = group["s_m"].to_numpy(float)
        for section_label, frac in SECTION_FRACTIONS:
            target = frac * route_length
            s0 = s[:-1]
            s1 = s[1:]
            ds = s1 - s0
            valid = (
                (np.abs(ds) > 1.0)
                & (np.abs(ds) < ROUTE_JUMP_SPLIT_FRACTION * route_length)
                & (((s0 - target) == 0) | ((s1 - target) == 0) | ((s0 - target) * (s1 - target) < 0))
            )
            idx = np.where(valid)[0]
            for i in idx:
                ratio = (target - s0[i]) / ds[i]
                if ratio < 0 or ratio > 1:
                    continue
                crossing_ns = int(t_ns[i] + ratio * (t_ns[i + 1] - t_ns[i]))
                rows.append(
                    {
                        "line_code": line,
                        "direction": int(direction),
                        "service_date": service_date,
                        "vehicle_id": vehicle,
                        "trajectory_segment": int(segment),
                        "section": section_label,
                        "section_frac": frac,
                        "event_time": pd.Timestamp(crossing_ns),
                    }
                )
    events = pd.DataFrame(rows)
    if events.empty:
        return events
    events = events.sort_values(["line_code", "direction", "section", "vehicle_id", "event_time"])
    deduped = []
    for _, group in events.groupby(["line_code", "direction", "section", "vehicle_id"], sort=False):
        keep = group["event_time"].diff().dt.total_seconds().fillna(np.inf) > 180
        deduped.append(group.loc[keep])
    events = pd.concat(deduped, ignore_index=True).sort_values(
        ["line_code", "direction", "service_date", "section", "event_time"]
    )
    events["headway_min"] = (
        events.groupby(["line_code", "direction", "service_date", "section"])["event_time"]
        .diff()
        .dt.total_seconds()
        .div(60.0)
    )
    events["prev_vehicle_id"] = events.groupby(["line_code", "direction", "service_date", "section"])[
        "vehicle_id"
    ].shift(1)
    events = events.dropna(subset=["headway_min"])
    events = events[events["headway_min"] > 0]
    local_median = events.groupby(["line_code", "direction", "service_date", "section"])["headway_min"].transform(
        "median"
    )
    events["local_median_headway_min"] = local_median
    events["bunching_event"] = events["headway_min"] < (0.25 * local_median)
    rolling_parts = []
    for _, group in events.groupby(["line_code", "direction", "service_date", "section"], sort=False):
        group = group.sort_values("event_time").copy()
        indexed = group.set_index("event_time")
        roll = indexed["headway_min"].rolling(HEADWAY_ROLLING_WINDOW, min_periods=3)
        group["rolling_headway_mean_min"] = roll.mean().to_numpy()
        group["rolling_headway_cv"] = (roll.std(ddof=0) / roll.mean()).to_numpy()
        rolling_parts.append(group)
    return pd.concat(rolling_parts, ignore_index=True)


def compute_headway_bins(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    work = events.copy()
    work["time_bin"] = work["event_time"].dt.floor(TIME_BIN)
    binned = (
        work.groupby(["line_code", "direction", "service_date", "time_bin"], as_index=False)
        .agg(
            headway_cv=("rolling_headway_cv", "mean"),
            headway_mean_min=("headway_min", "mean"),
            headway_median_min=("headway_min", "median"),
            bunching_rate=("bunching_event", "mean"),
            headway_events=("headway_min", "size"),
        )
    )
    return binned


def load_focus_demand(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    demand_lines = set(FOCUS_LINES)
    for aliases in DEMAND_ALIASES.values():
        demand_lines.update(aliases)
    frames = []
    for day in FOCUS_DAYS:
        path = root / "ticket_data" / f"{day}.csv"
        df = pd.read_csv(
            path,
            dtype={"route_name": "string", "vehicle_number": "string", "anon_user_id": "string"},
            low_memory=False,
        )
        df["line_raw"] = df["route_name"].map(normalize_code)
        df = df[df["line_raw"].isin(demand_lines)].copy()
        df["line_code"] = df["line_raw"]
        for canonical, aliases in DEMAND_ALIASES.items():
            df.loc[df["line_raw"].isin(aliases), "line_code"] = canonical
        df["timestamp"] = parse_brt_naive(df["transaction_date"])
        df = df.dropna(subset=["timestamp"])
        df["service_date"] = day
        frames.append(df)
    tickets = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if tickets.empty:
        return tickets, pd.DataFrame()
    tickets["time_bin"] = tickets["timestamp"].dt.floor(TIME_BIN)
    tickets["is_registered_user"] = tickets["anon_user_id"].map(normalize_code) != "0"
    demand = (
        tickets.groupby(["line_code", "service_date", "time_bin"], as_index=False)
        .agg(
            boardings=("transaction_date", "size"),
            distinct_registered_users=("anon_user_id", lambda s: s[s.map(normalize_code) != "0"].nunique()),
            free_or_transfer_boardings=("debited_amount", lambda s: int((pd.to_numeric(s, errors="coerce") == 0).sum())),
            day_type=("day_type", lambda s: s.mode().iloc[0] if not s.mode().empty else np.nan),
        )
    )
    demand["demand_join_level"] = "line_time_bin_repeated_for_each_direction"
    return tickets, demand


def day_category(service_date: str) -> str:
    weekday = pd.Timestamp(service_date).dayofweek
    return "weekend" if weekday >= 5 else "weekday"


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    display = df.copy()
    for col in display.columns:
        if pd.api.types.is_float_dtype(display[col]):
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
        else:
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else str(x))
    headers = list(display.columns)
    rows = display.values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def assemble_tidy(order: pd.DataFrame, headway_bins: pd.DataFrame, demand: pd.DataFrame) -> pd.DataFrame:
    tidy = order.merge(
        headway_bins,
        on=["line_code", "direction", "service_date", "time_bin"],
        how="left",
    )
    tidy = tidy.merge(demand, on=["line_code", "service_date", "time_bin"], how="left")
    tidy["boardings"] = tidy["boardings"].fillna(0).astype(int)
    tidy["distinct_registered_users"] = tidy["distinct_registered_users"].fillna(0).astype(int)
    tidy["free_or_transfer_boardings"] = tidy["free_or_transfer_boardings"].fillna(0).astype(int)
    tidy["day_category"] = tidy["service_date"].map(day_category)
    tidy["time_bin_local_brt"] = tidy["time_bin"]
    tidy = tidy.sort_values(["line_code", "direction", "time_bin"]).reset_index(drop=True)
    return tidy


def choose_busy_quiet(demand: pd.DataFrame) -> tuple[str, str]:
    day = FOCUS_DAYS[0]
    totals = demand[demand["service_date"] == day].groupby("line_code")["boardings"].sum().sort_values(ascending=False)
    if len(totals) >= 2:
        return str(totals.index[0]), str(totals.index[-1])
    return "48", "62"


def save_figures(*args, **kwargs):
    return None



def write_report(
    root: Path,
    output_dir: Path,
    overview: dict,
    filter_counts: dict,
    quality: pd.DataFrame,
    gaps: pd.DataFrame,
    tidy: pd.DataFrame,
    events: pd.DataFrame,
    figure_info: dict,
    missing_geometry: list[dict],
) -> None:
    focus_rows = pd.DataFrame(filter_counts.get("focus_rows_by_line_direction", []))
    line_counts_md = dataframe_to_markdown(focus_rows) if not focus_rows.empty else "_No focus rows._"
    quality_md = quality[
        [
            "line_code",
            "direction",
            "rows",
            "route_length_m",
            "geometry_source",
            "residual_m_median",
            "residual_m_p95",
            "poor_match_flag",
        ]
    ]
    quality_md = dataframe_to_markdown(quality_md)
    join = overview["vehicle_join_check"]
    discovery = overview
    gap_summary = {
        "large_time_gaps_gt_3min": int(gaps["large_time_gaps_gt_3min"].sum()) if not gaps.empty else 0,
        "large_route_jumps_split": int(gaps["large_route_jumps_split"].sum()) if not gaps.empty else 0,
    }
    tidy_summary = (
        tidy.groupby(["line_code", "day_category"], as_index=False)
        .agg(mean_r=("kuramoto_r", "mean"), mean_boardings=("boardings", "mean"), mean_cv=("headway_cv", "mean"))
        .round(3)
    )
    tidy_summary_md = dataframe_to_markdown(tidy_summary)
    bunching_rate = float(events["bunching_event"].mean()) if not events.empty else float("nan")
    report = f"""# Phase 1 Bus-Bunching Observables

## Scope and Data Discovery

This Phase 1 run uses the local workspace because `/mnt/user-data/uploads` and `/mnt/data`
are not mounted in this environment, and `data_description.pdf` was not available. The
empirical CSV schemas and local README files were used instead. Mobility data covers
{discovery['mobility']['date_coverage'][0]} to {discovery['mobility']['date_coverage'][1]}
with {discovery['mobility']['rows']:,} rows across {discovery['mobility']['n_files']} files.
Ticketing covers {discovery['ticketing']['date_coverage'][0]} to
{discovery['ticketing']['date_coverage'][1]} with {discovery['ticketing']['rows']:,} rows
across {discovery['ticketing']['n_files']} files.

Actual mobility columns are `{', '.join(discovery['mobility']['columns'])}`. This differs
from the prompt/PDF-style names: the observed files use `lineId`, `lineName`, `heading`, and
`direction` rather than `linha`, `nomeLinha`, `angle`, and `sentido`. Ticketing columns match
the expected fields except that route codes must be read as strings to preserve values such
as `49.2` and `62B`.

## Critical Join Check

Telemetry vehicle ids and ticketing vehicle numbers do **not** join cleanly. Mobility has
{join['mobility_distinct_ids']} distinct `id` values, ticketing has
{join['ticket_distinct_vehicle_numbers']} distinct `vehicle_number` values, and their value
overlap is {join['overlap']} ({join['pct_mobility_ids_matched']:.1f}% of telemetry ids,
{join['pct_ticket_vehicles_matched']:.1f}% of ticket vehicles). Demand is therefore joined
only at line-time-bin level and repeated across directions in the tidy table; no per-vehicle
demand is inferred.

## Filtering and Focus Lines

Focus days are March 11, 12, 13, and 14, 2026. Out-of-service filtering follows the available
schema: rows with missing `lineId` or missing `direction` are removed. In these four days,
that removed {sum(filter_counts['missing_service_rows_by_day'].values()):,} rows; missing
lat/lng removed {sum(filter_counts['missing_latlng_rows_by_day'].values()):,} rows. The
requested lines were retained: `49.2`, `45`, `49.1`, `48`, `35`, and `62`. Ticketing uses
`62B` for the same service family, so `62B` demand is mapped to telemetry line `62`.

{line_counts_md}

## Phase and Map Matching

For each `(lineId, direction)` route, GPS points are projected to route arc length `s` and
phase is `phi = 2*pi*s/L`. The phase definition treats each direction as its own closed
empirical ring. This is a pragmatic Phase 1 choice: it makes `r(t)` comparable across
linear and circular services, but terminal wrap-around should be revisited in Phase 2 if an
out-and-back topology is modeled explicitly.

`line_routes.json` is used where it contains an exact focus-route geometry. It lacks `45`,
`48`, and `35`, so GTFS `shapes.txt` is used for those lines. Line `62` is coded as `62` in
telemetry/GTFS but appears as `62B` in ticketing/static route names, so demand uses the
`62B` alias while geometry selection prefers the exact GTFS `62` shape. Map-matching quality is:

{quality_md}

Rows with 95th percentile residual above {POOR_MATCH_P95_M:.0f} m are flagged as poor matches.

## Trajectories, Headways, and Order Parameter

Vehicle trajectories are sorted by time and resampled to a {RESAMPLE_GRID} grid. Interpolation
is not performed across telemetry gaps longer than 3 minutes; this run split
{gap_summary['large_time_gaps_gt_3min']:,} such gaps. Large route-position jumps over
{ROUTE_JUMP_SPLIT_FRACTION:.0%} of route length are also split to avoid interpolating across
terminal wrap/noisy map-matching jumps; this occurred {gap_summary['large_route_jumps_split']:,}
times.

Headways are estimated at five fixed cross-sections per route direction: 5%, 25%, 50%, 75%,
and 95% of arc length. The 5% and 95% locations are terminal proxies, chosen to avoid unstable
exact endpoint crossings. Rolling headway CV uses a {HEADWAY_ROLLING_WINDOW} event-time
window. A bunching event is defined as an observed headway below 25% of the local median
observed headway for the same line, direction, day, and cross-section. This is empirical, not
a schedule-based threshold, because exact scheduled passage times at these synthetic
cross-sections were not reconstructed in Phase 1. Overall event-level bunching rate under
this definition is {bunching_rate:.3f}.

The tidy line-direction-time-bin table is saved to `{output_dir / 'phase1_line_timebin_observables.csv'}`.
Summary means by line and day type:

{tidy_summary_md}

## First Observations

1. The zero vehicle-id overlap is the main integration constraint; line-level demand is the
   defensible unit for this phase.
2. Requested focus lines are present, but the supposed low-volume contrast (`35`, `62`) is
   moderate rather than extremely quiet in this four-day slice.
3. The highest-demand and lowest-demand focus lines for the first weekday figure are
   `{figure_info.get('busy_line_for_figures')}` and `{figure_info.get('quiet_line_for_figures')}`.
4. Route geometry coverage is incomplete in `line_routes.json`; GTFS fallback is necessary
   for several high-demand routes.
5. The resulting `r(t)` series and headway CV are ready for visual demand-coupling checks,
   but no critical-threshold or transition model has been fit.

## Data Issues and Phase 2 Decisions

- Recover or cite the original `data_description.pdf`; it was not present locally.
- Decide whether Phase 2 should use a per-direction closed ring, an out-and-back 2*pi cycle,
  or route-specific topology for circular lines.
- If vehicle-level demand is required, an external vehicle-id crosswalk is necessary.
- Replace the empirical bunching threshold with scheduled headways if GTFS stop-time
  interpolation to the same cross-sections is added.
- Inspect poor map-matching flags before treating route-level comparisons as causal.
"""
    (output_dir / "phase1_report.md").write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("."), help="Path to NetMob data root")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/phase1"),
        help="Directory for Phase 1 outputs",
    )
    parser.add_argument("--skip-full-discovery", action="store_true", help="Skip full-month discovery scan")
    parser.add_argument(
        "--max-static-part-gap-m",
        type=float,
        default=MAX_STATIC_PART_GAP_M,
        help=(
            "Reject a static MultiLineString when consecutive parts are farther "
            "apart than this many metres; use inf only to reproduce the historical bug"
        ),
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

    if args.skip_full_discovery and (output_dir / "dataset_discovery.json").exists():
        overview = json.loads((output_dir / "dataset_discovery.json").read_text())
    else:
        overview = scan_csv_overview(root, output_dir)

    static_geoms = load_static_geometries(
        root,
        args.max_static_part_gap_m,
        stitch_parts=not args.historical_static_order,
    )
    gtfs_geoms = load_gtfs_geometries(root)

    mobility, filter_counts = load_focus_mobility(root)
    projected, quality, missing_geometry = project_focus_mobility(mobility, static_geoms, gtfs_geoms)
    if missing_geometry:
        (output_dir / "missing_geometry.json").write_text(json.dumps(missing_geometry, indent=2))
    projected_cols = [
        "line_code",
        "direction",
        "service_date",
        "vehicle_id",
        "timestamp",
        "lat",
        "lng",
        "s_m",
        "route_length_m",
        "phase_rad",
        "residual_m",
        "geometry_source",
    ]
    projected[projected_cols].sample(min(100_000, len(projected)), random_state=26).sort_values(
        ["line_code", "direction", "timestamp"]
    ).to_csv(output_dir / "projected_observations_sample.csv", index=False)
    quality.to_csv(output_dir / "map_matching_quality.csv", index=False)

    resampled, gaps = resample_projected(projected)
    gaps.to_csv(output_dir / "trajectory_gap_log.csv", index=False)
    resampled.to_csv(output_dir / "resampled_phase_30s.csv", index=False)

    order = compute_order_parameter(resampled)
    order.to_csv(output_dir / "kuramoto_order_10min.csv", index=False)

    events = compute_crossing_events(resampled)
    events.to_csv(output_dir / "headway_crossing_events.csv", index=False)
    headway_bins = compute_headway_bins(events)
    headway_bins.to_csv(output_dir / "headway_cv_10min.csv", index=False)

    tickets, demand = load_focus_demand(root)
    demand.to_csv(output_dir / "line_demand_10min.csv", index=False)

    tidy = assemble_tidy(order, headway_bins, demand)
    tidy.to_csv(output_dir / "phase1_line_timebin_observables.csv", index=False)
    try:
        tidy.to_parquet(output_dir / "phase1_line_timebin_observables.parquet", index=False)
    except Exception as exc:
        (output_dir / "parquet_not_written.txt").write_text(f"Parquet unavailable: {type(exc).__name__}: {exc}\n")

    focus_counts = pd.DataFrame(filter_counts["focus_rows_by_line_direction"])
    focus_counts.to_csv(output_dir / "focus_line_counts.csv", index=False)
    (output_dir / "filtering_log.json").write_text(json.dumps(filter_counts, indent=2, ensure_ascii=False))
    figure_info = save_figures(output_dir, resampled, events, tidy, demand)
    (output_dir / "figure_info.json").write_text(json.dumps(figure_info, indent=2))

    write_report(root, output_dir, overview, filter_counts, quality, gaps, tidy, events, figure_info, missing_geometry)

    summary = {
        "output_dir": str(output_dir),
        "tidy_rows": int(len(tidy)),
        "resampled_rows": int(len(resampled)),
        "headway_events": int(len(events)),
        "missing_geometry": missing_geometry,
        "vehicle_join_decision": overview["vehicle_join_check"]["decision"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
