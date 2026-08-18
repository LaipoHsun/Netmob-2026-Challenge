"""Build reusable route, vehicle, demand, supply, map-matching, and phase features.

Data contract
-------------
Reads only:  data_after_cleaning/
Writes only: data_after_all_processed/

This stage performs feature engineering, not scientific hypothesis tests and not
publication plotting. Ticket demand is emitted under both the provider-reported clock
and the unconfirmed +3-hour candidate clock so later analyses must choose explicitly.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "data_after_cleaning"
TARGET = PROJECT / "data_after_all_processed"
RNG = np.random.default_rng(2602)

TOP_ROUTE_TARGET = 20
NULL_DRAWS = 5_000
MAX_MEDIAN_RESIDUAL_M = 150.0
LAYOVER_TERMINAL_FRAC = 0.75
LAYOVER_MIN_DURATION_MIN = 10.0
LAYOVER_MAX_SPAN_M = 150.0
MIDPOINT_MAX_DISTANCE_MIN = 20.0
STATIC_ALIASES = {"62": ["62", "62B"]}
TICKET_CANONICAL = {"62B": "62"}


def write_csv(df: pd.DataFrame, relative: str) -> None:
    path = TARGET / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def normalize_code(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    return re.sub(r"^(\d+)\.0$", r"\1", text)


def ticket_line(value: object) -> str:
    code = normalize_code(value)
    return TICKET_CANONICAL.get(code, code)


def lonlat_to_xy(lon: np.ndarray, lat: np.ndarray, lon0: float, lat0: float) -> np.ndarray:
    radius = 6_371_000.0
    x = radius * np.deg2rad(lon - lon0) * math.cos(math.radians(lat0))
    y = radius * np.deg2rad(lat - lat0)
    return np.column_stack([x, y])


@dataclass
class RouteGeometry:
    line_code: str
    direction: int | None
    source: str
    coords: list[tuple[float, float]]
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
    def from_coords(cls, line_code: str, direction: int | None, source: str, coords: list[tuple[float, float]]):
        clean = [(float(lon), float(lat)) for lon, lat in coords if lon is not None and lat is not None]
        if len(clean) < 2:
            raise ValueError("geometry has fewer than two coordinates")
        lon0 = float(np.mean([point[0] for point in clean]))
        lat0 = float(np.mean([point[1] for point in clean]))
        xy = lonlat_to_xy(np.array([point[0] for point in clean]), np.array([point[1] for point in clean]), lon0, lat0)
        start = xy[:-1]
        vec = xy[1:] - start
        seg_len = np.sqrt(np.sum(vec * vec, axis=1))
        keep = seg_len > 0
        start, vec, seg_len = start[keep], vec[keep], seg_len[keep]
        if not len(seg_len):
            raise ValueError("geometry has no nonzero segments")
        seg_len2 = seg_len * seg_len
        offset = np.concatenate([[0.0], np.cumsum(seg_len[:-1])])
        tree = cKDTree(start + 0.5 * vec)
        return cls(line_code, direction, source, clean, float(seg_len.sum()), lon0, lat0, start, vec, seg_len2, seg_len, offset, tree)

    def project(self, lon: np.ndarray, lat: np.ndarray, k: int = 48) -> tuple[np.ndarray, np.ndarray]:
        xy = lonlat_to_xy(lon.astype(float), lat.astype(float), self.lon0, self.lat0)
        if len(self.seg_len) == 1:
            idx = np.zeros((len(xy), 1), dtype=int)
        else:
            _, idx = self.tree.query(xy, k=min(k, len(self.seg_len)))
            if idx.ndim == 1:
                idx = idx[:, None]
        starts, vecs = self.seg_start[idx], self.seg_vec[idx]
        delta = xy[:, None, :] - starts
        t = np.clip(np.sum(delta * vecs, axis=2) / self.seg_len2[idx], 0.0, 1.0)
        projection = starts + t[:, :, None] * vecs
        distance = np.sqrt(np.sum((xy[:, None, :] - projection) ** 2, axis=2))
        best = np.argmin(distance, axis=1)
        row = np.arange(len(xy))
        segment = idx[row, best]
        s = self.seg_offset[segment] + t[row, best] * self.seg_len[segment]
        return s, distance[row, best]


def geometry_parts(geometry: dict) -> list[list[tuple[float, float]]]:
    if geometry.get("type") == "LineString":
        return [[tuple(point) for point in geometry.get("coordinates", [])]]
    if geometry.get("type") == "MultiLineString":
        return [[tuple(point) for point in part] for part in geometry.get("coordinates", []) if len(part) >= 2]
    return []


def combine_parts(parts: list[list[tuple[float, float]]]) -> list[tuple[float, float]]:
    combined: list[tuple[float, float]] = []
    for part in parts:
        if combined and part and combined[-1] == part[0]:
            combined.extend(part[1:])
        else:
            combined.extend(part)
    return combined


def parse_static_name(name: object) -> tuple[str, int | None] | None:
    text = "" if pd.isna(name) else str(name)
    if not text.startswith("L"):
        return None
    body = text[1:]
    direction = None
    for suffix, value in [("_Ida", 0), ("_Volta", 1), ("_Circular", None)]:
        if body.endswith(suffix):
            body = body[: -len(suffix)]
            direction = value
            break
    return body.replace("_", "."), direction


def load_geometries() -> dict[tuple[str, int | None], list[RouteGeometry]]:
    result: dict[tuple[str, int | None], list[RouteGeometry]] = defaultdict(list)
    static = pd.read_csv(SOURCE / "auxiliar_data" / "line_routes.csv")
    for row in static.itertuples(index=False):
        parsed = parse_static_name(getattr(row, "tx_nome", None))
        if not parsed or pd.isna(row.geometry_json):
            continue
        code, direction = parsed
        coords = combine_parts(geometry_parts(json.loads(row.geometry_json)))
        try:
            result[(code, direction)].append(RouteGeometry.from_coords(code, direction, "auxiliary:line_routes", coords))
        except ValueError:
            pass

    for folder in sorted((SOURCE / "GTFS_data").iterdir()):
        if not folder.is_dir():
            continue
        routes = pd.read_csv(folder / "routes.csv", dtype="string")
        trips = pd.read_csv(folder / "trips.csv", dtype="string")
        shapes = pd.read_csv(folder / "shapes.csv", dtype="string")
        routes["route_short_name"] = routes["route_short_name"].map(normalize_code)
        trips["direction_id"] = pd.to_numeric(trips["direction_id"], errors="coerce").astype("Int64")
        merged = trips.merge(routes[["route_id", "route_short_name"]], on="route_id", how="left")
        for column in ["shape_pt_sequence", "shape_pt_lon", "shape_pt_lat"]:
            shapes[column] = pd.to_numeric(shapes[column], errors="coerce")
        for (code, direction), group in merged.dropna(subset=["route_short_name", "direction_id", "shape_id"]).groupby(["route_short_name", "direction_id"]):
            shape_id = group["shape_id"].value_counts().index[0]
            shape = shapes[shapes["shape_id"] == shape_id].dropna(subset=["shape_pt_sequence", "shape_pt_lon", "shape_pt_lat"]).sort_values("shape_pt_sequence")
            coords = list(zip(shape["shape_pt_lon"].astype(float), shape["shape_pt_lat"].astype(float)))
            try:
                result[(str(code), int(direction))].append(RouteGeometry.from_coords(str(code), int(direction), f"GTFS:{folder.name}:{shape_id}", coords))
            except ValueError:
                pass
    return result


def geometry_candidates(geometries, line: str, direction: int) -> list[RouteGeometry]:
    candidates = geometries.get((line, direction), []) + geometries.get((line, None), [])
    for alias in STATIC_ALIASES.get(line, []):
        candidates += geometries.get((alias, direction), []) + geometries.get((alias, None), [])
    unique, seen = [], set()
    for geometry in candidates:
        key = (geometry.source, round(geometry.total_length_m, 3))
        if key not in seen:
            seen.add(key)
            unique.append(geometry)
    return unique


def build_ticket_features() -> tuple[pd.DataFrame, Counter, dict, dict]:
    demand_parts = []
    route_counts = Counter()
    vehicle_transactions = Counter()
    vehicle_days: dict[str, set] = defaultdict(set)
    vehicle_routes: dict[str, set] = defaultdict(set)
    for path in sorted((SOURCE / "ticket_data").glob("*.csv")):
        df = pd.read_csv(path, dtype={"route_name": "string", "vehicle_key": "string", "anon_user_id": "string"}, low_memory=False)
        df["line_code"] = df["route_name"].map(ticket_line)
        route_counts.update(df["line_code"].dropna())
        vehicle_transactions.update(df["vehicle_key"].dropna())
        for key, routes in df.groupby("vehicle_key")["line_code"]:
            vehicle_days[str(key)].add(path.stem)
            vehicle_routes[str(key)].update(routes.dropna().astype(str))
        for basis, column in [
            ("reported_brt", "transaction_time_brt_reported"),
            ("candidate_shifted_3h", "transaction_time_candidate_shifted_3h"),
        ]:
            timestamp = pd.to_datetime(df[column].str.slice(0, 19), errors="coerce")
            work = pd.DataFrame(
                {
                    "line_code": df["line_code"],
                    "service_date": timestamp.dt.strftime("%Y-%m-%d"),
                    "hour": timestamp.dt.floor("h"),
                    "anon_user_id": df["anon_user_id"],
                    "debited_amount": pd.to_numeric(df["debited_amount"], errors="coerce"),
                    "clock_basis": basis,
                }
            ).dropna(subset=["line_code", "hour"])
            grouped = work.groupby(["line_code", "service_date", "hour", "clock_basis"], as_index=False).agg(
                boardings=("line_code", "size"),
                distinct_registered_users=("anon_user_id", lambda x: x[x != "0"].nunique()),
                free_or_transfer_boardings=("debited_amount", lambda x: int((x == 0).sum())),
            )
            demand_parts.append(grouped)
    demand = pd.concat(demand_parts, ignore_index=True).sort_values(["clock_basis", "line_code", "hour"])
    vehicle = pd.DataFrame(
        {
            "ticket_vehicle_key": list(vehicle_transactions),
            "transactions": [vehicle_transactions[key] for key in vehicle_transactions],
            "days_observed": [len(vehicle_days[key]) for key in vehicle_transactions],
            "routes_observed": [len(vehicle_routes[key]) for key in vehicle_transactions],
            "route_list": [" | ".join(sorted(vehicle_routes[key])) for key in vehicle_transactions],
        }
    ).sort_values("transactions", ascending=False)
    return demand, route_counts, vehicle, vehicle_days


def select_routes(route_counts: Counter, geometries) -> pd.DataFrame:
    rows = []
    selected = 0
    for rank, (line, boardings) in enumerate(route_counts.most_common(), 1):
        has0 = bool(geometry_candidates(geometries, line, 0))
        has1 = bool(geometry_candidates(geometries, line, 1))
        use = has0 and has1 and selected < TOP_ROUTE_TARGET
        selected += int(use)
        rows.append({"rank_by_boardings": rank, "line_code": line, "boardings": boardings, "has_direction_0_geometry": has0, "has_direction_1_geometry": has1, "selected_for_phase_features": use})
    return pd.DataFrame(rows)


def choose_geometry(candidates: list[RouteGeometry], group: pd.DataFrame) -> tuple[RouteGeometry | None, dict]:
    if not candidates:
        return None, {}
    sample = group.sample(min(3000, len(group)), random_state=2602) if len(group) else group
    lon = sample["lng"].to_numpy(float)
    lat = sample["lat"].to_numpy(float)
    scores = []
    for geometry in candidates:
        _, residual = geometry.project(lon, lat)
        scores.append((float(np.nanpercentile(residual, 95)), float(np.nanmedian(residual)), geometry))
    scores.sort(key=lambda item: (item[0], item[1]))
    p95, median, geometry = scores[0]
    return geometry, {"geometry_source": geometry.source, "sample_residual_median": median, "sample_residual_p95": p95, "candidate_geometries": len(candidates)}


def bus_hour_from_projected(group: pd.DataFrame, geometry: RouteGeometry, s: np.ndarray, residual: np.ndarray) -> pd.DataFrame:
    part = group[["id", "timestamp", "lineId", "direction", "source_date", "date_quality_flag"]].copy()
    part["vehicle_id"] = part["id"].astype(str)
    part["line_code"] = part["lineId"].map(normalize_code)
    part["s_m"] = s
    part["s_frac"] = s / geometry.total_length_m
    part["phase_arc_rad"] = (2 * np.pi * part["s_frac"]) % (2 * np.pi)
    part["residual_m"] = residual
    part["hour"] = part["timestamp"].dt.floor("h")
    part["abs_midpoint_min"] = (part["timestamp"] - (part["hour"] + pd.Timedelta(minutes=30))).abs().dt.total_seconds() / 60
    part["terminal_sample"] = (part["s_frac"] <= 0.05) | (part["s_frac"] >= 0.95)
    keys = ["line_code", "direction", "source_date", "vehicle_id", "hour"]
    representative = part.sort_values(keys + ["abs_midpoint_min"]).drop_duplicates(keys)[keys + ["phase_arc_rad", "s_m", "s_frac", "abs_midpoint_min"]].rename(columns={"phase_arc_rad": "phase_mid_arc_rad", "s_m": "s_mid_m", "s_frac": "s_mid_frac", "abs_midpoint_min": "nearest_midpoint_min"})
    result = part.groupby(keys, as_index=False).agg(
        samples=("timestamp", "size"),
        first_ts=("timestamp", "min"),
        last_ts=("timestamp", "max"),
        s_min_m=("s_m", "min"),
        s_max_m=("s_m", "max"),
        terminal_frac=("terminal_sample", "mean"),
        median_residual_m=("residual_m", "median"),
        date_quality_flag=("date_quality_flag", "first"),
    ).merge(representative, on=keys, how="left")
    result["route_length_m"] = geometry.total_length_m
    result["geometry_source"] = geometry.source
    result["duration_min"] = (result["last_ts"] - result["first_ts"]).dt.total_seconds() / 60
    result["s_span_m"] = result["s_max_m"] - result["s_min_m"]
    result["exclude_layover"] = (result.samples >= 3) & (result.duration_min >= LAYOVER_MIN_DURATION_MIN) & (result.terminal_frac >= LAYOVER_TERMINAL_FRAC) & (result.s_span_m <= LAYOVER_MAX_SPAN_M)
    result["exclude_sparse_midpoint"] = result.nearest_midpoint_min > MIDPOINT_MAX_DISTANCE_MIN
    result["exclude_bad_residual"] = result.median_residual_m > MAX_MEDIAN_RESIDUAL_M
    result["active_for_order"] = ~(result.exclude_layover | result.exclude_sparse_midpoint | result.exclude_bad_residual)
    return result


def build_mobility_features(geometries, route_selection: pd.DataFrame):
    selected = set(route_selection.loc[route_selection.selected_for_phase_features, "line_code"])
    supply_parts, bus_hour_parts, quality_rows = [], [], []
    vehicle_observations = Counter()
    vehicle_days: dict[str, set] = defaultdict(set)
    vehicle_routes: dict[str, set] = defaultdict(set)
    chosen_geometry: dict[tuple[str, int], RouteGeometry] = {}
    for path in sorted((SOURCE / "mobility_data").glob("*.csv")):
        df = pd.read_csv(path, dtype={"id": "string", "lineId": "string"}, low_memory=False)
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["line_code"] = df["lineId"].map(normalize_code)
        df["direction"] = pd.to_numeric(df["direction"], errors="coerce").astype("Int64")
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
        df["lng"] = pd.to_numeric(df["lng"], errors="coerce")
        df = df.dropna(subset=["id", "timestamp", "line_code", "direction", "lat", "lng"])
        df["hour"] = df.timestamp.dt.floor("h")
        supply_parts.append(df.groupby(["source_date", "line_code", "direction", "hour", "date_quality_flag"], as_index=False).agg(gps_observations=("id", "size"), active_gps_ids=("id", "nunique"), mean_lat=("lat", "mean"), mean_lng=("lng", "mean")))
        vehicle_observations.update(df.id.astype(str))
        for key, routes in df.groupby("id").line_code:
            key = str(key); vehicle_days[key].add(path.stem); vehicle_routes[key].update(routes.astype(str))

        focus = df[df.line_code.isin(selected)]
        for (line, direction), group in focus.groupby(["line_code", "direction"], sort=False):
            key = (str(line), int(direction))
            geometry = chosen_geometry.get(key)
            info = {}
            if geometry is None:
                geometry, info = choose_geometry(geometry_candidates(geometries, key[0], key[1]), group)
                if geometry is not None:
                    chosen_geometry[key] = geometry
            if geometry is None:
                quality_rows.append({"source_date": path.stem, "line_code": line, "direction": int(direction), "geometry_source": "missing", "rows": len(group)})
                continue
            s, residual = geometry.project(group.lng.to_numpy(float), group.lat.to_numpy(float))
            quality_rows.append({"source_date": path.stem, "line_code": line, "direction": int(direction), "geometry_source": geometry.source, "rows": len(group), "residual_m_median": float(np.median(residual)), "residual_m_p95": float(np.percentile(residual, 95)), **info})
            bus_hour_parts.append(bus_hour_from_projected(group, geometry, s, residual))

    supply = pd.concat(supply_parts, ignore_index=True)
    bus_hours = pd.concat(bus_hour_parts, ignore_index=True) if bus_hour_parts else pd.DataFrame()
    vehicle = pd.DataFrame({
        "mobility_id": list(vehicle_observations),
        "gps_observations": [vehicle_observations[key] for key in vehicle_observations],
        "days_observed": [len(vehicle_days[key]) for key in vehicle_observations],
        "routes_observed": [len(vehicle_routes[key]) for key in vehicle_observations],
        "route_list": [" | ".join(sorted(vehicle_routes[key])) for key in vehicle_observations],
    }).sort_values("gps_observations", ascending=False)
    return supply, bus_hours, pd.DataFrame(quality_rows), vehicle


def order_and_null(bus_hours: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    active = bus_hours[bus_hours.active_for_order].copy()
    active["cos_phi"] = np.cos(active.phase_mid_arc_rad)
    active["sin_phi"] = np.sin(active.phase_mid_arc_rad)
    order = active.groupby(["line_code", "direction", "source_date", "hour", "date_quality_flag"], as_index=False).agg(
        mean_cos=("cos_phi", "mean"), mean_sin=("sin_phi", "mean"), N=("vehicle_id", "nunique"), bus_hour_rows=("vehicle_id", "size"), median_residual_m=("median_residual_m", "median")
    )
    order["r1_arc"] = np.sqrt(order.mean_cos**2 + order.mean_sin**2)
    order["psi_arc"] = np.arctan2(order.mean_sin, order.mean_cos)
    null_rows = []
    for n in sorted(order.N.unique()):
        phases = RNG.uniform(0, 2 * np.pi, size=(NULL_DRAWS, int(n)))
        r = np.abs(np.exp(1j * phases).mean(axis=1))
        null_rows.append({"N": int(n), "draws": NULL_DRAWS, "null_mean": r.mean(), "null_sd": r.std(ddof=1), "null_q025": np.quantile(r, .025), "null_q975": np.quantile(r, .975)})
    nulls = pd.DataFrame(null_rows)
    order = order.merge(nulls, on="N", how="left")
    order["r_excess_arc"] = (order.r1_arc - order.null_mean) / (1 - order.null_mean)
    order["z_arc"] = (order.r1_arc - order.null_mean) / order.null_sd
    order["ppc_arc"] = np.where(order.N > 1, (order.N * order.r1_arc**2 - 1) / (order.N - 1), np.nan)
    return order, nulls


def route_correspondence(ticket_counts: Counter, supply: pd.DataFrame) -> pd.DataFrame:
    mobility_counts = supply.groupby("line_code").gps_observations.sum().to_dict()
    routes = sorted(set(ticket_counts) | set(mobility_counts))
    return pd.DataFrame({
        "line_code": routes,
        "ticket_transactions": [ticket_counts.get(route, 0) for route in routes],
        "mobility_observations": [mobility_counts.get(route, 0) for route in routes],
        "present_in_ticket": [route in ticket_counts for route in routes],
        "present_in_mobility": [route in mobility_counts for route in routes],
    })


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    demand, ticket_counts, ticket_vehicle, _ = build_ticket_features()
    geometries = load_geometries()
    selection = select_routes(ticket_counts, geometries)
    supply, bus_hours, map_quality, mobility_vehicle = build_mobility_features(geometries, selection)
    order, nulls = order_and_null(bus_hours)
    cells = order.merge(
        demand,
        left_on=["line_code", "source_date", "hour"],
        right_on=["line_code", "service_date", "hour"],
        how="left",
    )
    cells["lambda_boardings_per_bus"] = cells.boardings / cells.N

    write_csv(demand, "demand/line_hour_demand.csv")
    write_csv(ticket_vehicle, "vehicle/ticket_vehicle_summary.csv")
    write_csv(mobility_vehicle, "vehicle/mobility_vehicle_summary.csv")
    write_csv(selection, "route/route_selection.csv")
    write_csv(route_correspondence(ticket_counts, supply), "route/route_correspondence.csv")
    write_csv(supply, "supply/line_direction_hour_supply.csv")
    write_csv(bus_hours, "supply/bus_hour_phase_candidates.csv")
    write_csv(map_quality, "quality/map_matching_quality_by_date.csv")
    write_csv(nulls, "index/finite_size_null_by_N.csv")
    write_csv(order, "index/line_direction_hour_order.csv")
    write_csv(cells, "analysis_cells/line_direction_hour_features.csv")
    summary = pd.DataFrame([
        {"object": "ticket_demand", "rows": len(demand)},
        {"object": "ticket_vehicle_summary", "rows": len(ticket_vehicle)},
        {"object": "mobility_vehicle_summary", "rows": len(mobility_vehicle)},
        {"object": "line_direction_hour_supply", "rows": len(supply)},
        {"object": "bus_hour_phase_candidates", "rows": len(bus_hours)},
        {"object": "line_direction_hour_order", "rows": len(order)},
        {"object": "line_direction_hour_features", "rows": len(cells)},
    ])
    write_csv(summary, "processing_summary.csv")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
