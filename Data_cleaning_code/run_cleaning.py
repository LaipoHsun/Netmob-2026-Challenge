"""Audit and clean the NetMob challenge release into CSV-only analysis inputs.

Data contract
-------------
Reads only:  data_original/
Writes only: data_after_cleaning/

The process is deliberately conservative. It removes exact duplicate rows and rows
that are structurally unusable, but retains dates with incomplete telemetry and marks
them with a quality flag. Source files under data_original are never modified.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sqlite3
from pathlib import Path

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "data_original"
TARGET = PROJECT / "data_after_cleaning"

MOBILITY_DTYPES = {
    "id": "string",
    "tripId": "string",
    "lineId": "string",
    "lineName": "string",
    "headsign": "string",
}
TICKET_STRING_COLUMNS = [
    "vehicle_number",
    "company_number",
    "route_detail_id",
    "route_name",
    "view_type",
    "anon_user_id",
]

DATE_QUALITY = {
    "2026-03-17": "partial_day_ends_10:46",
    "2026-03-20": "reduced_readings_after_outage",
    "2026-03-22": "abnormal_telemetry_interval",
    "2026-03-28": "fewer_readings_than_expected",
}


def canonical_route(value: object) -> object:
    """Normalize formatting only; preserve meaningful suffixes and leading zeroes."""
    if pd.isna(value):
        return pd.NA
    text = str(value).strip()
    return re.sub(r"^(\d+)\.0$", r"\1", text)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_column_profile(records: list[dict], dataset: str, source_file: str, df: pd.DataFrame) -> None:
    for column in df.columns:
        records.append(
            {
                "dataset": dataset,
                "source_file": source_file,
                "column": column,
                "dtype_after_cleaning": str(df[column].dtype),
                "rows": len(df),
                "missing_count": int(df[column].isna().sum()),
                "missing_rate_pct": 100.0 * float(df[column].isna().mean()) if len(df) else 0.0,
            }
        )


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def clean_mobility(log: list[dict], profiles: list[dict]) -> None:
    output_dir = TARGET / "mobility_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted((SOURCE / "mobility_data").glob("*.csv")):
        raw = pd.read_csv(path, dtype=MOBILITY_DTYPES, low_memory=False)
        raw_rows = len(raw)
        duplicate_mask = raw.duplicated(keep="first")
        df = raw.loc[~duplicate_mask].copy()

        parsed_time = pd.to_datetime(df["timestamp"], errors="coerce")
        lat = pd.to_numeric(df["lat"], errors="coerce")
        lng = pd.to_numeric(df["lng"], errors="coerce")
        heading = pd.to_numeric(df["heading"], errors="coerce")
        direction = pd.to_numeric(df["direction"], errors="coerce")
        invalid = (
            df["id"].isna()
            | df["id"].str.strip().eq("")
            | parsed_time.isna()
            | lat.isna()
            | lng.isna()
            | ~lat.between(-90, 90)
            | ~lng.between(-180, 180)
        )
        invalid_rows = int(invalid.sum())
        df = df.loc[~invalid].copy()
        parsed_time = parsed_time.loc[~invalid]
        lat = lat.loc[~invalid]
        lng = lng.loc[~invalid]
        heading = heading.loc[~invalid]
        direction = direction.loc[~invalid]

        df["id"] = df["id"].str.strip()
        df["timestamp"] = parsed_time.dt.strftime("%Y-%m-%d %H:%M:%S")
        df["tripId"] = df["tripId"].str.strip()
        df["lat"] = lat
        df["lng"] = lng
        df["heading"] = heading
        df["lineId_original"] = df["lineId"]
        df["lineId"] = df["lineId"].map(canonical_route).astype("string")
        df["lineName"] = df["lineName"].str.strip()
        df["headsign"] = df["headsign"].str.strip()
        df["direction"] = direction.astype("Int64")
        df["source_date"] = path.stem
        df["date_quality_flag"] = DATE_QUALITY.get(path.stem, "expected_coverage")
        df["service_status"] = "in_service"
        missing_service = df["lineId"].isna() | df["lineId"].str.strip().eq("") | df["direction"].isna()
        df.loc[missing_service, "service_status"] = "route_or_direction_unavailable"

        write_csv(df, output_dir / path.name)
        add_column_profile(profiles, "mobility", path.name, df)
        log.append(
            {
                "dataset": "mobility",
                "source_file": path.name,
                "raw_rows": raw_rows,
                "exact_duplicates_removed": int(duplicate_mask.sum()),
                "structurally_invalid_rows_removed": invalid_rows,
                "output_rows": len(df),
                "quality_flag": DATE_QUALITY.get(path.stem, "expected_coverage"),
                "cleaning_actions": "parse timestamp/numerics; validate coordinates; trim IDs/names; conservative route normalization; add quality/service flags",
            }
        )


def clean_ticket(log: list[dict], profiles: list[dict]) -> None:
    output_dir = TARGET / "ticket_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    dtype = {column: "string" for column in TICKET_STRING_COLUMNS}
    for path in sorted((SOURCE / "ticket_data").glob("*.csv")):
        raw = pd.read_csv(path, dtype=dtype, low_memory=False)
        raw_rows = len(raw)
        duplicate_mask = raw.duplicated(keep="first")
        df = raw.loc[~duplicate_mask].copy()
        original_time = df["transaction_date"].astype("string")
        parsed_utc = pd.to_datetime(original_time, errors="coerce", utc=True)
        invalid = (
            parsed_utc.isna()
            | df["vehicle_number"].isna()
            | df["company_number"].isna()
            | df["vehicle_number"].str.strip().eq("")
            | df["company_number"].str.strip().eq("")
        )
        invalid_rows = int(invalid.sum())
        df = df.loc[~invalid].copy()
        parsed_utc = parsed_utc.loc[~invalid]
        original_time = original_time.loc[~invalid]
        reported_brt = parsed_utc.dt.tz_convert("America/Sao_Paulo")

        for column in TICKET_STRING_COLUMNS:
            df[column] = df[column].str.strip()
        df["transaction_date_original"] = original_time
        df["transaction_time_brt_reported"] = reported_brt.dt.strftime("%Y-%m-%d %H:%M:%S-03:00")
        df["transaction_time_candidate_shifted_3h"] = (reported_brt + pd.Timedelta(hours=3)).dt.strftime(
            "%Y-%m-%d %H:%M:%S-03:00"
        )
        df["candidate_shift_status"] = "diagnostic_only_not_provider_confirmed"
        df["route_name_original"] = df["route_name"]
        df["route_name"] = df["route_name"].map(canonical_route).astype("string")
        df["vehicle_key"] = df["company_number"] + "::" + df["vehicle_number"]
        df["source_date"] = path.stem

        preferred = [
            "transaction_date_original",
            "transaction_time_brt_reported",
            "transaction_time_candidate_shifted_3h",
            "candidate_shift_status",
        ]
        remaining = [column for column in df.columns if column not in preferred and column != "transaction_date"]
        df = df[preferred + remaining]
        write_csv(df, output_dir / path.name)
        add_column_profile(profiles, "ticket", path.name, df)
        log.append(
            {
                "dataset": "ticket",
                "source_file": path.name,
                "raw_rows": raw_rows,
                "exact_duplicates_removed": int(duplicate_mask.sum()),
                "structurally_invalid_rows_removed": invalid_rows,
                "output_rows": len(df),
                "quality_flag": "expected_coverage",
                "cleaning_actions": "remove exact duplicates; parse reported BRT; preserve original time; add unconfirmed +3h candidate; trim identifiers; add composite vehicle_key; conservative route normalization",
            }
        )


def sniff_delimiter(path: Path) -> str:
    sample = path.read_bytes()[:8192].decode("utf-8-sig", errors="replace")
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ";" if sample.count(";") > sample.count(",") else ","


def clean_delimited_file(
    path: Path,
    output: Path,
    dataset: str,
    log: list[dict],
    profiles: list[dict],
) -> None:
    delimiter = sniff_delimiter(path)
    raw = pd.read_csv(path, sep=delimiter, dtype="string", low_memory=False)
    raw.columns = [str(column).strip() for column in raw.columns]
    raw_rows = len(raw)
    raw = raw.dropna(axis=1, how="all").dropna(axis=0, how="all")
    duplicate_mask = raw.duplicated(keep="first")
    df = raw.loc[~duplicate_mask].copy()
    write_csv(df, output)
    add_column_profile(profiles, dataset, str(path.relative_to(SOURCE)), df)
    log.append(
        {
            "dataset": dataset,
            "source_file": str(path.relative_to(SOURCE)),
            "raw_rows": raw_rows,
            "exact_duplicates_removed": int(duplicate_mask.sum()),
            "structurally_invalid_rows_removed": int(raw_rows - len(raw)),
            "output_rows": len(df),
            "quality_flag": "not_applicable",
            "cleaning_actions": f"detect source delimiter {delimiter!r}; normalize output to UTF-8 comma-separated CSV; trim column names; remove blank rows/columns and exact duplicates",
        }
    )


def geojson_to_csv(path: Path, output: Path, dataset: str, log: list[dict], profiles: list[dict]) -> None:
    obj = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for feature in obj.get("features", []):
        row = dict(feature.get("properties") or {})
        row["feature_id"] = feature.get("id")
        geometry = feature.get("geometry")
        row["geometry_type"] = geometry.get("type") if geometry else None
        row["geometry_json"] = json.dumps(geometry, ensure_ascii=False, separators=(",", ":")) if geometry else None
        rows.append(row)
    df = pd.DataFrame(rows)
    duplicate_mask = df.duplicated(keep="first") if len(df) else pd.Series(dtype=bool)
    df = df.loc[~duplicate_mask].copy() if len(df) else df
    write_csv(df, output)
    add_column_profile(profiles, dataset, str(path.relative_to(SOURCE)), df)
    log.append(
        {
            "dataset": dataset,
            "source_file": str(path.relative_to(SOURCE)),
            "raw_rows": len(rows),
            "exact_duplicates_removed": int(duplicate_mask.sum()) if len(duplicate_mask) else 0,
            "structurally_invalid_rows_removed": 0,
            "output_rows": len(df),
            "quality_flag": "not_applicable",
            "cleaning_actions": "flatten GeoJSON properties; retain complete geometry as compact JSON text in geometry_json",
        }
    )


def dictionary_json_to_csv(path: Path, output: Path, dataset: str, log: list[dict], profiles: list[dict]) -> None:
    obj = json.loads(path.read_text(encoding="utf-8"))
    rows = [{"source_key": key, **(value if isinstance(value, dict) else {"value": value})} for key, value in obj.items()]
    df = pd.DataFrame(rows)
    write_csv(df, output)
    add_column_profile(profiles, dataset, str(path.relative_to(SOURCE)), df)
    log.append(
        {
            "dataset": dataset,
            "source_file": str(path.relative_to(SOURCE)),
            "raw_rows": len(rows),
            "exact_duplicates_removed": 0,
            "structurally_invalid_rows_removed": 0,
            "output_rows": len(df),
            "quality_flag": "not_applicable",
            "cleaning_actions": "flatten keyed JSON object to CSV while preserving the original key",
        }
    )


def gpkg_to_csv(path: Path, output_dir: Path, log: list[dict], profiles: list[dict]) -> None:
    connection = sqlite3.connect(path)
    layers = pd.read_sql_query("SELECT table_name FROM gpkg_contents", connection)["table_name"].tolist()
    geometry_columns = {
        row[0]: row[1]
        for row in connection.execute("SELECT table_name, column_name FROM gpkg_geometry_columns").fetchall()
    }
    for layer in layers:
        geometry_column = geometry_columns.get(layer)
        columns = [row[1] for row in connection.execute(f'PRAGMA table_info("{layer}")').fetchall()]
        expressions = []
        for column in columns:
            quoted = column.replace('"', '""')
            if column == geometry_column:
                expressions.append(f'hex("{quoted}") AS geometry_gpkg_hex')
            else:
                expressions.append(f'"{quoted}"')
        df = pd.read_sql_query(f'SELECT {", ".join(expressions)} FROM "{layer}"', connection)
        output = output_dir / f"{layer}.csv"
        write_csv(df, output)
        add_column_profile(profiles, "census_geometry", f"{path.name}:{layer}", df)
        log.append(
            {
                "dataset": "census_geometry",
                "source_file": f"{path.relative_to(SOURCE)}:{layer}",
                "raw_rows": len(df),
                "exact_duplicates_removed": 0,
                "structurally_invalid_rows_removed": 0,
                "output_rows": len(df),
                "quality_flag": "not_applicable",
                "cleaning_actions": "export GeoPackage attributes to CSV; preserve GeoPackage geometry bytes losslessly as hexadecimal geometry_gpkg_hex",
            }
        )
    connection.close()


def clean_reference_data(log: list[dict], profiles: list[dict]) -> None:
    for path in sorted((SOURCE / "GTFS_data").glob("*/*.txt")):
        output = TARGET / "GTFS_data" / path.parent.name / f"{path.stem}.csv"
        clean_delimited_file(path, output, "gtfs", log, profiles)

    auxiliary = SOURCE / "auxiliar_data"
    clean_delimited_file(
        auxiliary / "meteorological_data.csv",
        TARGET / "auxiliar_data" / "meteorological_data.csv",
        "auxiliary_weather",
        log,
        profiles,
    )
    for filename in ["line_routes.json", "stops.json", "stops_integration_metropolitan.json"]:
        geojson_to_csv(
            auxiliary / filename,
            TARGET / "auxiliar_data" / f"{Path(filename).stem}.csv",
            "auxiliary_geography",
            log,
            profiles,
        )
    dictionary_json_to_csv(
        auxiliary / "stops_integration_city.json",
        TARGET / "auxiliar_data" / "stops_integration_city.csv",
        "auxiliary_geography",
        log,
        profiles,
    )

    for path in sorted((SOURCE / "social_data").glob("*.csv")):
        clean_delimited_file(path, TARGET / "social_data" / path.name, "social", log, profiles)

    census_root = SOURCE / "ibge_census_data_2022"
    for path in sorted(census_root.rglob("*.csv")):
        relative = path.relative_to(census_root)
        clean_delimited_file(path, TARGET / "ibge_census_data_2022" / relative, "census", log, profiles)
    gpkg_to_csv(
        census_root / "niteroi_census_borders.gpkg",
        TARGET / "ibge_census_data_2022" / "geometry",
        log,
        profiles,
    )


def original_inventory() -> pd.DataFrame:
    rows = []
    for path in sorted(SOURCE.rglob("*")):
        if path.is_file():
            rows.append(
                {
                    "relative_path": str(path.relative_to(SOURCE)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                    "extension": path.suffix.lower(),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Missing source directory: {SOURCE}")
    TARGET.mkdir(parents=True, exist_ok=True)
    log: list[dict] = []
    profiles: list[dict] = []
    clean_mobility(log, profiles)
    clean_ticket(log, profiles)
    clean_reference_data(log, profiles)
    quality_dir = TARGET / "quality_reports"
    write_csv(pd.DataFrame(log), quality_dir / "cleaning_log.csv")
    write_csv(pd.DataFrame(profiles), quality_dir / "column_profile.csv")
    write_csv(original_inventory(), quality_dir / "original_file_manifest.csv")
    issues = pd.DataFrame(
        [
            {"dataset": "mobility", "date": date, "issue": issue, "action": "retained_and_flagged"}
            for date, issue in DATE_QUALITY.items()
        ]
        + [
            {"dataset": "mobility", "date": "2026-03-18", "issue": "source_file_unavailable", "action": "not_present"},
            {"dataset": "mobility", "date": "2026-03-19", "issue": "source_file_unavailable", "action": "not_present"},
            {"dataset": "ticket", "date": "all", "issue": "reported_clock_requires_plus_3h_candidate_for_cross_system_alignment", "action": "original_preserved_candidate_column_added_not_confirmed"},
        ]
    )
    write_csv(issues, quality_dir / "known_quality_issues.csv")
    totals = pd.DataFrame(log).groupby("dataset", as_index=False).agg(
        raw_rows=("raw_rows", "sum"),
        exact_duplicates_removed=("exact_duplicates_removed", "sum"),
        structurally_invalid_rows_removed=("structurally_invalid_rows_removed", "sum"),
        output_rows=("output_rows", "sum"),
        files=("source_file", "nunique"),
    )
    write_csv(totals, quality_dir / "cleaning_summary.csv")
    print(totals.to_string(index=False))


if __name__ == "__main__":
    main()
