# Data cleaning code

This directory contains the only canonical program allowed to read `data_original/`.

## Run

```bash
python Data_cleaning_code/run_cleaning.py
```

The script writes only CSV files under `data_after_cleaning/`.

## Cleaning policy

- Preserve the downloaded challenge release unchanged under `data_original/`.
- Remove exact duplicate rows.
- Remove only structurally unusable rows: invalid timestamps, missing vehicle keys, or invalid coordinates.
- Preserve incomplete Mobility dates and label them with `date_quality_flag`.
- Preserve Ticket timestamps exactly in `transaction_date_original`.
- Add `transaction_time_brt_reported` and an explicitly unconfirmed `transaction_time_candidate_shifted_3h`; never overwrite the source timestamp.
- Treat identifiers and route labels as strings. Route normalization trims whitespace and changes only integer-like `.0` suffixes.
- Convert GTFS, auxiliary, social, census, GeoJSON, and GeoPackage contents to CSV. GeoJSON geometry is retained as JSON text; GeoPackage geometry bytes are retained losslessly as hexadecimal text.

## Quality outputs

`data_after_cleaning/quality_reports/` contains:

- `cleaning_log.csv`: row-level decisions summarized by source file.
- `cleaning_summary.csv`: totals by dataset.
- `column_profile.csv`: output dtypes and missingness.
- `known_quality_issues.csv`: incomplete dates and the unresolved Ticket clock issue.
- `original_file_manifest.csv`: byte size and SHA-256 checksum of every original file.

No downstream processing or scientific analysis belongs in this directory.
