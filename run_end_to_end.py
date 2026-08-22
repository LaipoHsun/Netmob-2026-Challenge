#!/usr/bin/env python3
"""Rebuild the corrected full-report analysis from raw data in one execution.

The working intermediates live in a temporary directory and disappear after a
successful run.  Only the data, result tables, and independent plots used by
the paper are published to ``output/``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


PACKAGE = Path(__file__).resolve().parent
STAGES = PACKAGE / "code" / "stages"
PIPELINES = PACKAGE / "code" / "pipelines"
DEFAULT_FINAL_OUTPUT = PACKAGE / "output"


DATA_FILES = [
    ("phase2", "phase2_hourly_observables.csv"),
    ("week1_2", "cells_scored_arc_uniform.csv"),
    ("week1_2", "bus_hours_with_phases.csv"),
    ("week3", "newell_potts_test.csv"),
    ("week3", "kuramoto_kc.csv"),
    ("week3", "scaling_collapse.csv"),
    ("week3", "scaling_collapse_interior.csv"),
    ("week3", "scaling_collapse_null.txt"),
    ("week3", "scaling_collapse_interior_null.txt"),
    ("phase4", "corridor_coupling_speed_controlled.csv"),
]

TABLE_FILES = [
    ("phase2", "line_selection.csv"),
    ("phase2", "mixed_model_r_excess_coefficients.csv"),
    ("phase2", "mixed_model_z_coefficients.csv"),
    ("phase2", "mixed_model_summary.json"),
    ("phase2", "transition_model_comparison.csv"),
    ("phase2", "piecewise_change_point_bootstrap.csv"),
    ("phase2", "weather_summary.json"),
    ("week1_2", "weekend_contrasts.csv"),
    ("week1_2", "null_tables.csv"),
    ("week1_2", "residual_n_dependence.csv"),
    ("week1_2", "daido_harmonics.csv"),
    ("week1_2", "terminal_sensitivity.csv"),
    ("week1_2", "terminal_occupancy.csv"),
    ("week1_2", "sweep_summary.json"),
    ("phase3", "criticality_susceptibility_by_N_lambda.csv"),
    ("phase3", "criticality_apparent_lambda_by_N.csv"),
    ("phase3", "criticality_mixture_bimodality.csv"),
    ("phase3", "criticality_summary.json"),
    ("phase3", "mediation_summary.json"),
    ("phase3", "spatial_amplification_by_lambda_tertile.csv"),
    ("phase3", "spatial_amplification_summary.json"),
    ("phase3", "corridor_within_pair_summary.json"),
    ("phase4", "corridor_coupling_gate_summary.json"),
    ("phase4", "amplification_driver_coefficients.csv"),
    ("phase4", "amplification_driver_summary.json"),
    ("week3", "synchronization_by_zone.csv"),
    ("week3", "load_effect_by_zone.csv"),
    ("week3", "terminal_fraction_control.csv"),
    ("week3", "route_position_summary.json"),
    ("week3", "theory_tests_summary.json"),
]


def execute(label: str, command: list[str], env: dict[str, str]) -> None:
    print(f"\n=== {label} ===", flush=True)
    subprocess.run(command, cwd=PACKAGE, env=env, check=True)


def copy_selected(outputs: Path, destination: Path) -> None:
    data_dir = destination / "data"
    table_dir = destination / "tables"
    figure_dir = destination / "figures"
    data_dir.mkdir(parents=True)
    table_dir.mkdir(parents=True)
    figure_dir.mkdir(parents=True)
    for stage, filename in DATA_FILES:
        source = outputs / stage / filename
        if not source.is_file():
            raise FileNotFoundError(f"required paper data was not produced: {source}")
        shutil.copy2(source, data_dir / filename)
    for stage, filename in TABLE_FILES:
        source = outputs / stage / filename
        if not source.is_file():
            raise FileNotFoundError(f"required paper table was not produced: {source}")
        shutil.copy2(source, table_dir / filename)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PACKAGE / "data_original",
        help="Raw-data directory with the same layout as the released dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_FINAL_OUTPUT,
        help="Directory that will receive the recomputed paper data, tables, and figures",
    )
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    raw = args.data_root.resolve()
    if not raw.is_dir():
        raise SystemExit(f"raw-data directory does not exist: {raw}")
    final_output = args.output_dir.resolve()
    if final_output == raw or raw in final_output.parents:
        raise SystemExit("output directory must not be the raw-data directory or one of its children")
    if final_output == PACKAGE:
        raise SystemExit("output directory must not be the package root")
    python = str(Path(args.python).resolve())

    with tempfile.TemporaryDirectory(prefix=".full_report_work_", dir=PACKAGE) as temp_name:
        work = Path(temp_name)
        outputs = work / "outputs"
        phase2 = outputs / "phase2"
        phase3 = outputs / "phase3"
        phase4 = outputs / "phase4"
        week1_2 = outputs / "week1_2"
        week3 = outputs / "week3"
        publish = work / "publish"

        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"
        env["MPLCONFIGDIR"] = str(work / "matplotlib")
        env["PYTHONPYCACHEPREFIX"] = str(work / "pycache")

        geometry = ["--max-static-part-gap-m", "250"]
        execute("Phase 2: raw GPS/tickets to corrected hourly observables", [
            python, str(STAGES / "phase2_analysis.py"),
            "--data-root", str(raw), "--output-dir", str(phase2), *geometry,
        ], env)
        execute("Phase 3: mechanisms, criticality, mediation, spatial and corridor tests", [
            python, str(STAGES / "phase3_analysis.py"),
            "--data-root", str(raw), "--phase2-dir", str(phase2),
            "--output-dir", str(phase3), *geometry,
        ], env)
        execute("Phase 4: common-speed corridor gate and amplification drivers", [
            python, str(STAGES / "phase4_analysis.py"),
            "--data-root", str(raw), "--phase2-dir", str(phase2),
            "--phase3-dir", str(phase3), "--output-dir", str(phase4), *geometry,
        ], env)
        execute("Rebuild arc/time cells", [
            python, str(PIPELINES / "build_cells.py"),
            "--phase2-dir", str(phase2), "--phase3-dir", str(phase3),
            "--out-dir", str(week1_2),
        ], env)
        execute("Estimator, null-model, and weekend sweep", [
            python, str(PIPELINES / "sweep.py"), "--data-dir", str(week1_2),
        ], env)
        execute("Terminal trimming sensitivity", [
            python, str(PIPELINES / "terminal_sensitivity.py"),
            "--data-dir", str(week1_2),
        ], env)
        execute("Route-position analysis", [
            python, str(PIPELINES / "route_position.py"),
            "--outputs-root", str(outputs), "--out-dir", str(week3),
        ], env)
        execute("Newell--Potts, Kuramoto, and scaling tests", [
            python, str(PIPELINES / "theory_tests.py"),
            "--outputs-root", str(outputs), "--out-dir", str(week3),
        ], env)

        copy_selected(outputs, publish)
        execute("Eighteen independent paper plots", [
            python, str(PIPELINES / "make_figures.py"),
            "--data-dir", str(publish / "data"),
            "--table-dir", str(publish / "tables"),
            "--figure-dir", str(publish / "figures"),
        ], env)

        final_output.parent.mkdir(parents=True, exist_ok=True)
        if final_output.exists():
            shutil.rmtree(final_output)
        shutil.move(str(publish), str(final_output))

    print(f"\nCompleted. Paper data, tables, and independent plots: {final_output}")


if __name__ == "__main__":
    main()
