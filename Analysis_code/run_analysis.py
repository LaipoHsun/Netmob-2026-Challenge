"""Create reproducible descriptive tables and figures from final processed CSVs.

Data contract
-------------
Reads only:  data_after_all_processed/
Writes only: Output/Table/ and Output/Figure/

The two Ticket clock bases are never silently combined: every comparison is either
stratified by ``clock_basis`` or uses an explicitly named basis.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "data_after_all_processed"
TABLES = PROJECT / "Output" / "Table"
FIGURES = PROJECT / "Output" / "Figure"


def save_table(frame: pd.DataFrame, name: str) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    frame.to_csv(TABLES / name, index=False, encoding="utf-8", lineterminator="\n")


def save_figure(fig: plt.Figure, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / name, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    summary = pd.read_csv(SOURCE / "processing_summary.csv")
    routes = pd.read_csv(SOURCE / "route" / "route_correspondence.csv", dtype={"line_code": "string"})
    selection = pd.read_csv(SOURCE / "route" / "route_selection.csv", dtype={"line_code": "string"})
    cells = pd.read_csv(SOURCE / "analysis_cells" / "line_direction_hour_features.csv", parse_dates=["hour"])

    save_table(summary, "processed_dataset_inventory.csv")

    route_table = routes.merge(
        selection[["line_code", "rank_by_boardings", "selected_for_phase_features"]],
        on="line_code",
        how="left",
    ).sort_values(["ticket_transactions", "mobility_observations"], ascending=False)
    save_table(route_table, "route_coverage_and_selection.csv")

    cells["day_type"] = np.where(cells["hour"].dt.dayofweek >= 5, "weekend", "weekday")
    clock_summary = (
        cells.groupby(["clock_basis", "day_type"], dropna=False)
        .agg(
            analysis_cells=("r1_arc", "size"),
            cells_with_ticket_demand=("boardings", "count"),
            total_boardings=("boardings", "sum"),
            median_boardings=("boardings", "median"),
            median_buses=("N", "median"),
            median_r1_arc=("r1_arc", "median"),
            median_r_excess_arc=("r_excess_arc", "median"),
            median_load=("lambda_boardings_per_bus", "median"),
        )
        .reset_index()
    )
    save_table(clock_summary, "clock_basis_descriptive_comparison.csv")

    # Raw order naturally depends on N; show the finite-size null beside observations.
    by_n = (
        cells.groupby("N", as_index=False)
        .agg(observed_median_r1=("r1_arc", "median"), null_mean=("null_mean", "first"), cells=("r1_arc", "size"))
        .sort_values("N")
    )
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    ax.plot(by_n.N, by_n.observed_median_r1, marker="o", ms=3, label="Observed median $r_1$")
    ax.plot(by_n.N, by_n.null_mean, color="black", linestyle="--", label="Same-$N$ null mean")
    ax.set(xlabel="Active buses in route-direction-hour (N)", ylabel="Circular order", title="Observed order and finite-size null")
    ax.legend(frameon=False)
    save_figure(fig, "processed_order_vs_finite_size_null.png")

    # Compare clocks explicitly; each cell remains separately labelled by clock basis.
    valid = cells.dropna(subset=["lambda_boardings_per_bus", "r_excess_arc"])
    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for basis, group in valid.groupby("clock_basis"):
        sample = group.sample(min(10_000, len(group)), random_state=2602)
        ax.scatter(sample.lambda_boardings_per_bus, sample.r_excess_arc, s=8, alpha=.20, label=basis)
    ax.set(xlabel="Boardings per active GPS bus", ylabel="Finite-size corrected order", title="Demand load and corrected bus order")
    ax.legend(frameon=False)
    save_figure(fig, "processed_load_vs_corrected_order_by_clock.png")

    selected = selection[selection.selected_for_phase_features.fillna(False)].head(20)
    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    plot = selected.sort_values("boardings")
    ax.barh(plot.line_code.astype(str), plot.boardings)
    ax.set(xlabel="Ticket transactions", ylabel="Route", title="Routes selected for phase feature construction")
    save_figure(fig, "processed_selected_routes.png")

    print("Wrote 3 tables and 3 figures from data_after_all_processed only.")


if __name__ == "__main__":
    main()
