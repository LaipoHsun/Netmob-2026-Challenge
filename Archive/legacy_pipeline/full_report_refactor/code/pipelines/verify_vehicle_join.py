"""Verify the claim that telemetry and ticketing share no vehicle key.

This single fact fixes the identification scope of the whole study -- it is why
demand can only be defined at the line-direction-hour level and why no
vehicle-level dwell mechanism can be tested -- so it is worth being able to
re-check it in one command, independently of the Phase 1 pipeline.

Run:  python3 pipelines/verify_vehicle_join.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def distinct(folder: str, column: str) -> set[str]:
    """Every distinct non-null value of `column` across all CSVs in `folder`."""
    vals: set[str] = set()
    files = sorted((DATA / folder).glob("*.csv"))
    if not files:
        sys.exit(f"no CSVs found in {DATA / folder}")
    for path in files:
        for chunk in pd.read_csv(path, usecols=[column], dtype=str, chunksize=500_000):
            # Same normalisation as the Phase 1 pipeline: strip whitespace and a
            # trailing ".0" left by float round-tripping, so a purely cosmetic
            # difference could not masquerade as a genuine mismatch.
            s = chunk[column].dropna().str.strip()
            s = s.str.replace(r"\.0$", "", regex=True)
            vals.update(s.unique().tolist())
    return vals


def describe(name: str, vals: set[str]) -> None:
    numeric = sorted(int(v) for v in vals if v.isdigit())
    lengths = Counter(len(v) for v in vals)
    print(f"  {name}")
    print(f"    distinct values : {len(vals):,}")
    print(f"    all numeric     : {all(v.isdigit() for v in vals)}")
    print(f"    digit lengths   : {dict(sorted(lengths.items()))}")
    if numeric:
        print(f"    numeric range   : {numeric[0]:,} .. {numeric[-1]:,}")
        print(f"    first five      : {numeric[:5]}")


def main() -> None:
    print("Reading telemetry ids and ticketing vehicle numbers "
          "(this touches every daily CSV; ~1 min)\n")
    mob = distinct("mobility_data", "id")
    tic = distinct("ticket_data", "vehicle_number")

    describe("mobility_data/*.csv  ->  column `id`", mob)
    print()
    describe("ticket_data/*.csv    ->  column `vehicle_number`", tic)

    overlap = mob & tic
    print("\n  INTERSECTION")
    print(f"    shared values   : {len(overlap)}")
    if overlap:
        print(f"    examples        : {sorted(overlap)[:10]}")

    mn = sorted(int(v) for v in mob if v.isdigit())
    tn = sorted(int(v) for v in tic if v.isdigit())
    if mn and tn:
        disjoint = mn[0] > tn[-1] or tn[0] > mn[-1]
        print(f"    numeric ranges disjoint : {disjoint}")
        if disjoint:
            print("      -> the two id spaces do not even overlap numerically, so no")
            print("         formatting fix or fuzzy match could join them.")

    print("\n  WHAT THE OFFICIAL FIELD DOCUMENTATION SAYS")
    print("    README_Mobility.md : `id` = 'Unique vehicle identifier "
          "(hardware/transponder ID)'")
    print("    README_Ticket.md   : `vehicle_number` = 'Unique bus ID'")
    print("    Different identifier systems by definition. Neither README documents a")
    print("    correspondence, and the release contains no crosswalk table.")

    # Do the two feeds even describe the same system? Not documented anywhere, so
    # check it rather than assume it.
    print("\n  DO THE TWO FEEDS COVER THE SAME SYSTEM?")
    mob_lines: set[str] = set()
    for path in sorted((DATA / "mobility_data").glob("*.csv")):
        for chunk in pd.read_csv(path, usecols=["lineId"], dtype=str, chunksize=500_000):
            mob_lines.update(chunk["lineId"].dropna().str.strip().unique())
    tick = pd.concat(
        [pd.read_csv(p, usecols=["route_name", "company_number"], dtype=str)
         for p in sorted((DATA / "ticket_data").glob("*.csv"))],
        ignore_index=True)
    tick["route_name"] = tick["route_name"].str.strip()
    tic_lines = set(tick["route_name"].dropna().unique())
    print(f"    route labels: mobility {len(mob_lines)}, ticketing {len(tic_lines)}, "
          f"shared {len(mob_lines & tic_lines)}")
    print(f"    operators in ticketing: {tick['company_number'].nunique()}")
    for comp, grp in tick.groupby("company_number"):
        routes = set(grp["route_name"].dropna().unique())
        cov = len(routes & mob_lines)
        print(f"      company {comp:>3}: {cov}/{len(routes)} routes have telemetry")
    print("    Labels differ cosmetically between feeds (24 vs 24.0, 3 vs 03, 62B vs")
    print("    62), so the demand join uses a normalised code with an alias table.")

    print("\n  COLUMNS THE TWO FEEDS DO SHARE")
    m_cols = pd.read_csv(next((DATA / 'mobility_data').glob('*.csv')), nrows=0).columns
    t_cols = pd.read_csv(next((DATA / 'ticket_data').glob('*.csv')), nrows=0).columns
    print(f"    mobility : {list(m_cols)}")
    print(f"    ticketing: {list(t_cols)}")
    print("    The only usable link is the route/line label (lineName vs route_name),")
    print("    which is why demand is defined per line-direction-hour.")

    verdict = "CONFIRMED: no vehicle-level join is possible" if not overlap \
        else "OVERLAP FOUND -- the paper's claim would need revising"
    print(f"\n  {verdict}")


if __name__ == "__main__":
    main()
