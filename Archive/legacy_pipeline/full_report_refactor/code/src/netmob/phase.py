"""Phase definitions for the bus-as-oscillator mapping.

The submitted abstract sets phi = 2 pi s / L, with s arc length along the route
polyline. That choice violates the assumption the Kuramoto analogy rests on: a
free-running oscillator advances its phase at a constant rate. A bus does not
cover arc length at a constant rate -- it crawls through congested stretches and
sits at stops. Arc-length phase therefore piles up where the bus is slow, and a
perfectly regular service reads as partially synchronised.

The fix is to parametrise by expected travel time rather than distance:

    phi_time = 2 pi * T(s) / T_cycle

with T(s) the median cumulative travel time to arc position s. Under this map a
free-running bus advances phi at a constant rate by construction, so an evenly
spaced timetable maps to an even spread of phase -- which is what the uniform
null assumes.

T(s) is estimated from the dense cross-section crossing events: each vehicle run
contributes its crossing times at the 5%, 10%, ..., 95% sections, and the median
inter-section travel time defines the profile per line-direction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["travel_time_profile", "arc_to_time_phase", "attach_time_phase"]

TWO_PI = 2.0 * np.pi


def travel_time_profile(
    crossings: pd.DataFrame,
    group_cols: tuple[str, ...] = ("line_code", "direction"),
    section_col: str = "section_frac",
    time_col: str = "event_time",
    run_cols: tuple[str, ...] = ("vehicle_id", "service_date"),
    min_runs: int = 20,
) -> pd.DataFrame:
    """Median cumulative travel time as a function of arc fraction.

    Returns one row per (group, section_frac) with columns
    ``t_cum_s`` (median seconds from the first section) and ``t_frac``
    (that time normalised to [0, 1] across the route).

    Inter-section times are taken per vehicle run and then aggregated by median,
    which is robust to the layover and mid-route gaps that contaminate a naive
    first-to-last difference.
    """
    df = crossings.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    # A vehicle runs the route several times a day, so a run must be cut into
    # individual trips first: sort by time and start a new trip wherever the
    # section fraction drops, i.e. the bus has turned around at the terminal.
    keys = list(group_cols) + list(run_cols)
    df = df.sort_values(keys + [time_col])
    prev_in_run = df.groupby(keys)[section_col].shift()
    df["trip_id"] = (
        (df[section_col] < prev_in_run).fillna(False).astype(int).groupby(
            [df[c] for c in keys]
        ).cumsum()
    )

    trip_keys = keys + ["trip_id"]
    df["dt_s"] = df.groupby(trip_keys)[time_col].diff().dt.total_seconds()
    df["prev_section"] = df.groupby(trip_keys)[section_col].shift()

    # Only consecutive sections within one trip, and only physically plausible
    # segment times (a bus crossing two adjacent 10% sections in under a second
    # or over an hour is a matching artefact, not a trip).
    step = df[section_col] - df["prev_section"]
    ok = step.notna() & (step > 0) & (step < 0.25) & df["dt_s"].between(1.0, 3600.0)
    seg = df.loc[ok]

    prof = (
        seg.groupby(list(group_cols) + ["prev_section", section_col])["dt_s"]
        .agg(["median", "size"])
        .reset_index()
        .rename(columns={"median": "dt_median_s", "size": "n_runs"})
    )
    prof = prof[prof["n_runs"] >= min_runs]

    out = []
    for key, grp in prof.groupby(list(group_cols), sort=False):
        grp = grp.sort_values("prev_section")
        sections = np.concatenate([[grp["prev_section"].iloc[0]], grp[section_col].to_numpy()])
        cum = np.concatenate([[0.0], np.cumsum(grp["dt_median_s"].to_numpy())])
        if cum[-1] <= 0:
            continue
        key = key if isinstance(key, tuple) else (key,)
        frame = pd.DataFrame({
            **dict(zip(group_cols, key)),
            "section_frac": sections,
            "t_cum_s": cum,
            "t_frac": cum / cum[-1],
        })
        frame["route_cycle_s"] = cum[-1]
        out.append(frame)

    if not out:
        return pd.DataFrame(columns=list(group_cols) + ["section_frac", "t_cum_s", "t_frac"])
    return pd.concat(out, ignore_index=True)


def arc_to_time_phase(s_frac: np.ndarray, profile: pd.DataFrame) -> np.ndarray:
    """Map arc fraction to time phase for one line-direction.

    Linear interpolation on the measured profile, extrapolated flat outside the
    5%-95% section range where no crossings were observed.
    """
    s_frac = np.asarray(s_frac, dtype=float)
    xs = profile["section_frac"].to_numpy(float)
    ys = profile["t_frac"].to_numpy(float)
    order = np.argsort(xs)
    return TWO_PI * np.interp(s_frac, xs[order], ys[order])


def attach_time_phase(
    bus_hours: pd.DataFrame,
    profiles: pd.DataFrame,
    group_cols: tuple[str, ...] = ("line_code", "direction"),
    s_frac_col: str = "s_mid_frac",
    out_col: str = "phase_time",
) -> pd.DataFrame:
    """Add a time-parametrised phase column to the per-bus-hour table.

    Rows whose line-direction has no usable travel-time profile get NaN, so they
    drop out of the time-phase analysis rather than silently falling back to the
    arc-length definition.
    """
    df = bus_hours.copy()
    df[out_col] = np.nan
    prof_groups = dict(list(profiles.groupby(list(group_cols), sort=False)))

    for key, idx in df.groupby(list(group_cols), sort=False).groups.items():
        key = key if isinstance(key, tuple) else (key,)
        prof = prof_groups.get(key)
        if prof is None or len(prof) < 3:
            continue
        df.loc[idx, out_col] = arc_to_time_phase(df.loc[idx, s_frac_col].to_numpy(float), prof)
    return df
