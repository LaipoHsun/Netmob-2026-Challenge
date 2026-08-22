"""Null models for the finite-size correction.

The submitted abstract uses one null: N iid phases uniform on the circle. That
null encodes "perfect service = buses evenly spread in arc length", which is not
what perfect service means. A route with a congested stretch accumulates buses
there even at perfectly regular headways, because arc-length phase advances
slowly where the bus moves slowly. Three alternatives relax that assumption:

    uniform     phi ~ U(0, 2 pi)                    the submitted null
    speed       phi drawn uniform in *travel time*, mapped to arc-length phase;
                equivalently arc-length density proportional to 1/v(s). This is
                the null implied by evenly spaced headways in time.
    occupancy   phi resampled from the observed marginal distribution of
                arc-length position for that line-direction; absorbs route
                geometry and dwell structure wholesale
    permutation phases resampled from the same line-direction at *other* hours;
                preserves operational structure, destroys simultaneity

All four reduce to: draw N phases from a reference distribution, take
|mean(exp(i phi))|. Only the reference distribution changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["NullTable", "uniform_null", "null_from_samples", "grouped_null", "build_null_bank"]

DEFAULT_DRAWS = 20_000
DEFAULT_SEED = 20260814


@dataclass
class NullTable:
    """Per-N null summary plus the sorted draws needed for the PIT transform."""

    table: pd.DataFrame                 # columns: N, draws, null_mean, null_sd, q025, q975
    samples: dict[int, np.ndarray]      # N -> sorted r1 draws
    kind: str

    def to_frame(self) -> pd.DataFrame:
        out = self.table.copy()
        out["null_kind"] = self.kind
        return out


def _order_parameter(phases: np.ndarray, m: int = 1) -> np.ndarray:
    """|mean(exp(i m phi))| along the last axis."""
    return np.abs(np.mean(np.exp(1j * m * phases), axis=-1))


def uniform_null(
    max_n: int,
    draws: int = DEFAULT_DRAWS,
    seed: int = DEFAULT_SEED,
    m: int = 1,
) -> NullTable:
    """iid uniform phases. Reproduces the submitted abstract's null."""
    rng = np.random.default_rng(seed)
    rows, samples = [], {}
    for n in range(1, max_n + 1):
        phi = rng.uniform(0.0, 2.0 * np.pi, size=(draws, n))
        r = np.sort(_order_parameter(phi, m))
        samples[n] = r
        rows.append(_summarise(n, draws, r))
    return NullTable(pd.DataFrame(rows), samples, kind=f"uniform_m{m}")


def null_from_samples(
    reference_phases: np.ndarray,
    max_n: int,
    draws: int = DEFAULT_DRAWS,
    seed: int = DEFAULT_SEED,
    m: int = 1,
    kind: str = "empirical",
) -> NullTable:
    """Draw N phases with replacement from an observed pool of phases.

    Serves the occupancy and permutation nulls, and the speed null once the
    travel-time map has been applied to the reference positions.
    """
    reference_phases = np.asarray(reference_phases, dtype=float)
    reference_phases = reference_phases[np.isfinite(reference_phases)]
    if reference_phases.size < 20:
        raise ValueError("reference pool too small for a stable null")

    rng = np.random.default_rng(seed)
    rows, samples = [], {}
    for n in range(1, max_n + 1):
        idx = rng.integers(0, reference_phases.size, size=(draws, n))
        r = np.sort(_order_parameter(reference_phases[idx], m))
        samples[n] = r
        rows.append(_summarise(n, draws, r))
    return NullTable(pd.DataFrame(rows), samples, kind=kind)


def _summarise(n: int, draws: int, r_sorted: np.ndarray) -> dict:
    return {
        "N": n,
        "draws": draws,
        "null_mean": float(r_sorted.mean()),
        "null_sd": float(r_sorted.std(ddof=1)),
        "null_q025": float(np.quantile(r_sorted, 0.025)),
        "null_q975": float(np.quantile(r_sorted, 0.975)),
    }


def _rowwise_sample_without_replacement(
    rng: np.random.Generator,
    pool_size: int,
    draws: int,
    n: int,
) -> np.ndarray:
    """Vectorised independent size-n permutations from one index pool."""
    if n > pool_size:
        raise ValueError("cannot sample more distinct phases than the group pool contains")
    idx = np.empty((draws, n), dtype=np.int64)
    for column in range(n):
        candidate = rng.integers(0, pool_size, size=draws)
        if column:
            duplicate = np.any(candidate[:, None] == idx[:, :column], axis=1)
            while np.any(duplicate):
                candidate[duplicate] = rng.integers(0, pool_size, size=int(duplicate.sum()))
                duplicate = np.any(candidate[:, None] == idx[:, :column], axis=1)
        idx[:, column] = candidate
    return idx


def grouped_null(
    bus_hours: pd.DataFrame,
    max_n: int,
    group_cols: list[str],
    phase_col: str,
    draws: int = 4_000,
    seed: int = DEFAULT_SEED,
    m: int = 1,
    min_pool: int = 50,
    kind: str = "grouped",
) -> tuple[pd.DataFrame, dict[tuple, dict[int, np.ndarray]]]:
    """One null per group, each drawn from that group's own observed phases.

    Returns a long table keyed by the group columns plus N, and a nested dict of
    sorted draws for the PIT transform.
    """
    rows: list[dict] = []
    samples: dict[tuple, dict[int, np.ndarray]] = {}
    rng = np.random.default_rng(seed)

    for key, grp in bus_hours.groupby(group_cols, sort=False):
        pool = grp[phase_col].to_numpy(float)
        pool = pool[np.isfinite(pool)]
        if pool.size < min_pool:
            continue
        key = key if isinstance(key, tuple) else (key,)
        n_max = min(max_n, int(grp["N"].max()) if "N" in grp else max_n)
        samples[key] = {}
        # Draw one random permutation prefix per Monte Carlo replicate.  Every
        # prefix is itself a uniform sample without replacement, so all fleet
        # sizes share valid marginal nulls without repeating O(N^2) sampling.
        permutation_prefixes = _rowwise_sample_without_replacement(
            rng, pool.size, draws, n_max
        )
        for n in range(1, n_max + 1):
            # A route permutation draws distinct observed phase records within
            # each synthetic cell.  The archived implementation sampled with
            # replacement, which was an empirical bootstrap rather than the
            # permutation null described in the manuscript.
            idx = permutation_prefixes[:, :n]
            r = np.sort(_order_parameter(pool[idx], m))
            samples[key][n] = r
            row = dict(zip(group_cols, key))
            row.update(_summarise(n, draws, r))
            rows.append(row)

    return pd.DataFrame(rows), samples


def build_null_bank(
    bus_hours: pd.DataFrame,
    max_n: int,
    arc_phase_col: str = "phase_arc",
    time_phase_col: str = "phase_time",
    group_cols: tuple[str, ...] = ("line_code", "direction"),
    draws: int = DEFAULT_DRAWS,
    seed: int = DEFAULT_SEED,
    m: int = 1,
) -> dict[str, NullTable]:
    """Build the three pooled nulls from the per-bus phase table.

    They form a nested hierarchy of preserved structure:

        uniform    nothing preserved
        speed      route speed profile preserved (phases uniform in travel time,
                   expressed back in arc-length phase)
        occupancy  full observed arc-length density preserved

    The fourth, per-line-direction permutation null, is built by grouped_null
    because it needs a separate reference pool per route rather than one pooled
    distribution.
    """
    bank: dict[str, NullTable] = {}
    bank["uniform"] = uniform_null(max_n, draws=draws, seed=seed, m=m)

    arc = bus_hours[arc_phase_col].to_numpy(float)
    bank["occupancy"] = null_from_samples(
        arc, max_n, draws=draws, seed=seed + 1, m=m, kind="occupancy"
    )

    if time_phase_col in bus_hours.columns:
        rng = np.random.default_rng(seed + 2)
        pool = []
        for _, grp in bus_hours.groupby(list(group_cols)):
            arc_g = np.sort(grp[arc_phase_col].to_numpy(float))
            time_g = np.sort(grp[time_phase_col].to_numpy(float))
            if arc_g.size < 50:
                continue
            # Draw uniform in time-phase, then invert through this route's
            # empirical time->arc map. Sampling arc phases at time-uniform
            # quantiles is exactly sampling from the 1/v(s) density.
            u = rng.uniform(0.0, 2.0 * np.pi, size=arc_g.size)
            pool.append(np.interp(u, time_g, arc_g))
        if pool:
            bank["speed"] = null_from_samples(
                np.concatenate(pool), max_n, draws=draws, seed=seed + 3, m=m, kind="speed"
            )
    return bank
