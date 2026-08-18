"""Five finite-size-aware synchronization estimators, plus Daido harmonics.

All five answer the same question -- how far above its same-N chance level does
this cell sit -- with different bias and variance properties:

    r_excess   (r1 - E_null) / (1 - E_null)     first-moment debiasing, the
                                                estimator used in the submitted
                                                abstract
    z          (r1 - E_null) / sd_null          studentised, but sd_null itself
                                                scales like 1/sqrt(N)
    ppc        (N r1^2 - 1) / (N - 1)           Vinck et al. 2010; analytically
                                                unbiased for the squared
                                                population resultant, no Monte
                                                Carlo needed
    rayleigh_z N r1^2                           the classical directional
                                                statistic; null mean is exactly 1
    pit_probit Phi^-1(F_null(r1 | N))           probability integral transform;
                                                exactly N-free under the null

r_excess and z need a null table; ppc and rayleigh_z are closed form; pit_probit
needs the null CDF.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "ESTIMATORS",
    "r_excess",
    "z_score",
    "ppc",
    "rayleigh_z",
    "pit_probit",
    "daido_order",
    "add_all_estimators",
]

ESTIMATORS = ("r_excess", "z", "ppc", "rayleigh_z", "pit_probit")


def r_excess(r1: np.ndarray, null_mean: np.ndarray) -> np.ndarray:
    """(r1 - E_null) / (1 - E_null). Undefined at N = 1 where E_null = 1."""
    denom = 1.0 - np.asarray(null_mean, dtype=float)
    out = np.full_like(np.asarray(r1, dtype=float), np.nan)
    ok = denom > 1e-12
    out[ok] = (np.asarray(r1, dtype=float)[ok] - np.asarray(null_mean, dtype=float)[ok]) / denom[ok]
    return out


def z_score(r1: np.ndarray, null_mean: np.ndarray, null_sd: np.ndarray) -> np.ndarray:
    sd = np.asarray(null_sd, dtype=float)
    out = np.full_like(np.asarray(r1, dtype=float), np.nan)
    ok = sd > 1e-12
    out[ok] = (np.asarray(r1, dtype=float)[ok] - np.asarray(null_mean, dtype=float)[ok]) / sd[ok]
    return out


def ppc(r1: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Pairwise phase consistency: (N r1^2 - 1) / (N - 1).

    Equals the mean of cos(phi_j - phi_k) over distinct pairs, and is unbiased
    for the squared population resultant under iid sampling -- for any true
    value, not only under the uniform null. Matches FieldTrip's
    ft_connectivity_ppc, which computes (|sum|^2 - n) / (n (n-1)).
    """
    r1 = np.asarray(r1, dtype=float)
    n = np.asarray(n, dtype=float)
    out = np.full_like(r1, np.nan)
    ok = n > 1
    out[ok] = (n[ok] * r1[ok] ** 2 - 1.0) / (n[ok] - 1.0)
    return out


def rayleigh_z(r1: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Rayleigh test statistic N r1^2; null expectation is exactly 1."""
    return np.asarray(n, dtype=float) * np.asarray(r1, dtype=float) ** 2


def pit_probit(
    r1: np.ndarray,
    n: np.ndarray,
    null_samples: dict[int, np.ndarray],
    clip: float = 1e-4,
) -> np.ndarray:
    """Probit of the null CDF evaluated at the observed r1.

    Under the null this is exactly standard normal for every N, so it removes
    all N-dependence by construction -- the strictest available control.
    Ties are broken with the mid-rank so the transform stays unbiased.
    """
    r1 = np.asarray(r1, dtype=float)
    n = np.asarray(n, dtype=float)
    out = np.full_like(r1, np.nan)
    for n_val in np.unique(n[np.isfinite(n)]):
        draws = null_samples.get(int(n_val))
        if draws is None:
            continue
        mask = n == n_val
        below = np.searchsorted(draws, r1[mask], side="left")
        equal = np.searchsorted(draws, r1[mask], side="right") - below
        u = (below + 0.5 * equal) / len(draws)
        out[mask] = stats.norm.ppf(np.clip(u, clip, 1.0 - clip))
    return out


def daido_order(phases: np.ndarray, m: int) -> float:
    """|Z_m| = |mean(exp(i m phi))|, the m-th Daido order parameter.

    m = 1 is the Kuramoto order parameter and only detects a single cluster;
    m = 2, 3 detect two- and three-cluster bunching that r1 misses entirely
    (Daido 1990).
    """
    phases = np.asarray(phases, dtype=float)
    phases = phases[np.isfinite(phases)]
    if phases.size == 0:
        return np.nan
    return float(np.abs(np.mean(np.exp(1j * m * phases))))


def add_all_estimators(
    df: pd.DataFrame,
    null_table: pd.DataFrame,
    null_samples: dict[int, np.ndarray] | None = None,
    r_col: str = "r1",
    n_col: str = "N",
    suffix: str = "",
) -> pd.DataFrame:
    """Attach all five estimators to a frame of cell-level (r, N) observations.

    null_table must carry columns N, null_mean, null_sd. null_samples maps N to
    a *sorted* array of null draws and is required only for pit_probit.
    """
    out = df.merge(null_table[["N", "null_mean", "null_sd"]], left_on=n_col, right_on="N",
                   how="left", suffixes=("", "_nulltab"))
    r = out[r_col].to_numpy(float)
    n = out[n_col].to_numpy(float)
    mu = out["null_mean"].to_numpy(float)
    sd = out["null_sd"].to_numpy(float)

    out[f"r_excess{suffix}"] = r_excess(r, mu)
    out[f"z{suffix}"] = z_score(r, mu, sd)
    out[f"ppc{suffix}"] = ppc(r, n)
    out[f"rayleigh_z{suffix}"] = rayleigh_z(r, n)
    if null_samples is not None:
        out[f"pit_probit{suffix}"] = pit_probit(r, n, null_samples)
    return out.drop(columns=[c for c in ("N_nulltab",) if c in out.columns])
