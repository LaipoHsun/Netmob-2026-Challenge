"""Contrast estimators for group comparisons that are confounded by fleet size.

The submitted abstract compares weekend against weekday cells marginally. Fleet
size N differs sharply between the two (4.5 vs 8.0), and the corrected
estimators still carry an N gradient, so the marginal contrast and the
conditional-on-N contrast answer different questions:

    marginal     total weekend effect, including the part that runs through
                 running fewer buses
    N-matched    weekend effect at a fixed fleet size, i.e. with the fleet-size
                 channel blocked

Neither is wrong. Reporting only one of them is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["marginal_contrast", "n_matched_contrast", "bootstrap_contrast", "three_part_table"]


def marginal_contrast(df: pd.DataFrame, value: str, group: str, a: str, b: str) -> float:
    """mean(value | group == b) - mean(value | group == a)."""
    ga = df.loc[df[group] == a, value].dropna()
    gb = df.loc[df[group] == b, value].dropna()
    if len(ga) == 0 or len(gb) == 0:
        return np.nan
    return float(gb.mean() - ga.mean())


def n_matched_contrast(
    df: pd.DataFrame,
    value: str,
    group: str,
    a: str,
    b: str,
    strat: str = "N",
    min_per_cell: int = 1,
) -> float:
    """Exact-N stratified contrast, weighted by stratum size.

    Each N stratum contributes its own within-stratum difference; strata are
    weighted by how many cells they contain, so the estimand is the weekend
    effect averaged over the observed fleet-size distribution.
    """
    diffs, weights = [], []
    for _, grp in df.groupby(strat):
        ga = grp.loc[grp[group] == a, value].dropna()
        gb = grp.loc[grp[group] == b, value].dropna()
        if len(ga) < min_per_cell or len(gb) < min_per_cell:
            continue
        diffs.append(gb.mean() - ga.mean())
        weights.append(len(grp))
    if not diffs:
        return np.nan
    return float(np.average(diffs, weights=weights))


def bootstrap_contrast(
    df: pd.DataFrame,
    value: str,
    group: str,
    a: str,
    b: str,
    kind: str = "marginal",
    strat: str = "N",
    draws: int = 2000,
    seed: int = 20260814,
) -> dict:
    """Percentile bootstrap over cells for either contrast."""
    fn = marginal_contrast if kind == "marginal" else n_matched_contrast
    kwargs = {} if kind == "marginal" else {"strat": strat}

    point = fn(df, value, group, a, b, **kwargs)
    rng = np.random.default_rng(seed)
    n = len(df)
    boots = np.empty(draws)
    for i in range(draws):
        boots[i] = fn(df.iloc[rng.integers(0, n, n)], value, group, a, b, **kwargs)

    boots = boots[np.isfinite(boots)]
    return {
        "estimator": value,
        "contrast": kind,
        "point": point,
        "ci_low": float(np.percentile(boots, 2.5)),
        "ci_high": float(np.percentile(boots, 97.5)),
        "boot_draws": int(boots.size),
    }


def three_part_table(
    df: pd.DataFrame,
    estimators: list[str],
    group: str,
    a: str,
    b: str,
    strat: str = "N",
    draws: int = 2000,
    seed: int = 20260814,
) -> pd.DataFrame:
    """Marginal and N-matched contrasts with CIs, for every estimator."""
    rows = []
    for est in estimators:
        if est not in df.columns:
            continue
        for kind in ("marginal", "n_matched"):
            rows.append(bootstrap_contrast(
                df, est, group, a, b, kind=kind, strat=strat, draws=draws, seed=seed
            ))
    return pd.DataFrame(rows)
