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
    values = pd.to_numeric(df[value], errors="coerce").to_numpy(float)
    groups = df[group].to_numpy()
    batch_size = 100
    if kind == "n_matched":
        strata, _ = pd.factorize(df[strat], sort=True)
        n_strata = int(strata.max()) + 1

    # The original implementation rebuilt and grouped a pandas DataFrame for
    # every draw.  These batched NumPy calculations use the identical row-index
    # bootstrap and estimands, but make the complete 35-spec audit practical.
    for first in range(0, draws, batch_size):
        count = min(batch_size, draws - first)
        sampled = rng.integers(0, n, size=(count, n))
        sampled_values = values[sampled]
        sampled_groups = groups[sampled]
        finite = np.isfinite(sampled_values)
        if kind == "marginal":
            mask_a = (sampled_groups == a) & finite
            mask_b = (sampled_groups == b) & finite
            mean_a = np.divide(
                np.where(mask_a, sampled_values, 0.0).sum(axis=1),
                mask_a.sum(axis=1),
                out=np.full(count, np.nan),
                where=mask_a.sum(axis=1) > 0,
            )
            mean_b = np.divide(
                np.where(mask_b, sampled_values, 0.0).sum(axis=1),
                mask_b.sum(axis=1),
                out=np.full(count, np.nan),
                where=mask_b.sum(axis=1) > 0,
            )
            boots[first:first + count] = mean_b - mean_a
            continue

        for offset in range(count):
            sampled_strata = strata[sampled[offset]]
            valid_strata = sampled_strata >= 0
            total = np.bincount(
                sampled_strata[valid_strata], minlength=n_strata
            ).astype(float)
            mask_a = (sampled_groups[offset] == a) & finite[offset] & valid_strata
            mask_b = (sampled_groups[offset] == b) & finite[offset] & valid_strata
            count_a = np.bincount(sampled_strata[mask_a], minlength=n_strata)
            count_b = np.bincount(sampled_strata[mask_b], minlength=n_strata)
            sum_a = np.bincount(
                sampled_strata[mask_a], weights=sampled_values[offset, mask_a],
                minlength=n_strata,
            )
            sum_b = np.bincount(
                sampled_strata[mask_b], weights=sampled_values[offset, mask_b],
                minlength=n_strata,
            )
            valid = (count_a > 0) & (count_b > 0)
            if not np.any(valid):
                boots[first + offset] = np.nan
                continue
            diffs = sum_b[valid] / count_b[valid] - sum_a[valid] / count_a[valid]
            boots[first + offset] = np.average(diffs, weights=total[valid])

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
