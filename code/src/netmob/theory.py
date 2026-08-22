"""Exact theory for the finite-size null, and the oscillator models behind it.

Two independent things live here:

1. The exact distribution of the Kuramoto order parameter under N iid uniform
   phases (Pearson's random walk, solved by Kluyver 1905). The analysis pipeline
   uses Monte Carlo nulls; these functions are the ground truth those nulls are
   validated against, which is what lets us claim we use an *exact* null rather
   than the Rayleigh asymptotic E[r] ~ 0.886/sqrt(N).

2. The Newell-Potts follow-the-leader recursion and the Kuramoto critical
   coupling, used to turn the empirical amplification and crossover results into
   quantitative theory tests.

References
----------
Kluyver, J. C. (1905). A local probability problem. Proc. Sect. Sci. Koninklijke
    Akademie van Wetenschappen te Amsterdam 8, 341-350.
Newell, G. F. & Potts, R. B. (1964). Maintaining a bus schedule. Proc. 2nd Conf.
    Australian Road Research Board, Melbourne, vol. 2, 388-393.
Strogatz, S. H. (2000). From Kuramoto to Crawford. Physica D 143, 1-20.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate, special

__all__ = [
    "kluyver_cdf",
    "kluyver_pdf",
    "exact_mean_r_uniform",
    "rayleigh_mean_r",
    "newell_potts_growth",
    "newell_potts_mode1_growth",
    "newell_potts_ring",
    "newell_potts_r1_trajectory",
    "kuramoto_critical_coupling",
]


# --------------------------------------------------------------------------
# 1. Exact null: Pearson random walk / Kluyver
# --------------------------------------------------------------------------

def kluyver_cdf(x: float, n: int, upper: float = 400.0, limit: int = 4000) -> float:
    """P(r_1 <= x) for n iid uniform phases, exactly.

    The resultant R = |sum_j exp(i phi_j)| of an n-step unit random walk has
    Kluyver's CDF  P(R <= a) = a * int_0^inf J_1(a t) J_0(t)^n dt.
    We report it for the normalised order parameter r_1 = R / n, so a = n * x.

    The integrand oscillates and decays only like t^{-(n+1)/2}, so it is
    integrated piecewise between zeros of J_1 rather than in one shot.
    """
    if not 0.0 <= x <= 1.0:
        raise ValueError("x must lie in [0, 1]")
    if n < 1:
        raise ValueError("n must be >= 1")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    if n == 1:
        # A one-step walk has r_1 = 1 deterministically.
        return 0.0
    if n == 2:
        # r_1 = |cos(Delta phi / 2)| has this closed-form CDF.  Using the
        # analytic expression also avoids slowly convergent quadrature at N=2.
        return float(2.0 * np.arcsin(x) / np.pi)

    a = float(n) * x

    def integrand(t: float) -> float:
        return special.j1(a * t) * special.j0(t) ** n

    # Break the range at zeros of J_1(a t) so each panel is a single lobe.
    zero_count = max(250, int(np.ceil(a * upper / np.pi)) + 10)
    zeros = special.jn_zeros(1, zero_count) / a
    knots = [0.0] + [z for z in zeros if z < upper] + [upper]

    total = 0.0
    for lo, hi in zip(knots[:-1], knots[1:]):
        val, _ = integrate.quad(integrand, lo, hi, limit=limit)
        total += val
    return float(np.clip(a * total, 0.0, 1.0))


def kluyver_pdf(x: float, n: int, upper: float = 400.0, limit: int = 4000) -> float:
    """Density of r_1 = R/n under n iid uniform phases.

    p(x) = n^2 x * int_0^inf t J_0(n x t) J_0(t)^n dt
    """
    if not 0.0 <= x <= 1.0:
        raise ValueError("x must lie in [0, 1]")
    a = float(n) * x

    def integrand(t: float) -> float:
        return t * special.j0(a * t) * special.j0(t) ** n

    zero_count = max(250, int(np.ceil(max(a, 1e-9) * upper / np.pi)) + 10)
    zeros = special.jn_zeros(0, zero_count) / max(a, 1e-9)
    knots = [0.0] + [z for z in zeros if z < upper] + [upper]

    total = 0.0
    for lo, hi in zip(knots[:-1], knots[1:]):
        val, _ = integrate.quad(integrand, lo, hi, limit=limit)
        total += val
    return float(n * n * x * total)


def exact_mean_r_uniform(n: int) -> float:
    """E[r_1] under n iid uniform phases from the random-walk integral.

    Closed forms exist only for small n; n = 2 gives 2/pi exactly, which
    tests/test_theory.py checks against.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if n == 1:
        return 1.0
    # For an n-step planar unit random walk,
    # E[R] = integral_0^infinity (1 - J_0(t)^n) / t^2 dt.
    # Beyond U the non-oscillatory leading tail is integral_U^infinity t^-2
    # dt = 1/U; U=500 leaves sub-micro precision for the N used here.
    upper = 500.0

    def integrand(t: float) -> float:
        if t == 0.0:
            return n / 4.0
        return (1.0 - special.j0(t) ** n) / (t * t)

    finite, _ = integrate.quad(integrand, 0.0, upper, limit=4000, epsabs=1e-10)
    return float((finite + 1.0 / upper) / n)


def rayleigh_mean_r(n: int) -> float:
    """Large-N asymptotic E[r_1] ~ 0.8862 / sqrt(N). Shown for contrast only."""
    return float(0.5 * np.sqrt(np.pi / n))


# --------------------------------------------------------------------------
# 2. Newell-Potts follow-the-leader instability
# --------------------------------------------------------------------------

def newell_potts_growth(k: float, theta: np.ndarray | float) -> np.ndarray:
    """Per-stop growth factor |A(theta)| of the Newell-Potts headway recursion.

    Dwell time is proportional to the passengers accumulated since the previous
    bus, so bus n dwells k * h[n, j] at stop j, with
    k = (passenger arrival rate) / (boarding rate). Headways then obey

        h[n, j+1] = (1 + k) h[n, j] - k h[n-1, j]

    and substituting h[n, j] = A^j exp(i theta n) gives

        A(theta) = (1 + k) - k exp(-i theta)
        |A|^2    = (1 + k)^2 - 2 k (1 + k) cos(theta) + k^2

    Two consequences matter for the paper:

    * |A(0)| = 1 exactly -- a uniform timetable shift is neutral, as it must be.
    * |A(theta)| > 1 for every theta != 0 and every k > 0. There is no threshold
      in k. Bunching is an unconditional instability whose *rate* grows with
      load, which is precisely a crossover rather than a critical transition.

    The mode the Kuramoto order parameter r_1 measures is theta = 2 pi / N, so
    r_1 is predicted to grow per stop by |A(2 pi / N)|, which for large N behaves
    like 1 + 2 pi^2 k (1 + k) / N^2. Headway CV, which loads on all modes, is
    dominated by theta = pi where |A| = 1 + 2k.
    """
    theta = np.asarray(theta, dtype=float)
    return np.abs((1.0 + k) - k * np.exp(-1j * theta))


def newell_potts_mode1_growth(k: float, n_buses: int) -> float:
    """Predicted per-stop growth of r_1 specifically: |A(2 pi / N)|."""
    return float(newell_potts_growth(k, 2.0 * np.pi / n_buses))


def newell_potts_ring(
    k: float, n_buses: int, n_stops: int, initial: np.ndarray | None = None,
    seed: int = 0,
) -> np.ndarray:
    """Propagate headway deviations around a ring of n_buses over n_stops.

    A ring is the right boundary condition for the loop-of-buses picture: the
    headways sum to the cycle time, so deviations sum to zero and the neutral
    theta = 0 mode is absent by construction.

    Returns an (n_stops + 1, n_buses) array of headway deviations.
    """
    if initial is None:
        rng = np.random.default_rng(seed)
        initial = rng.normal(size=n_buses)
    delta = np.asarray(initial, dtype=float).copy()
    delta -= delta.mean()

    out = np.zeros((n_stops + 1, n_buses))
    out[0] = delta
    for j in range(n_stops):
        delta = (1.0 + k) * delta - k * np.roll(delta, 1)
        out[j + 1] = delta
    return out


def newell_potts_r1_trajectory(
    k: float,
    n_buses: int,
    n_stops: int,
    amplitude: float = 1e-3,
    mode: int = 1,
    seed: int | None = None,
) -> np.ndarray:
    """r_1 of a Newell-Potts ring as a function of stop index.

    r_1 responds to exactly one Fourier mode of the headway field -- the one
    with theta = 2 pi / N -- so by default the ring is seeded with that mode
    alone. A random seed instead excites every mode, and because the antiphase
    mode grows like (1 + 2k) it quickly swamps the mode-1 signal and drives the
    phases past wrap-around, where r_1 stops being interpretable. Keep the
    amplitude small enough to stay in the linear regime.
    """
    theta = 2.0 * np.pi * mode / n_buses
    idx = np.arange(n_buses)
    if seed is None:
        delta0 = amplitude * np.cos(theta * idx)
    else:
        rng = np.random.default_rng(seed)
        delta0 = amplitude * rng.normal(size=n_buses)

    delta = newell_potts_ring(k, n_buses, n_stops, initial=delta0)
    base = 2.0 * np.pi * idx / n_buses

    out = np.empty(n_stops + 1)
    for j in range(n_stops + 1):
        # Headway deviations are in units of the mean headway; cumulative sum
        # turns them back into angular positions.
        offsets = np.cumsum(delta[j]) * (2.0 * np.pi / n_buses)
        phases = base + offsets - offsets.mean()
        out[j] = np.abs(np.mean(np.exp(1j * phases)))
    return out


# --------------------------------------------------------------------------
# 3. Kuramoto critical coupling from an empirical frequency distribution
# --------------------------------------------------------------------------

def kuramoto_critical_coupling(omega: np.ndarray, bandwidth: float | None = None) -> dict:
    """K_c = 2 / (pi g(0)) for a mean-field Kuramoto ensemble.

    g is the density of natural frequencies centred on its own mean, estimated
    by a Gaussian KDE. Returned alongside the ensemble's frequency spread so the
    two can be compared on the same scale.
    """
    from scipy.stats import gaussian_kde

    omega = np.asarray(omega, dtype=float)
    omega = omega[np.isfinite(omega)]
    if omega.size < 10:
        raise ValueError("need at least 10 finite frequencies")

    centred = omega - omega.mean()
    kde = gaussian_kde(centred, bw_method=bandwidth)
    g0 = float(kde(0.0)[0])
    return {
        "n": int(omega.size),
        "omega_mean": float(omega.mean()),
        "omega_sd": float(omega.std(ddof=1)),
        "g0": g0,
        "K_c": float(2.0 / (np.pi * g0)),
        "K_c_over_omega_sd": float(2.0 / (np.pi * g0) / omega.std(ddof=1)),
    }
