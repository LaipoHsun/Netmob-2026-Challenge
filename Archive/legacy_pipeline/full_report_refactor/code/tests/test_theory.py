"""Validate the Monte Carlo null against the exact Kluyver solution.

This is the test that backs the paper's claim to use an exact finite-size null
rather than the Rayleigh asymptotic. With a median fleet size of 5, the
asymptotic is not good enough and we need to show the difference is handled.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from netmob import estimators, nulls, theory  # noqa: E402


# --------------------------------------------------------------------------
# Exact null distribution
# --------------------------------------------------------------------------

def test_kluyver_cdf_is_a_cdf():
    for n in (2, 3, 5, 8):
        vals = [theory.kluyver_cdf(x, n) for x in (0.1, 0.3, 0.5, 0.7, 0.9)]
        assert all(0.0 <= v <= 1.0 for v in vals)
        assert all(b >= a - 1e-6 for a, b in zip(vals[:-1], vals[1:])), f"not monotone at n={n}"


def test_n2_exact_mean_is_two_over_pi():
    """For N = 2, r1 = |cos(dphi/2)| and E[r1] = 2/pi exactly."""
    assert theory.exact_mean_r_uniform(2) == pytest.approx(2.0 / np.pi, abs=2e-3)


def test_monte_carlo_null_matches_kluyver():
    """The MC null used in the pipeline must reproduce the exact CDF."""
    bank = nulls.uniform_null(max_n=8, draws=60_000, seed=7)
    for n in (2, 3, 4, 5, 6, 7, 8):
        draws = bank.samples[n]
        for x in (0.2, 0.4, 0.6, 0.8):
            mc = float(np.searchsorted(draws, x) / draws.size)
            exact = theory.kluyver_cdf(x, n)
            assert mc == pytest.approx(exact, abs=0.01), f"n={n}, x={x}: MC {mc} vs exact {exact}"


def test_second_moment_is_exactly_one_over_n():
    """E[r1^2] = 1/N holds exactly for every N, not just asymptotically."""
    bank = nulls.uniform_null(max_n=12, draws=200_000, seed=11)
    for n in range(2, 13):
        assert float(np.mean(bank.samples[n] ** 2)) == pytest.approx(1.0 / n, rel=0.02)


def test_rayleigh_asymptotic_is_wrong_at_small_n():
    """Justifies using the exact null: the asymptotic is off where the data live."""
    exact_n2 = theory.exact_mean_r_uniform(2)
    assert abs(exact_n2 - theory.rayleigh_mean_r(2)) > 0.008
    # and converges by the time N is large
    bank = nulls.uniform_null(max_n=40, draws=40_000, seed=3)
    assert float(bank.samples[40].mean()) == pytest.approx(theory.rayleigh_mean_r(40), abs=0.005)


# --------------------------------------------------------------------------
# Estimators
# --------------------------------------------------------------------------

def test_ppc_matches_pairwise_definition():
    """(N r^2 - 1)/(N - 1) must equal mean cos(phi_j - phi_k) over j < k."""
    rng = np.random.default_rng(0)
    for n in (2, 3, 7, 15):
        phi = rng.uniform(0, 2 * np.pi, n)
        r = np.abs(np.mean(np.exp(1j * phi)))
        direct = np.mean([np.cos(phi[j] - phi[k])
                          for j in range(n) for k in range(j + 1, n)])
        formula = estimators.ppc(np.array([r]), np.array([n]))[0]
        assert formula == pytest.approx(direct, abs=1e-10)


def test_ppc_is_unbiased_under_the_null():
    """PPC has null expectation 0 at every N -- no Monte Carlo required."""
    bank = nulls.uniform_null(max_n=12, draws=100_000, seed=5)
    for n in (2, 3, 5, 8, 12):
        vals = estimators.ppc(bank.samples[n], np.full(bank.samples[n].shape, n))
        assert float(vals.mean()) == pytest.approx(0.0, abs=0.01)


def test_r_excess_is_unbiased_under_its_own_null_but_z_is_not_scale_free():
    bank = nulls.uniform_null(max_n=12, draws=60_000, seed=9)
    for n in (3, 6, 12):
        draws = bank.samples[n]
        mu = bank.table.loc[bank.table.N == n, "null_mean"].iloc[0]
        sd = bank.table.loc[bank.table.N == n, "null_sd"].iloc[0]
        rx = estimators.r_excess(draws, np.full(draws.shape, mu))
        z = estimators.z_score(draws, np.full(draws.shape, mu), np.full(draws.shape, sd))
        assert float(rx.mean()) == pytest.approx(0.0, abs=0.01)
        assert float(z.mean()) == pytest.approx(0.0, abs=0.02)
        # r_excess variance shrinks with N; that residual N-dependence is the
        # reason marginal group comparisons stay confounded after correction.
        assert float(rx.std()) > 0.0


def test_r_excess_variance_shrinks_with_n():
    bank = nulls.uniform_null(max_n=20, draws=40_000, seed=13)
    mu = dict(zip(bank.table.N, bank.table.null_mean))
    sds = []
    for n in (2, 5, 10, 20):
        draws = bank.samples[n]
        rx = estimators.r_excess(draws, np.full(draws.shape, mu[n]))
        sds.append(float(rx.std()))
    assert sds == sorted(sds, reverse=True), f"expected decreasing spread, got {sds}"


def test_pit_probit_is_standard_normal_under_the_null():
    """The strictest control: exactly N-free by construction."""
    bank = nulls.uniform_null(max_n=10, draws=50_000, seed=17)
    for n in (2, 5, 10):
        draws = bank.samples[n]
        vals = estimators.pit_probit(draws, np.full(draws.shape, n), bank.samples)
        assert float(np.mean(vals)) == pytest.approx(0.0, abs=0.03)
        assert float(np.std(vals)) == pytest.approx(1.0, abs=0.05)


def test_daido_detects_two_clusters_that_r1_misses():
    """Two antipodal clusters: r1 ~ 0 but r2 ~ 1."""
    phi = np.concatenate([np.zeros(20), np.full(20, np.pi)])
    assert estimators.daido_order(phi, 1) == pytest.approx(0.0, abs=1e-9)
    assert estimators.daido_order(phi, 2) == pytest.approx(1.0, abs=1e-9)


# --------------------------------------------------------------------------
# Newell-Potts
# --------------------------------------------------------------------------

def test_newell_potts_periodic_modes_grow_for_any_k():
    """No threshold in k for periodic modes: |A| > 1 whenever theta != 0."""
    theta = np.linspace(0.05, np.pi, 40)
    for k in (0.05, 0.2, 0.5, 0.9):
        assert np.all(theory.newell_potts_growth(k, theta) > 1.0)


def test_newell_potts_growth_increases_with_load():
    """Growth is monotone in k = arrival rate / boarding rate."""
    theta = np.pi
    vals = [float(theory.newell_potts_growth(k, theta)) for k in (0.1, 0.3, 0.5, 0.7)]
    assert vals == sorted(vals)


def test_newell_potts_uniform_mode_is_neutral():
    """A uniform shift of all buses is not an instability."""
    assert float(theory.newell_potts_growth(0.4, 0.0)) == pytest.approx(1.0, abs=1e-12)


def test_newell_potts_fastest_mode_is_antiphase():
    """theta = pi maximises growth, with |A| = 1 + 2k."""
    for k in (0.1, 0.3, 0.6):
        theta = np.linspace(0.0, np.pi, 200)
        g = theory.newell_potts_growth(k, theta)
        assert theta[int(np.argmax(g))] == pytest.approx(np.pi, abs=0.02)
        assert float(theory.newell_potts_growth(k, np.pi)) == pytest.approx(1.0 + 2 * k, abs=1e-12)


def test_mode1_growth_matches_large_n_expansion():
    """|A(2pi/N)| -> 1 + 2 pi^2 k (1+k) / N^2 for large N."""
    k = 0.25
    for n in (40, 80, 160):
        exact = theory.newell_potts_mode1_growth(k, n)
        approx = 1.0 + 2 * np.pi**2 * k * (1 + k) / n**2
        assert exact == pytest.approx(approx, rel=0.05)


def test_ring_conserves_cycle_time():
    """Headway deviations on a ring must keep summing to zero."""
    traj = theory.newell_potts_ring(0.3, n_buses=12, n_stops=30, seed=1)
    assert np.allclose(traj.sum(axis=1), 0.0, atol=1e-8)


def test_r1_grows_downstream_on_a_ring():
    """The empirical downstream amplification is what the model predicts."""
    traj = theory.newell_potts_r1_trajectory(0.3, n_buses=8, n_stops=25)
    assert traj[-1] > traj[0]


def test_r1_growth_rate_matches_analytic_mode1_prediction():
    """The headline quantitative prediction: d ln r1 / d stop = ln|A(2 pi / N)|.

    This is what makes the downstream-amplification result a theory test rather
    than a description: k and N are both measurable, so the slope is predicted.
    """
    for n in (4, 8, 16):
        for k in (0.05, 0.2, 0.5):
            traj = theory.newell_potts_r1_trajectory(k, n_buses=n, n_stops=15, amplitude=1e-4)
            measured = float(np.log(traj[-1] / traj[0]) / 15)
            predicted = float(np.log(theory.newell_potts_mode1_growth(k, n)))
            assert measured == pytest.approx(predicted, rel=1e-3)


def test_r1_growth_rate_increases_with_load():
    """Higher load per bus means faster growth -- the crossover, quantified."""
    rates = []
    for k in (0.05, 0.2, 0.5):
        traj = theory.newell_potts_r1_trajectory(k, n_buses=8, n_stops=15, amplitude=1e-4)
        rates.append(float(np.log(traj[-1] / traj[0])))
    assert rates == sorted(rates)


def test_r1_growth_rate_falls_with_fleet_size():
    """Mode-1 growth scales like 1/N^2, so big fleets bunch into one clump slowly."""
    k = 0.3
    rates = [float(np.log(theory.newell_potts_mode1_growth(k, n))) for n in (4, 8, 16, 32)]
    assert rates == sorted(rates, reverse=True)
