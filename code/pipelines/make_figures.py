#!/usr/bin/env python3
"""Create the paper plots as independent, single-axis PDF files.

Every function below creates exactly one plot and one output file.  The old
multi-panel manuscript figures are deliberately not reconstructed here: their
scientific panels are emitted separately so that a paper author can place,
resize, or omit each result without rerunning unrelated plots.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT / "src"))

from netmob import nulls, theory  # noqa: E402

INK = "#1A1A1A"
RAW = "#B0413E"
CORR = "#1F4E79"
GOLD = "#D99A2B"
GREEN = "#2E7D5B"
GREY = "#9A9A9A"


def set_style() -> None:
    mpl.rcParams.update({
        "font.size": 9.0,
        "axes.labelsize": 9.0,
        "axes.titlesize": 10.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "figure.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "legend.frameon": False,
    })


def one_axis(width: float = 4.2, height: float = 3.1):
    """Return one figure containing one plotting axis (never a subplot grid)."""
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0.16, 0.17, 0.79, 0.74])
    return fig, ax


def save(fig, figure_dir: Path, name: str) -> None:
    fig.savefig(figure_dir / f"{name}.pdf")
    plt.close(fig)
    print(f"{name}.pdf")


def raw_r1_by_fleet_size(data: Path, tables: Path, figures: Path) -> None:
    cells = pd.read_csv(data / "cells_scored_arc_uniform.csv")
    nt = pd.read_csv(tables / "null_tables.csv")
    nt = nt[(nt.phase_def == "arc") & (nt.null_kind == "uniform")].sort_values("N")
    obs = (cells.groupby("N").r1_arc
           .agg(med="median", lo=lambda s: s.quantile(.25),
                hi=lambda s: s.quantile(.75), n="size").reset_index())
    obs = obs[obs.n >= 60]
    nt = nt[nt.N.between(2, int(obs.N.max()))]
    fig, ax = one_axis()
    ax.fill_between(nt.N, nt.null_q025, nt.null_q975, color=GREY, alpha=.32,
                    lw=0, label="null 95% band")
    ax.plot(nt.N, nt.null_mean, color=INK, lw=1.6, label="same-$N$ null mean")
    ax.fill_between(obs.N, obs.lo, obs.hi, color=RAW, alpha=.22, lw=0)
    ax.plot(obs.N, obs.med, color=RAW, lw=1.7, label="observed median (IQR)")
    ax.set(xlim=(2, int(obs.N.max())), ylim=(0, 1), xlabel="active buses $N$",
           ylabel=r"raw $r_1$", title="Raw order tracks fleet size, not coordination")
    ax.legend(loc="upper right")
    save(fig, figures, "01_raw_r1_by_fleet_size")


def null_approximation_error(data: Path, tables: Path, figures: Path) -> None:
    cells = pd.read_csv(data / "cells_scored_arc_uniform.csv")
    ns = np.array([2, 3, 4, 5, 6, 8, 12, 20])
    bank = nulls.uniform_null(int(ns.max()), draws=400_000, seed=7)
    exact = np.array([theory.exact_mean_r_uniform(int(n)) for n in ns])
    mc = np.array([float(bank.samples[int(n)].mean()) for n in ns])
    rayleigh = 0.8862 / np.sqrt(ns)
    fig, ax = one_axis()
    ax.axhline(0, color=INK, lw=.8)
    ax.plot(ns, 100 * (rayleigh - exact) / exact, "^-", color=GOLD, ms=5,
            label="Rayleigh asymptotic")
    ax.plot(ns, 100 * (mc - exact) / exact, "o-", color=CORR, ms=4.3,
            label="Monte Carlo null")
    counts = cells.N.value_counts().reindex(ns, fill_value=0).to_numpy(float)
    scaled = -2.9 + .65 * counts / max(counts.max(), 1)
    ax.vlines(ns, -2.9, scaled, color=GREY, alpha=.35, lw=7,
              label="relative cell frequency")
    ax.set(xlim=(1.5, 21), ylim=(-3.0, 1.5), xlabel="$N$",
           ylabel="error versus exact null (%)",
           title="The asymptotic null is biased where the data lie")
    ax.legend(loc="lower right")
    save(fig, figures, "02_null_approximation_error")


def weekend_contrasts(data: Path, tables: Path, figures: Path) -> None:
    cells = pd.read_csv(data / "cells_scored_arc_uniform.csv")
    con = pd.read_csv(tables / "weekend_contrasts.csv")
    con = con[con.spec == "arc|uniform"]
    order = ["r_excess", "z", "ppc", "rayleigh_z", "pit_probit"]
    labels = {"r_excess": r"$r_{\rm exc}$", "z": "$z$", "ppc": "PPC",
              "rayleigh_z": r"$N r_1^2$", "pit_probit": "probit PIT"}
    spread = {name: cells[name].std() for name in order}
    fig, ax = one_axis(4.6, 3.2)
    for i, estimator in enumerate(order):
        for contrast, color, offset, label in [
            ("marginal", RAW, .16, "marginal"),
            ("n_matched", CORR, -.16, "matched on $N$"),
        ]:
            row = con[(con.estimator == estimator) & (con.contrast == contrast)].iloc[0]
            sd = spread[estimator]
            ax.errorbar(row.point / sd, i + offset,
                        xerr=[[abs(row.point - row.ci_low) / sd],
                              [abs(row.ci_high - row.point) / sd]],
                        fmt="o", ms=4, color=color, capsize=2,
                        label=label if i == 0 else None)
    ax.axvline(0, color=INK, lw=.8)
    ax.set_yticks(range(len(order)), [labels[name] for name in order])
    ax.set(xlabel="weekend minus weekday (SD units)",
           title="Marginal and fleet-size-matched contrasts answer different questions")
    ax.legend(loc="lower left")
    save(fig, figures, "03_weekend_contrasts")


def corrected_load_response(data: Path, tables: Path, figures: Path) -> None:
    """Adjusted load-response curves using the Phase-2 model specification.

    The dots are equal-count-bin means of the piecewise-model partial residual:
    line-direction fixed effects and the non-load controls have been removed and
    then restored at their sample-average value.  Curves hold the same controls
    at that reference value.  This makes the visual correspond to the AIC table
    without presenting the descriptive breakpoint as a critical point.
    """
    frame = pd.read_csv(data / "phase2_hourly_observables.csv")
    frame = frame[frame.analysis_eligible].replace([np.inf, -np.inf], np.nan)
    needed = ["r_excess", "lambda_boardings_per_bus", "N", "rain_hour",
              "line_code", "direction", "day_category", "hour_of_day"]
    frame = frame.dropna(subset=needed).reset_index(drop=True)
    frame["group"] = frame.line_code.astype(str) + "_d" + frame.direction.astype(str)
    frame["weekend"] = (frame.day_category == "weekend").astype(float)
    frame["rain"] = frame.rain_hour.astype(float)
    frame["hour_sin"] = np.sin(2 * np.pi * frame.hour_of_day / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * frame.hour_of_day / 24)
    lam = frame.lambda_boardings_per_bus.to_numpy(float)
    n = frame.N.to_numpy(float)
    lam_mean, lam_sd = float(lam.mean()), float(lam.std(ddof=0))
    n_mean, n_sd = float(n.mean()), float(n.std(ddof=0))
    lam_std = (lam - lam_mean) / lam_sd
    n_std = (n - n_mean) / n_sd

    groups = pd.get_dummies(frame.group, prefix="group", drop_first=True, dtype=float)
    controls = pd.DataFrame({
        "intercept": np.ones(len(frame)), "N_std": n_std,
        "hour_sin": frame.hour_sin, "hour_cos": frame.hour_cos,
        "weekend": frame.weekend, "rain": frame.rain,
    })
    base = pd.concat([controls, groups.reset_index(drop=True)], axis=1).to_numpy(float)
    base_ref = base.mean(axis=0)
    y = frame.r_excess.to_numpy(float)
    comparison = pd.read_csv(tables / "transition_model_comparison.csv").set_index("model")

    c_piece = float(comparison.loc["piecewise", "lambda_c"])
    hinge = np.maximum(0.0, lam - c_piece)
    hinge_mean, hinge_sd = float(hinge.mean()), float(hinge.std(ddof=0))
    hinge_std = (hinge - hinge_mean) / hinge_sd
    x_piece = np.column_stack([base, lam_std, hinge_std])
    beta_piece, *_ = np.linalg.lstsq(x_piece, y, rcond=None)

    x_linear = np.column_stack([base, lam_std])
    beta_linear, *_ = np.linalg.lstsq(x_linear, y, rcond=None)

    c_sig = float(comparison.loc["sigmoid", "lambda_c"])
    scale_sig = float(comparison.loc["sigmoid", "scale"])
    sigmoid = 1 / (1 + np.exp(-(lam - c_sig) / scale_sig))
    sigmoid_mean, sigmoid_sd = float(sigmoid.mean()), float(sigmoid.std(ddof=0))
    sigmoid_std = (sigmoid - sigmoid_mean) / sigmoid_sd
    x_sigmoid = np.column_stack([base, sigmoid_std])
    beta_sigmoid, *_ = np.linalg.lstsq(x_sigmoid, y, rcond=None)

    lo, hi = np.quantile(lam, [.01, .99])
    grid = np.linspace(lo, hi, 300)
    grid_std = (grid - lam_mean) / lam_sd
    grid_hinge = (np.maximum(0, grid - c_piece) - hinge_mean) / hinge_sd
    grid_sigmoid = ((1 / (1 + np.exp(-(grid - c_sig) / scale_sig)))
                    - sigmoid_mean) / sigmoid_sd
    reference = float(base_ref @ beta_piece[:base.shape[1]])
    pred_piece = reference + grid_std * beta_piece[-2] + grid_hinge * beta_piece[-1]
    pred_linear = float(base_ref @ beta_linear[:base.shape[1]]) + grid_std * beta_linear[-1]
    pred_sigmoid = float(base_ref @ beta_sigmoid[:base.shape[1]]) + grid_sigmoid * beta_sigmoid[-1]

    partial = y - base @ beta_piece[:base.shape[1]] + reference
    bin_id = pd.qcut(pd.Series(lam).rank(method="first"), 18, labels=False)
    binned = pd.DataFrame({"lam": lam, "partial": partial, "bin": bin_id}).groupby("bin")
    summary = binned.agg(lam=("lam", "mean"), mean=("partial", "mean"),
                         sd=("partial", "std"), cells=("partial", "size")).reset_index()
    summary["se"] = summary.sd / np.sqrt(summary.cells)

    fig, ax = one_axis(4.7, 3.3)
    ax.errorbar(summary.lam, summary["mean"], yerr=1.96 * summary.se,
                fmt="o", ms=3.8, color=CORR, ecolor="#70879a", capsize=2,
                label="adjusted bin mean (95% CI)")
    ax.plot(grid, pred_piece, color=RAW, lw=1.8,
            label="piecewise ($\\Delta$AIC 0)")
    ax.plot(grid, pred_linear, color=GREY, lw=1.2, ls="--",
            label=f"linear ($\\Delta$AIC {comparison.loc['linear', 'delta_aic']:.1f})")
    ax.plot(grid, pred_sigmoid, color=GREEN, lw=1.2, ls=":",
            label=f"sigmoid ($\\Delta$AIC {comparison.loc['sigmoid', 'delta_aic']:.1f})")
    ax.axvline(c_piece, color=RAW, lw=.9, ls="--")
    ax.text(c_piece, ax.get_ylim()[1], f"  descriptive scale {c_piece:.0f}",
            color=RAW, va="top", fontsize=7.5)
    ax.set(xlim=(lo, hi), xlabel="realized boardings per active bus $\\lambda$",
           ylabel=r"adjusted $r_{\rm exc}$",
           title="Adjusted load response has a descriptive, not critical, scale")
    ax.legend(loc="best", fontsize=7)
    save(fig, figures, "18_corrected_load_response_models")


def bus_position_distribution(data: Path, tables: Path, figures: Path) -> None:
    bh = pd.read_csv(data / "bus_hours_with_phases.csv")
    share = float(((bh.s_mid_frac <= .10) | (bh.s_mid_frac >= .90)).mean())
    fig, ax = one_axis()
    ax.hist(bh.s_mid_frac, bins=60, color=CORR, edgecolor="none", alpha=.85)
    ax.axvspan(0, .10, color=RAW, alpha=.20, lw=0)
    ax.axvspan(.90, 1, color=RAW, alpha=.20, lw=0)
    ax.text(.5, .82, f"{share:.1%} within terminal bands", transform=ax.transAxes,
            ha="center", color=RAW)
    ax.set(xlim=(0, 1), xlabel="position along route $s/L$",
           ylabel="active bus-hours", title="Bus occupancy is concentrated at route ends")
    save(fig, figures, "04_bus_position_distribution")


def synchronization_by_zone(data: Path, tables: Path, figures: Path) -> None:
    tab = pd.read_csv(tables / "synchronization_by_zone.csv")
    tab = tab.set_index("zone").reindex(["terminal", "near_terminal", "interior"])
    fig, ax = one_axis()
    bars = ax.bar(range(3), tab.mean_r_excess, color=[RAW, GOLD, CORR], width=.64)
    for bar, value in zip(bars, tab.mean_r_excess):
        ax.text(bar.get_x() + bar.get_width() / 2,
                value + (.05 if value >= 0 else -.09), f"{value:+.3f}", ha="center")
    ax.axhline(0, color=INK, lw=.8)
    ax.set_xticks(range(3), ["terminal\n<2%", "near\n2--10%", "interior\n10--90%"])
    ax.set(ylabel=r"mean $r_{\rm exc}$",
           title="Terminal and interior corrected order have opposite signs")
    save(fig, figures, "05_synchronization_by_zone")


def terminal_trim_sensitivity(data: Path, tables: Path, figures: Path) -> None:
    tab = pd.read_csv(tables / "terminal_sensitivity.csv")
    fig, ax = one_axis()
    for phase_def, color, marker in [("arc", CORR, "o"), ("time", GREEN, "s")]:
        group = tab[tab.phase_def == phase_def].sort_values("trim")
        ax.plot(group.trim * 100, group.mean_r_excess, marker + "-", color=color,
                ms=4, label=f"{phase_def} phase")
    ax.axhline(0, color=INK, lw=.8)
    ax.axvspan(2, 10, color=GREY, alpha=.16, lw=0)
    ax.set(xlabel="route trimmed at each end (%)", ylabel=r"mean $r_{\rm exc}$",
           title="Removing the terminal band reverses the sign")
    ax.legend()
    save(fig, figures, "06_terminal_trim_sensitivity")


def load_effect_by_zone(data: Path, tables: Path, figures: Path) -> None:
    tab = pd.read_csv(tables / "load_effect_by_zone.csv")
    order = ["zone=interior", "zone=near_terminal", "zone=terminal",
             "whole route (as submitted)"]
    labels = {"whole route (as submitted)": "whole route", "zone=terminal": "terminal",
              "zone=near_terminal": "near terminal", "zone=interior": "interior"}
    tab = tab[tab.spec.isin(order)].set_index("spec").reindex(order)
    fig, ax = one_axis(4.5, 3.0)
    for i, (spec, row) in enumerate(tab.iterrows()):
        color = RAW if spec.startswith("whole") else CORR
        ax.errorbar(row.r_excess_lam_coef, i,
                    xerr=[[row.r_excess_lam_coef - row.r_excess_lam_ci_lo],
                          [row.r_excess_lam_ci_hi - row.r_excess_lam_coef]],
                    fmt="o", ms=4.5, color=color, capsize=2)
    ax.axvline(0, color=INK, lw=.8)
    ax.set_yticks(range(len(tab)), [labels[spec] for spec in tab.index])
    ax.set(xlabel=r"load effect on $r_{\rm exc}$",
           title="The whole-route association is not detected within zones")
    save(fig, figures, "07_load_effect_by_zone")


def newell_potts_theory_curves(data: Path, tables: Path, figures: Path) -> None:
    theta = np.linspace(0, np.pi, 400)
    fig, ax = one_axis()
    for k, color in zip([.05, .15, .30, .50], [GREY, GOLD, GREEN, RAW]):
        ax.plot(theta / np.pi, theory.newell_potts_growth(k, theta), color=color,
                label=f"$k={k}$")
    ax.axhline(1, color=INK, lw=.9, ls="--")
    ax.set(xlim=(0, 1.05), ylim=(.95, 2.14), xlabel=r"headway mode $\theta/\pi$",
           ylabel=r"growth per stop $|A(\theta)|$",
           title="Every nonzero Newell--Potts mode grows for positive $k$")
    ax.legend(loc="upper left")
    save(fig, figures, "08_newell_potts_theory_curves")


def newell_potts_empirical_test(data: Path, tables: Path, figures: Path) -> None:
    frame = pd.read_csv(data / "newell_potts_test.csv")
    frame = frame[["predicted_growth_per_stop", "measured_growth_per_stop"]].dropna()
    frame = frame[(frame.predicted_growth_per_stop < .32) &
                  frame.measured_growth_per_stop.between(-.06, .19)]
    summary = json.loads((tables / "theory_tests_summary.json").read_text())
    corr = float(summary["newell_potts"]["corr_measured_predicted"])
    fig, ax = one_axis(4.5, 3.3)
    density = ax.hexbin(frame.predicted_growth_per_stop, frame.measured_growth_per_stop,
                        gridsize=34, cmap="Blues", mincnt=1, linewidths=0)
    colorbar = fig.colorbar(density, ax=ax, pad=.02)
    colorbar.set_label("cells")
    bins = np.linspace(0, .30, 9)
    mid = .5 * (bins[:-1] + bins[1:])
    med = [frame.measured_growth_per_stop[
        (frame.predicted_growth_per_stop >= lo) &
        (frame.predicted_growth_per_stop < hi)].median() for lo, hi in zip(bins[:-1], bins[1:])]
    ax.plot(mid, med, "o-", color=RAW, ms=4, label="binned median")
    ax.plot([0, .32], [0, .32], "--", color=INK, lw=1, label="1:1")
    ax.text(.97, .05, f"$r={corr:.3f}$", transform=ax.transAxes, ha="right")
    ax.set(xlim=(0, .32), ylim=(-.06, .19),
           xlabel=r"predicted $\ln(1+2k)$ per stop",
           ylabel=r"measured $d\ln\mathrm{CV}/d\,$stop",
           title="Measured load does not rank which cells amplify")
    ax.legend(loc="upper left")
    save(fig, figures, "09_newell_potts_empirical_test")


def kuramoto_threshold(data: Path, tables: Path, figures: Path) -> None:
    frame = pd.read_csv(data / "kuramoto_kc.csv")
    fig, ax = one_axis()
    ax.scatter(frame.omega_sd, frame.K_c, s=22, color=CORR, alpha=.75,
               edgecolor="white", linewidth=.4)
    lo, hi = frame.omega_sd.min() * .92, frame.omega_sd.max() * 1.06
    x = np.linspace(lo, hi, 50)
    ax.plot(x, np.sqrt(8 / np.pi) * x, "--", color=GREY,
            label=r"Gaussian $g$: $K_c=\sqrt{8/\pi}\,\sigma_\omega$")
    ax.axhline(frame.K_c.median(), color=RAW, lw=1.1, ls=":")
    ax.set(xlim=(lo, hi), xlabel=r"frequency spread $\sigma_\omega$ (rad/min)",
           ylabel=r"$K_c=2/\pi g(0)$ (rad/min)",
           title="Frequency heterogeneity implies a high mean-field threshold")
    ax.legend(loc="upper left")
    save(fig, figures, "10_kuramoto_threshold")


def collapse_curve(data: Path, tables: Path, figures: Path, source: str,
                   summary_key: str, output_name: str, title: str) -> None:
    frame = pd.read_csv(data / source)
    summary = json.loads((tables / "theory_tests_summary.json").read_text())[summary_key]
    fig, ax = one_axis()
    strata = sorted(frame.stratum.unique(), key=lambda s: int(s.split("-")[0].replace("N", "")))
    colors = plt.cm.viridis(np.linspace(.06, .86, len(strata)))
    for color, stratum in zip(colors, strata):
        group = frame[frame.stratum == stratum]
        ax.plot(group.x_scaled, group.y_scaled, "o", ms=3, color=color, alpha=.9,
                mec="white", mew=.3, label=stratum.replace("N", "$N$="))
    xlo, xhi = np.percentile(frame.x_scaled, [1, 90])
    ylo, yhi = np.percentile(frame.y_scaled, [2, 95])
    inside = frame[frame.x_scaled.between(xlo, xhi)].sort_values("x_scaled")
    if len(inside) >= 9:
        trend = inside.y_scaled.rolling(7, center=True, min_periods=4).median()
        ax.plot(inside.x_scaled, trend, color=INK, lw=1, label="trend")
    status = "identified" if summary.get("collapse_identified", False) else "not identified"
    ax.set(xlim=(xlo, xhi), ylim=(ylo, yhi), xlabel=r"$(\lambda-\lambda_c)N^a$",
           ylabel=r"$r_{\rm exc}N^b$", title=f"{title}: {status}")
    ax.legend(fontsize=6, ncol=2)
    save(fig, figures, output_name)


def scaling_optimizer_diagnostic(data: Path, tables: Path, figures: Path) -> None:
    summaries = json.loads((tables / "theory_tests_summary.json").read_text())
    specs = [
        ("scaling_collapse_interior_null.txt", CORR, "interior", summaries["scaling_collapse_interior"]),
        ("scaling_collapse_null.txt", RAW, "whole route", summaries["scaling_collapse_whole"]),
    ]
    fig, ax = one_axis(4.6, 3.2)
    xmax = 1.0
    for filename, color, label, summary in specs:
        values = np.loadtxt(data / filename)
        observed = float(summary["improvement_ratio"])
        broad = float(summary.get("broad_start_improvement_ratio", observed))
        xmax = max(xmax, float(np.max(values)), observed, broad)
        ax.hist(values, bins=18, color=color, alpha=.30, density=True, label=f"{label} null")
        ax.axvline(observed, color=color, lw=1.5)
        if not np.isclose(broad, observed):
            ax.axvline(broad, color=color, lw=1.2, ls="--")
    ax.set(xlim=(1, xmax * 1.08), xlabel="collapse improvement ratio",
           ylabel="null density", title="Scaling solutions depend on optimizer start and bounds")
    ax.legend()
    save(fig, figures, "13_scaling_optimizer_diagnostic")


def daido_harmonics(data: Path, tables: Path, figures: Path) -> None:
    harmonics = pd.read_csv(tables / "daido_harmonics.csv")
    cells = pd.read_csv(data / "cells_scored_arc_uniform.csv")
    fig, ax = one_axis()
    for phase_def, color, marker in [("arc", CORR, "o"), ("time", GREEN, "s")]:
        group = harmonics[harmonics.phase_def == phase_def].sort_values("m")
        ax.plot(group.m, group.mean_r_excess, marker + "-", color=color, ms=4.5,
                label=f"{phase_def} phase")
    ns = cells.N.to_numpy(int)
    bank = nulls.uniform_null(int(ns.max()), draws=20_000, seed=11)
    means = dict(zip(bank.table.N, bank.table.null_mean))
    null_mean = np.array([means[n] for n in ns])

    def corrected(kappa: float, harmonic: int, seed: int = 3) -> float:
        rng = np.random.default_rng(seed)
        observed = np.array([abs(np.mean(np.exp(1j * harmonic * rng.vonmises(0, kappa, n))))
                             for n in ns])
        return float(np.mean((observed - null_mean) / (1 - null_mean)))

    target = float(harmonics[(harmonics.phase_def == "arc") &
                             (harmonics.m == 1)].mean_r_excess.iloc[0])
    lo, hi = 1e-3, 12.0
    for _ in range(28):
        mid = .5 * (lo + hi)
        if corrected(mid, 1) < target:
            lo = mid
        else:
            hi = mid
    reference = [corrected(.5 * (lo + hi), m) for m in (1, 2, 3)]
    ax.plot([1, 2, 3], reference, "^--", color=GREY, ms=4, label="smooth cluster")
    ax.set_xticks([1, 2, 3])
    ax.set(xlim=(.8, 3.2), ylim=(-.03, .40), xlabel="harmonic $m$",
           ylabel=r"corrected $|Z_m|$", title="Flat higher harmonics indicate compact clumps")
    ax.legend()
    save(fig, figures, "14_daido_harmonics")


def corridor_speed_control(data: Path, tables: Path, figures: Path) -> None:
    pairs = pd.read_csv(data / "corridor_coupling_speed_controlled.csv")
    summary = json.loads((tables / "corridor_coupling_gate_summary.json").read_text())
    plot = pairs.dropna(subset=["raw_shared_minus_off", "speed_controlled_shared_minus_off"])
    rng = np.random.default_rng(20260814)
    x0 = rng.normal(0, .015, len(plot)); x1 = rng.normal(1, .015, len(plot))
    fig, ax = one_axis(4.3, 3.2)
    for i, (_, row) in enumerate(plot.iterrows()):
        ax.plot([x0[i], x1[i]], [row.raw_shared_minus_off,
                                row.speed_controlled_shared_minus_off],
                color="#bdbdbd", lw=.6, alpha=.45)
    ax.scatter(x0, plot.raw_shared_minus_off, s=15, color=CORR, alpha=.8)
    ax.scatter(x1, plot.speed_controlled_shared_minus_off, s=15, color=RAW, alpha=.8)
    ax.errorbar([0, 1], [summary["raw_gap_mean"], summary["speed_controlled_gap_mean"]],
                yerr=[[summary["raw_gap_mean"] - summary["raw_gap_ci_low"],
                       summary["speed_controlled_gap_mean"] - summary["speed_controlled_gap_ci_low"]],
                      [summary["raw_gap_ci_high"] - summary["raw_gap_mean"],
                       summary["speed_controlled_gap_ci_high"] - summary["speed_controlled_gap_mean"]]],
                fmt="none", color=INK, capsize=3)
    ax.axhline(0, color=INK, lw=.8)
    ax.set_xticks([0, 1], ["raw", "speed controlled"])
    ax.set(ylabel=r"shared minus off-trunk corr($r_{excess}$)",
           title="Shared-corridor coherence largely disappears after speed control")
    save(fig, figures, "15_corridor_speed_control_effect")


def corridor_lag_distribution(data: Path, tables: Path, figures: Path) -> None:
    pairs = pd.read_csv(data / "corridor_coupling_speed_controlled.csv")
    lags = pairs.loc[pairs.significant_nonzero_lag, "best_nonzero_lag_hours"].dropna()
    fig, ax = one_axis()
    ax.hist(lags, bins=np.arange(-4.5, 5.5, 1), color=GREEN, edgecolor="white")
    ax.axvline(0, color=INK, lw=.8)
    ax.set(xlabel="best significant non-zero lag (hours)", ylabel="pairs",
           title="Significant lead--lag peaks are rare")
    save(fig, figures, "16_corridor_significant_lag_distribution")


def amplification_drivers(data: Path, tables: Path, figures: Path) -> None:
    coefficients = pd.read_csv(tables / "amplification_driver_coefficients.csv")
    names = {"initial_headway_cv_std": "initial CV",
             "lambda_boardings_per_bus_std": "demand per bus",
             "stop_density_per_km_std": "stops/km",
             "route_length_km_std": "route length", "sinuosity_std": "sinuosity"}
    plot = coefficients[coefficients.term.isin(names)].copy()
    plot["label"] = plot.term.map(names)
    plot = plot.iloc[::-1]
    fig, ax = one_axis(4.4, 3.0)
    y = np.arange(len(plot))
    ax.errorbar(plot.estimate, y,
                xerr=[plot.estimate - plot.ci_low, plot.ci_high - plot.estimate],
                fmt="o", color=CORR, ecolor="#555555", capsize=3)
    ax.axvline(0, color=INK, lw=.8)
    ax.set_yticks(y, plot.label)
    ax.set(xlabel="standardized effect on amplification slope",
           title="Inherited irregularity dominates downstream amplification")
    save(fig, figures, "17_amplification_driver_coefficients")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    package = Path(__file__).resolve().parents[2]
    parser.add_argument("--data-dir", type=Path, default=package / "output" / "data")
    parser.add_argument("--table-dir", type=Path, default=package / "output" / "tables")
    parser.add_argument("--figure-dir", type=Path, default=package / "output" / "figures")
    args = parser.parse_args()
    data = args.data_dir.resolve(); tables = args.table_dir.resolve()
    figures = args.figure_dir.resolve(); figures.mkdir(parents=True, exist_ok=True)
    set_style()
    raw_r1_by_fleet_size(data, tables, figures)
    null_approximation_error(data, tables, figures)
    weekend_contrasts(data, tables, figures)
    corrected_load_response(data, tables, figures)
    bus_position_distribution(data, tables, figures)
    synchronization_by_zone(data, tables, figures)
    terminal_trim_sensitivity(data, tables, figures)
    load_effect_by_zone(data, tables, figures)
    newell_potts_theory_curves(data, tables, figures)
    newell_potts_empirical_test(data, tables, figures)
    kuramoto_threshold(data, tables, figures)
    collapse_curve(data, tables, figures, "scaling_collapse.csv",
                   "scaling_collapse_whole", "11_scaling_collapse_whole_route",
                   "Whole-route collapse")
    collapse_curve(data, tables, figures, "scaling_collapse_interior.csv",
                   "scaling_collapse_interior", "12_scaling_collapse_interior",
                   "Interior collapse")
    scaling_optimizer_diagnostic(data, tables, figures)
    daido_harmonics(data, tables, figures)
    corridor_speed_control(data, tables, figures)
    corridor_lag_distribution(data, tables, figures)
    amplification_drivers(data, tables, figures)


if __name__ == "__main__":
    main()
