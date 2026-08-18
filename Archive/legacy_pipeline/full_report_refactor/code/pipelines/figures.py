"""Figures for the full report. Writes PDFs to figures/.

Design rules kept consistent across all four figures: one accent colour for the
uncorrected/terminal quantities, one for the corrected/interior ones, annotation
placed inside the axes rather than in the caption where it carries the argument,
and no raw scatter blobs -- densities are shown as binned bands or hexbins so the
structure is visible rather than the ink.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import special

CODE_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(CODE_ROOT / "src"))

from netmob import nulls, theory  # noqa: E402

DATA = ARCHIVE_ROOT / "outputs"
FIG = PROJECT_ROOT / "Output" / "Figure"

mpl.rcParams.update({
    "font.size": 7.6, "axes.labelsize": 7.6, "axes.titlesize": 8.2,
    "xtick.labelsize": 6.9, "ytick.labelsize": 6.9, "legend.fontsize": 6.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .7, "xtick.major.width": .7, "ytick.major.width": .7,
    "xtick.major.size": 2.6, "ytick.major.size": 2.6,
    "axes.titlepad": 6, "axes.labelpad": 2.5,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": .02,
    "lines.linewidth": 1.3, "legend.frameon": False,
    "legend.handlelength": 1.4, "legend.handletextpad": .5,
    "legend.borderaxespad": .3, "legend.labelspacing": .3,
})

INK = "#1A1A1A"
RAW = "#B0413E"     # uncorrected, terminal, "the problem"
CORR = "#1F4E79"    # corrected, interior, "the fix"
GOLD = "#D99A2B"
GREEN = "#2E7D5B"
GREY = "#9A9A9A"


def titled(ax, letter: str, text: str) -> None:
    """Panel title with its letter folded in, left-aligned."""
    ax.set_title(f"$\\bf{{({letter})}}$  {text}", loc="left",
                 fontsize=mpl.rcParams["axes.titlesize"])


# ------------------------------------------------------------------ Fig 1
def fig_observable() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35))
    cells = pd.read_csv(DATA / "week1_2" / "cells_scored_arc_uniform.csv")
    nt = pd.read_csv(DATA / "week1_2" / "null_tables.csv")
    nt = nt[(nt.phase_def == "arc") & (nt.null_kind == "uniform")].sort_values("N")
    nt = nt[nt.N.between(2, 28)]

    # --- (a) observed r1 against its same-N null, as bands not scatter
    ax = axes[0]
    obs = (cells.groupby("N").r1_arc
           .agg(med="median", lo=lambda s: s.quantile(.25),
                hi=lambda s: s.quantile(.75), n="size").reset_index())
    obs = obs[obs.n >= 60]          # keep only N with a stable quantile estimate
    nmax = int(obs.N.max())
    nt = nt[nt.N <= nmax]
    ax.fill_between(nt.N, nt.null_q025, nt.null_q975, color=GREY, alpha=.32,
                    lw=0, label="null 95% band")
    ax.plot(nt.N, nt.null_mean, color=INK, lw=1.6, label="exact null mean")
    ax.fill_between(obs.N, obs.lo, obs.hi, color=RAW, alpha=.22, lw=0)
    ax.plot(obs.N, obs.med, color=RAW, lw=1.7, label="observed median (IQR)")
    ax.set_xlim(2, nmax); ax.set_ylim(0, 1)
    ax.set_xlabel("active buses $N$"); ax.set_ylabel(r"raw $r_1$")
    titled(ax, "a", "Raw $r_1$ tracks fleet size,\nnot coordination")
    ax.legend(loc="upper right")
    ax.annotate("observed sits just above\nchance at every $N$",
                xy=(.52, .11), xycoords="axes fraction", ha="center",
                fontsize=6.3, color=INK)

    # --- (b) how wrong the asymptotic null is, where the data actually are
    ax = axes[1]
    ns = np.array([2, 3, 4, 5, 6, 8, 12, 20])
    bank = nulls.uniform_null(int(ns.max()), draws=400_000, seed=7)
    ex = np.array([theory.exact_mean_r_uniform(int(n)) for n in ns])
    mc = np.array([float(bank.samples[int(n)].mean()) for n in ns])
    ray = 0.8862 / np.sqrt(ns)

    ax2 = ax.twinx()
    ax2.hist(cells.N, bins=np.arange(1.5, 21.5), color=GREY, alpha=.20, lw=0)
    ax2.set_yticks([]); ax2.set_ylim(0, ax2.get_ylim()[1] * 3.0)
    for s in ax2.spines.values():
        s.set_visible(False)

    ax.axhline(0, color=INK, lw=.8)
    ax.plot(ns, 100 * (ray - ex) / ex, "^-", color=GOLD, ms=4.5, lw=1.4,
            label="Rayleigh asymptotic", zorder=4)
    ax.plot(ns, 100 * (mc - ex) / ex, "o-", color=CORR, ms=3.8, lw=1.4,
            label="our Monte Carlo null", zorder=5)
    ax.set_xlim(1.5, 21); ax.set_ylim(-3.0, 1.5)
    ax.set_xlabel("$N$"); ax.set_ylabel("error vs exact null (%)")
    titled(ax, "b", "The asymptotic null is biased\nwhere the data live")
    ax.legend(loc="lower right")
    ax.annotate("grey: where the\ncells actually are", xy=(5.0, 1.16),
                fontsize=6.3, color=GREY, ha="left", va="top")
    ax.set_zorder(ax2.get_zorder() + 1); ax.patch.set_visible(False)

    # --- (c) the weekend contrast, in comparable units
    ax = axes[2]
    con = pd.read_csv(DATA / "week1_2" / "weekend_contrasts.csv")
    con = con[con.spec == "arc|uniform"]
    order = ["r_excess", "z", "ppc", "rayleigh_z", "pit_probit"]
    labels = {"r_excess": r"$r_{\rm exc}$", "z": "$z$", "ppc": "PPC",
              "rayleigh_z": r"$N r_1^2$", "pit_probit": "probit PIT"}
    # Standardize by each estimator's own cell-level spread, so the five sit on
    # one axis as standardized mean differences rather than arbitrary units.
    sd = {e: cells[e].std() for e in order}
    for i, e in enumerate(order):
        for kind, col, off, lab in [("marginal", RAW, .17, "marginal"),
                                    ("n_matched", CORR, -.17, "matched on $N$")]:
            row = con[(con.estimator == e) & (con.contrast == kind)].iloc[0]
            s = sd[e]
            ax.errorbar(row.point / s, i + off,
                        xerr=[[abs(row.point - row.ci_low) / s],
                              [abs(row.ci_high - row.point) / s]],
                        fmt="o", ms=3.6, color=col, lw=1.2, capsize=2,
                        label=lab if i == 0 else None)
    ax.axvline(0, color=INK, lw=.8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([labels[e] for e in order])
    ax.set_ylim(-.95, len(order) - .35)
    ax.set_xlabel("weekend $-$ weekday (SD units)")
    titled(ax, "c", "The reversal is a choice of estimand,\nnot of estimator")
    ax.legend(loc="lower left", ncol=2, columnspacing=.9)

    fig.tight_layout(w_pad=1.9)
    fig.savefig(FIG / "f1_observable.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ Fig 2
def fig_route_position() -> None:
    fig, axes = plt.subplots(1, 4, figsize=(7.35, 2.15))
    bh = pd.read_csv(DATA / "week1_2" / "bus_hours_with_phases.csv")

    ax = axes[0]
    ax.hist(bh.s_mid_frac, bins=60, color=CORR, edgecolor="none", alpha=.85)
    ax.axvspan(0, .10, color=RAW, alpha=.20, lw=0)
    ax.axvspan(.90, 1, color=RAW, alpha=.20, lw=0)
    ax.set_xlim(0, 1)
    ax.set_xlabel("position along route  $s/L$")
    ax.set_ylabel("active bus-hours")
    titled(ax, "a", "Half the fleet sits\nat the route ends")
    ax.annotate("51% inside\nthe shaded bands", xy=(.5, .80),
                xycoords="axes fraction", ha="center", fontsize=6.5, color=RAW)

    ax = axes[1]
    bz = pd.read_csv(DATA / "week3" / "synchronization_by_zone.csv")
    names = {"terminal": "terminal\n$<$2%", "near_terminal": "near\n2-10%",
             "interior": "interior\n10-90%"}
    bz = bz.set_index("zone").reindex(["terminal", "near_terminal", "interior"])
    bars = ax.bar(range(3), bz.mean_r_excess, color=[RAW, GOLD, CORR], width=.64)
    for b, v in zip(bars, bz.mean_r_excess):
        ax.text(b.get_x() + b.get_width() / 2, v + (.06 if v > 0 else -.10),
                f"{v:+.2f}", ha="center", fontsize=6.6,
                color=INK, fontweight="bold")
    ax.axhline(0, color=INK, lw=.8)
    ax.set_xticks(range(3))
    ax.set_xticklabels([names[i] for i in bz.index], fontsize=6.2)
    ax.set_ylim(-.42, 1.22)
    ax.set_ylabel(r"mean $r_{\rm exc}$")
    titled(ax, "b", "The two zones have\nopposite signs")

    ax = axes[2]
    # The two historical copies were byte-identical; retain the canonical Week 1–2 copy.
    ts = pd.read_csv(DATA / "week1_2" / "terminal_sensitivity.csv")
    for pdname, col, mk in [("arc", CORR, "o"), ("time", GREEN, "s")]:
        s = ts[ts.phase_def == pdname].sort_values("trim")
        ax.plot(s.trim * 100, s.mean_r_excess, mk + "-", color=col, ms=3.4,
                label=f"{pdname} phase")
    ax.axhline(0, color=INK, lw=.8)
    ax.axvspan(2, 10, color=GREY, alpha=.16, lw=0)
    ax.set_xlabel("route trimmed at each end (%)")
    ax.set_ylabel(r"mean $r_{\rm exc}$")
    titled(ax, "c", "Trimming 2% flips\nthe sign")
    ax.legend(loc="upper center")
    ax.text(6, .135, "informative\nrange", ha="center", fontsize=6.1, color=GREY)

    ax = axes[3]
    le = pd.read_csv(DATA / "week3" / "load_effect_by_zone.csv")
    keep = ["zone=interior", "zone=near_terminal", "zone=terminal",
            "whole route (as submitted)"]
    lbl = {"whole route (as submitted)": "whole route", "zone=terminal": "terminal",
           "zone=near_terminal": "near", "zone=interior": "interior"}
    le = le[le.spec.isin(keep)].set_index("spec").reindex(keep)
    for i, (sp, row) in enumerate(le.iterrows()):
        col = RAW if sp.startswith("whole") else CORR
        ax.errorbar(row.r_excess_lam_coef, i,
                    xerr=[[row.r_excess_lam_coef - row.r_excess_lam_ci_lo],
                          [row.r_excess_lam_ci_hi - row.r_excess_lam_coef]],
                    fmt="o", ms=3.8, color=col, lw=1.2, capsize=2)
    ax.axvline(0, color=INK, lw=.8)
    ax.set_yticks(range(len(le)))
    ax.set_yticklabels([lbl[s] for s in le.index], fontsize=6.6)
    ax.set_ylim(-.6, len(le) - .4)
    ax.set_xlabel(r"load effect on $r_{\rm exc}$")
    titled(ax, "d", "The load effect lives\nonly in the mixture")

    fig.tight_layout(w_pad=1.7)
    fig.savefig(FIG / "f2_route_position.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ Fig 3
def fig_theory() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35))

    ax = axes[0]
    th = np.linspace(0, np.pi, 400)
    for k, col in zip([0.05, 0.15, 0.3, 0.5], [GREY, GOLD, GREEN, RAW]):
        ax.plot(th / np.pi, theory.newell_potts_growth(k, th), color=col,
                label=f"$k={k}$")
    ax.axhline(1, color=INK, lw=.9, ls=(0, (2, 2)))
    ax.text(.53, .968, "neutral: uniform timetable shift", fontsize=6.2, color=INK)
    ax.annotate(r"$|A(\pi)|=1+2k$", xy=(1.0, 2.0), xytext=(.52, 1.83),
                fontsize=6.6, color=RAW,
                arrowprops=dict(arrowstyle="->", color=RAW, lw=.7))
    ax.set_xlim(0, 1.05); ax.set_ylim(.95, 2.14)
    ax.set_xlabel(r"headway mode  $\theta/\pi$")
    ax.set_ylabel(r"growth per stop  $|A(\theta)|$")
    titled(ax, "a", "No threshold in $k$:\nevery mode grows")
    ax.legend(loc="upper left")

    ax = axes[1]
    npt = pd.read_csv(DATA / "week3" / "newell_potts_test.csv")
    d = npt[["predicted_growth_per_stop", "measured_growth_per_stop"]].dropna()
    d = d[(d.predicted_growth_per_stop < .32) & d.measured_growth_per_stop.between(-.06, .19)]
    hb = ax.hexbin(d.predicted_growth_per_stop, d.measured_growth_per_stop,
                   gridsize=34, cmap="Blues", mincnt=1, linewidths=0)
    cb = fig.colorbar(hb, ax=ax, pad=.02, fraction=.045)
    cb.set_label("cells", fontsize=6.4); cb.ax.tick_params(labelsize=6)
    cb.outline.set_visible(False)
    bins = np.linspace(0, .30, 9)
    mid = .5 * (bins[:-1] + bins[1:])
    med = [d.measured_growth_per_stop[
        (d.predicted_growth_per_stop >= lo) & (d.predicted_growth_per_stop < hi)].median()
        for lo, hi in zip(bins[:-1], bins[1:])]
    ax.plot(mid, med, "o-", color=RAW, ms=3.4, lw=1.4, label="binned median")
    ax.plot([0, .32], [0, .32], ls=(0, (3, 2)), color=INK, lw=1, label="1:1")
    ax.axhline(0, color=GREY, lw=.7)
    ax.set_xlim(0, .32); ax.set_ylim(-.06, .19)
    ax.set_xlabel(r"predicted $\ln(1+2k)$ per stop")
    ax.set_ylabel(r"measured $d\ln\mathrm{CV}/d\,$stop")
    titled(ax, "b", "Right magnitude, but $k$ does not\npredict which cells amplify")
    
    ax.legend(loc="upper left")
    ax.text(.305, -.045, r"$r=0.004$", ha="right", fontsize=6.6, color=INK)

    ax = axes[2]
    kc = pd.read_csv(DATA / "week3" / "kuramoto_kc.csv")
    ax.scatter(kc.omega_sd, kc.K_c, s=17, color=CORR, alpha=.75,
               edgecolor="white", linewidth=.4)
    lo, hi = kc.omega_sd.min() * .92, kc.omega_sd.max() * 1.06
    xs = np.linspace(lo, hi, 50)
    # Gaussian g(omega) gives K_c = sqrt(8/pi) * sigma; the reference line shows
    # how closely the empirical spreads follow that.
    ax.plot(xs, np.sqrt(8 / np.pi) * xs, ls=(0, (3, 2)), color=GREY, lw=1.1,
            label=r"Gaussian $g$: $K_c=\sqrt{8/\pi}\,\sigma_\omega$")
    ax.axhline(kc.K_c.median(), color=RAW, lw=1.1, ls=(0, (1, 2)))
    ax.text(hi, kc.K_c.median() * 1.06, f"median $K_c={kc.K_c.median():.3f}$",
            ha="right", fontsize=6.4, color=RAW)
    ax.set_xlim(lo, hi)
    ax.set_xlabel(r"frequency spread  $\sigma_\omega$ (rad/min)")
    ax.set_ylabel(r"$K_c=2/\pi g(0)$ (rad/min)")
    titled(ax, "c", "Heterogeneity sets a high\nmean-field threshold")
    ax.legend(loc="upper left")

    fig.tight_layout(w_pad=2.0)
    fig.savefig(FIG / "f3_theory.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ Fig 4
def _plot_collapse(ax, fname: str, title: str, letter: str) -> None:
    """Markers only: connecting the noisy small-N strata makes the panel
    unreadable without adding information. The shared trend is drawn once,
    across all strata, as a rolling median."""
    cur = pd.read_csv(DATA / "week3" / fname)
    strata = sorted(cur.stratum.unique(),
                    key=lambda s: int(s.split("-")[0].replace("N", "")))
    cmap = plt.cm.viridis(np.linspace(.06, .86, len(strata)))
    for col, st in zip(cmap, strata):
        g = cur[cur.stratum == st]
        ax.plot(g.x_scaled, g.y_scaled, "o", ms=2.4, color=col, alpha=.9,
                mec="white", mew=.3, label=st.replace("N", "$N$="))

    xlo, xhi = np.percentile(cur.x_scaled, [1, 90])
    ylo, yhi = np.percentile(cur.y_scaled, [2, 95])
    ins = cur[(cur.x_scaled >= xlo) & (cur.x_scaled <= xhi)].sort_values("x_scaled")
    if len(ins) >= 9:
        roll = ins.y_scaled.rolling(7, center=True, min_periods=4).median()
        ax.plot(ins.x_scaled, roll, color=INK, lw=.9, alpha=.75,
                label="trend")
    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
    ax.set_xlabel(r"$(\lambda-\lambda_c)\,N^{a}$")
    ax.set_ylabel(r"$r_{\rm exc}\,N^{b}$")
    titled(ax, letter, title)
    ax.legend(fontsize=4.4, loc="best", labelspacing=.16, handletextpad=.28,
              ncol=2, columnspacing=.5, borderpad=.2)


def fig_collapse_and_harmonics() -> None:
    """Sized for a single column, so it can be placed with \\linewidth."""
    small = {"font.size": 5.2, "axes.labelsize": 5.2, "axes.titlesize": 5.6,
             "xtick.labelsize": 4.6, "ytick.labelsize": 4.6,
             "xtick.major.size": 1.8, "ytick.major.size": 1.8,
             "axes.labelpad": 1.6, "axes.titlepad": 3.0}
    ctx = mpl.rc_context(small)
    ctx.__enter__()
    fig, axgrid = plt.subplots(2, 2, figsize=(3.45, 3.35))
    axes = axgrid.ravel()

    _plot_collapse(axes[0], "scaling_collapse.csv", "Whole route: collapses", "a")
    _plot_collapse(axes[1], "scaling_collapse_interior.csv",
                   "Interior only: does not", "b")

    # --- (c) the actual evidence: observed improvement vs permutation null
    ax = axes[2]
    for tag, col, lab, obs, p, ytxt in [
            ("_interior", CORR, "interior", 1.92, "$p=0.82$", .44),
            ("", RAW, "whole route", 6.56, "$p<0.017$", .70)]:
        null = np.loadtxt(DATA / "week3" / f"scaling_collapse{tag}_null.txt")
        ax.hist(null, bins=18, color=col, alpha=.30, lw=0, density=True,
                label=f"{lab} null")
        ax.axvline(obs, color=col, lw=1.3)
        ax.annotate(f"{lab}\n{p}", xy=(obs, ytxt),
                    xycoords=("data", "axes fraction"),
                    xytext=(-3 if not tag else 3, 0), textcoords="offset points",
                    fontsize=4.6, color=col,
                    ha="right" if not tag else "left", va="top")
    ax.set_xlim(1.0, 9.2)
    ax.set_xlabel("collapse improvement ratio")
    ax.set_ylabel("null density")
    titled(ax, "c", "Against the control")
    ax.legend(loc="upper right", fontsize=4.4, borderpad=.2, labelspacing=.16)

    # --- (d) harmonics against a smooth-unimodal reference
    ax = axes[3]
    dh = pd.read_csv(DATA / "week1_2" / "daido_harmonics.csv")
    cells = pd.read_csv(DATA / "week1_2" / "cells_scored_arc_uniform.csv")
    for pdname, col, mk in [("arc", CORR, "o"), ("time", GREEN, "s")]:
        s = dh[dh.phase_def == pdname].sort_values("m")
        ax.plot(s.m, s.mean_r_excess, mk + "-", color=col, ms=3.4, lw=1.2,
                label=f"{pdname} phase")

    # Reference: a single smooth (von Mises) cluster, sampled at the observed N
    # distribution and put through the identical same-N correction, with the
    # concentration tuned so its *corrected* |Z_1| matches the observed one. That
    # matching is what makes m = 2, 3 an apples-to-apples comparison: a smooth
    # unimodal density must fall away with m, whereas compact clumps do not.
    ns = cells.N.to_numpy(int)
    tab = nulls.uniform_null(int(ns.max()), draws=20_000, seed=11)
    mu = dict(zip(tab.table.N, tab.table.null_mean))
    mu_arr = np.array([mu[n] for n in ns])

    def corrected_Zm(kappa: float, m: int, seed: int = 3) -> float:
        rng = np.random.default_rng(seed)
        r = np.array([abs(np.mean(np.exp(1j * m * rng.vonmises(0.0, kappa, n))))
                      for n in ns])
        return float(np.mean((r - mu_arr) / (1 - mu_arr)))

    target = float(dh[(dh.phase_def == "arc") & (dh.m == 1)].mean_r_excess.iloc[0])
    lo, hi = 1e-3, 12.0
    for _ in range(28):                      # bisection on a monotone function
        mid = .5 * (lo + hi)
        if corrected_Zm(mid, 1) < target:
            lo = mid
        else:
            hi = mid
    kappa = .5 * (lo + hi)
    ref = [corrected_Zm(kappa, m) for m in (1, 2, 3)]
    ax.plot([1, 2, 3], ref, "^--", color=GREY, ms=3.2, lw=1.1,
            label="smooth cluster")

    ax.set_xticks([1, 2, 3]); ax.set_xlim(.8, 3.25); ax.set_ylim(-.03, .40)
    ax.set_xlabel("harmonic $m$")
    ax.set_ylabel(r"corrected $|Z_m|$")
    titled(ax, "d", "Flat in $m$: compact clumps")
    ax.legend(loc="upper left", fontsize=4.4, ncol=1, labelspacing=.16,
              borderpad=.2, handlelength=1.2, handletextpad=.35)

    fig.tight_layout(w_pad=1.0, h_pad=1.1)
    fig.savefig(FIG / "f4_collapse_harmonics.pdf")
    plt.close(fig)
    ctx.__exit__(None, None, None)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    fig_observable(); print("f1_observable.pdf")
    fig_route_position(); print("f2_route_position.pdf")
    fig_theory(); print("f3_theory.pdf")
    fig_collapse_and_harmonics(); print("f4_collapse_harmonics.pdf")
    print(f"wrote {FIG}")


if __name__ == "__main__":
    main()
