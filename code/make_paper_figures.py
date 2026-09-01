#!/usr/bin/env python3
"""Recreate the five figures in the NetMob 2026 camera-ready paper.

The plotted values are frozen below so the public repository does not need to
ship analysis CSV or JSON files. The full raw-data analysis remains separate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


INK = "#1A1A1A"
RAW = "#B0413E"
CORRECTED = "#1F4E79"
GOLD = "#D99A2B"
GREEN = "#2E7D5B"
GREY = "#9A9A9A"


FLEET_SIZE = {
    "N": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    "observed_median": [0.755783, 0.526703, 0.494672, 0.502011, 0.458131,
                        0.458686, 0.463928, 0.429237, 0.438156, 0.437817,
                        0.438060, 0.436172, 0.413319, 0.425791, 0.421895],
    "observed_q25": [0.418264, 0.352902, 0.361935, 0.356563, 0.350979,
                     0.340372, 0.357934, 0.319202, 0.352231, 0.359963,
                     0.299689, 0.327310, 0.292207, 0.281509, 0.299722],
    "observed_q75": [0.963496, 0.738213, 0.627707, 0.611730, 0.576127,
                     0.571839, 0.571417, 0.555499, 0.550236, 0.542930,
                     0.549066, 0.555312, 0.525300, 0.564862, 0.577178],
    "null_mean": [0.635980, 0.527186, 0.448053, 0.400409, 0.366899,
                  0.335756, 0.315213, 0.297090, 0.280904, 0.267392,
                  0.257252, 0.245463, 0.236785, 0.228977, 0.222307],
    "null_q025": [0.043482, 0.124044, 0.069788, 0.073617, 0.065545,
                  0.061770, 0.055317, 0.054715, 0.050548, 0.050745,
                  0.046288, 0.043442, 0.043541, 0.041658, 0.040989],
    "null_q975": [0.999060, 0.969699, 0.900887, 0.821596, 0.758663,
                  0.695102, 0.657015, 0.618430, 0.591539, 0.559829,
                  0.544305, 0.519415, 0.503627, 0.488261, 0.473214],
}

LOAD_RESPONSE_POINTS = {
    "x": [0.015105, 2.015493, 5.116658, 9.313408, 15.036222, 21.410906,
          28.504736, 34.238883, 38.747722, 42.839621, 47.021492, 51.524835,
          56.998330, 64.678216, 74.641654, 88.043058, 111.191136, 231.662431],
    "y": [0.120678, 0.172369, 0.172831, 0.131555, 0.142874, 0.146075,
          0.177934, 0.159062, 0.149952, 0.168516, 0.144795, 0.147211,
          0.149336, 0.142299, 0.143881, 0.121988, 0.155666, 0.250651],
    "low": [0.062107, 0.128345, 0.131417, 0.096573, 0.110224, 0.115119,
            0.150469, 0.135435, 0.127606, 0.144293, 0.119331, 0.122963,
            0.124937, 0.118838, 0.116567, 0.094862, 0.122274, 0.204266],
    "high": [0.179249, 0.216394, 0.214244, 0.166538, 0.175525, 0.177030,
             0.205398, 0.182689, 0.172297, 0.192740, 0.170259, 0.171459,
             0.173736, 0.165760, 0.171195, 0.149113, 0.189057, 0.297037],
}

POSITION_COUNTS = [
    9770, 3370, 1358, 1241, 907, 766, 663, 627, 711, 720,
    673, 1367, 756, 624, 510, 730, 585, 563, 609, 667,
    640, 603, 694, 981, 852, 588, 790, 649, 612, 612,
    696, 619, 541, 547, 603, 562, 980, 840, 685, 632,
    630, 599, 640, 662, 822, 909, 1086, 554, 623, 708,
    641, 780, 763, 834, 833, 1234, 2130, 3424, 3305, 7346,
]

ZONE_ORDER = [0.9973866001, 0.9186338088, -0.1887380982]
ZONE_LOAD = {
    "estimate": [0.0011115661, -0.0003754656, 0.0000447741, 0.0375847570],
    "low": [-0.0246022113, -0.0047635876, -0.0000930317, 0.0109678698],
    "high": [0.0268253435, 0.0040126563, 0.0001825799, 0.0642016441],
}


def set_style() -> None:
    mpl.rcParams.update({
        "font.size": 9.0,
        "axes.labelsize": 9.0,
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
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0.17, 0.18, 0.78, 0.77])
    return fig, ax


def save(fig, figure_dir: Path, stem: str) -> None:
    fig.savefig(figure_dir / f"{stem}.pdf")
    fig.savefig(figure_dir / f"{stem}.png", dpi=240)
    plt.close(fig)
    print(f"{stem}.pdf and {stem}.png")


def figure_1_finite_size(figures: Path) -> None:
    n = np.asarray(FLEET_SIZE["N"], dtype=float)
    fig, ax = one_axis()
    ax.fill_between(n, FLEET_SIZE["null_q025"], FLEET_SIZE["null_q975"],
                    color=GREY, alpha=0.32, linewidth=0, label="null 95% band")
    ax.plot(n, FLEET_SIZE["null_mean"], color=INK, linewidth=1.6,
            label="same-$N$ null mean")
    ax.fill_between(n, FLEET_SIZE["observed_q25"], FLEET_SIZE["observed_q75"],
                    color=RAW, alpha=0.22, linewidth=0)
    ax.plot(n, FLEET_SIZE["observed_median"], color=RAW, linewidth=1.7,
            label="observed median (IQR)")
    ax.set(xlim=(2, 16), ylim=(0, 1),
           xlabel="active fleet size $N$ (buses per line-direction-hour)",
           ylabel=r"raw first-order coherence $r_1$")
    ax.legend(loc="upper right")
    save(fig, figures, "fig1_finite_size")


def figure_2_load_response(figures: Path) -> None:
    x = np.linspace(0.0, 312.93, 300)
    change_point = 135.74
    piecewise = 0.1541519826 - 0.0001127155 * x + 0.0013185611 * np.maximum(0, x - change_point)
    linear = 0.1204760698 + 0.0006814720 * x
    sigmoid = 0.1411128589 + 0.0832495425 / (1 + np.exp(-(x - 101.76) / 24.8141447368))

    point_x = np.asarray(LOAD_RESPONSE_POINTS["x"])
    point_y = np.asarray(LOAD_RESPONSE_POINTS["y"])
    yerr = np.vstack([
        point_y - np.asarray(LOAD_RESPONSE_POINTS["low"]),
        np.asarray(LOAD_RESPONSE_POINTS["high"]) - point_y,
    ])
    fig, ax = one_axis(4.7, 3.2)
    ax.errorbar(point_x, point_y, yerr=yerr, fmt="o", markersize=3.8,
                color=CORRECTED, ecolor="#70879a", capsize=2,
                label="adjusted bin mean (95% CI)")
    ax.plot(x, piecewise, color=RAW, linewidth=1.8, label="piecewise ($\\Delta$AIC 0)")
    ax.plot(x, linear, color=GREY, linewidth=1.2, linestyle="--",
            label="linear ($\\Delta$AIC 26.4)")
    ax.plot(x, sigmoid, color=GREEN, linewidth=1.2, linestyle=":",
            label="sigmoid ($\\Delta$AIC 88.2)")
    ax.axvline(change_point, color=RAW, linewidth=0.9, linestyle="--")
    ax.text(change_point, ax.get_ylim()[1], "  descriptive scale 136",
            color=RAW, verticalalignment="top", fontsize=7.5)
    ax.set(xlim=(0, 312.93),
           xlabel="realized route-hour boardings per active bus $\\lambda$",
           ylabel=r"adjusted excess order $r_{\rm exc}$")
    ax.legend(loc="best", fontsize=7)
    save(fig, figures, "fig2_load_response")


def figure_3_position_distribution(figures: Path) -> None:
    edges = np.linspace(0, 1, len(POSITION_COUNTS) + 1)
    fig, ax = one_axis()
    ax.bar(edges[:-1], POSITION_COUNTS, width=np.diff(edges), align="edge",
           color=CORRECTED, linewidth=0, alpha=0.85)
    ax.axvspan(0, 0.10, color=RAW, alpha=0.20, linewidth=0)
    ax.axvspan(0.90, 1, color=RAW, alpha=0.20, linewidth=0)
    ax.text(0.5, 0.82, "51.4% within outer deciles", transform=ax.transAxes,
            horizontalalignment="center", color=RAW)
    ax.set(xlim=(0, 1),
           xlabel="normalized route position $s/L$ (0: origin; 1: destination)",
           ylabel="active bus-hours (count)")
    save(fig, figures, "fig3_position_distribution")


def figure_4_zone_order(figures: Path) -> None:
    fig, ax = one_axis(4.2, 2.85)
    bars = ax.bar(range(3), ZONE_ORDER, color=[RAW, GOLD, CORRECTED], width=0.64)
    for bar, value in zip(bars, ZONE_ORDER):
        offset = 0.05 if value >= 0 else -0.09
        ax.text(bar.get_x() + bar.get_width() / 2, value + offset, f"{value:+.3f}",
                horizontalalignment="center")
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xticks(range(3), ["terminal\n<2%", "near\n2--10%", "interior\n10--90%"])
    ax.set(ylabel=r"mean within-zone excess order $r_{\rm exc}$")
    save(fig, figures, "fig4_zone_order")


def figure_5_load_effect_by_zone(figures: Path) -> None:
    labels = ["interior", "near terminal", "terminal", "whole route"]
    estimate = np.asarray(ZONE_LOAD["estimate"])
    low = np.asarray(ZONE_LOAD["low"])
    high = np.asarray(ZONE_LOAD["high"])
    fig, ax = one_axis(4.5, 3.0)
    for index, label in enumerate(labels):
        color = RAW if label == "whole route" else CORRECTED
        ax.errorbar(estimate[index], index,
                    xerr=[[estimate[index] - low[index]], [high[index] - estimate[index]]],
                    fmt="o", markersize=4.5, color=color, capsize=2)
    ax.axvline(0, color=INK, linewidth=0.8)
    ax.set_yticks(range(len(labels)), labels)
    ax.set(xlabel=r"coefficient of standardized realized load on $r_{\rm exc}$",
           ylabel="analysis region")
    save(fig, figures, "fig5_load_effect_by_zone")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    repository = Path(__file__).resolve().parents[1]
    parser.add_argument("--figure-dir", type=Path,
                        default=repository / "output" / "figures")
    args = parser.parse_args()
    figures = args.figure_dir.resolve()
    figures.mkdir(parents=True, exist_ok=True)
    set_style()
    figure_1_finite_size(figures)
    figure_2_load_response(figures)
    figure_3_position_distribution(figures)
    figure_4_zone_order(figures)
    figure_5_load_effect_by_zone(figures)


if __name__ == "__main__":
    main()
