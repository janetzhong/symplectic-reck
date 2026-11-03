"""Plot N=10 residuals and paired tomography-minus-in-situ differences."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter

from ..io_utils import ROOT

CURVES = [
    ("tomo_10k", "tomography (finite shots)", "tab:red", "o-"),
    ("insitu_equalq", "in-situ (finite shots)", "tab:blue", "D-"),
    ("tomo_exact", "tomography (exact)", "tab:red", "o:"),
    ("insitu_exact", "in-situ (exact)", "tab:blue", "D:"),
]
EPS_MIN, EPS_MAX = 0.001, 0.1


def displayed_eps(rows):
    return sorted({row["eps"] for row in rows if EPS_MIN <= row["eps"] <= EPS_MAX})


def residual_stats(rows, eps_values, key):
    """Compute residual quartiles. Saved values already include 1/sqrt(2N)."""
    stats = []
    for eps in eps_values:
        values = [row[key] for row in rows if row["eps"] == eps and row[key] is not None]
        stats.append(np.quantile(values, [0.5, 0.25, 0.75]) if values else [np.nan] * 3)
    return np.asarray(stats).T


def paired_difference_stats(rows, eps_values):
    """Subtract in-situ from tomography within each trial, then compute quartiles."""
    stats = []
    for eps in eps_values:
        differences = [
            row["tomo_10k"] - row["insitu_equalq"]
            for row in rows
            if row["eps"] == eps
            and row["tomo_10k"] is not None
            and row["insitu_equalq"] is not None
        ]
        if not differences:
            raise ValueError(f"No matched finite-shot trials at epsilon={eps:g}")
        stats.append(np.quantile(differences, [0.5, 0.25, 0.75]))
    return np.asarray(stats).T


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results/insitu_vs_tomog_with_gate_errors")
    args = parser.parse_args()
    rows = json.loads((args.results_dir / "N10/trials.json").read_text())["rows"]
    eps_values = displayed_eps(rows)
    if not eps_values:
        parser.error("No N=10 trials within the displayed epsilon range")
    fig, (left, right) = plt.subplots(1, 2, figsize=(13.6, 5.5))
    for key, label, color, style in CURVES:
        med, lo, hi = residual_stats(rows, eps_values, key)
        left.errorbar(eps_values, med, yerr=[med - lo, hi - med], fmt=style,
                      color=color, label=label, linewidth=2, elinewidth=2, markersize=5.8, capsize=3)
    left.set_yscale("log")
    left.set_ylabel("residual error $\\mathcal{E}$", fontsize=20)

    median, q25, q75 = paired_difference_stats(rows, eps_values)
    right.errorbar(eps_values, median, yerr=[median - q25, q75 - median],
                   fmt="o-", color="black", label="tomography − in-situ (finite shots)",
                   linewidth=2, elinewidth=2, markersize=5.8, capsize=3)
    right.axhline(0, color="0.4", linestyle="--", linewidth=1.2)
    right.set_yscale("linear")
    right.set_ylim(min(-0.02, float(q25.min()) * 1.5), float(q75.max()) * 1.15)
    right.set_ylabel("$\\mathcal{E}_{\\mathrm{tomog}}-\\mathcal{E}_{\\mathrm{in\\!-\\!situ}}$", fontsize=20)
    right.yaxis.set_minor_locator(AutoMinorLocator())
    for ax in (left, right):
        ax.set_xscale("log")
        ax.set_xlim(EPS_MIN / 1.15, EPS_MAX * 1.15)
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10)))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.grid(False, which="both")
        ax.set_xlabel("maximum fixed gate-parameter error $\\epsilon_{\\max}$", fontsize=18)
        ax.tick_params(which="major", labelsize=16, length=5)
        ax.tick_params(which="minor", length=3)
    left.legend(loc="lower center", ncol=2, frameon=False, fontsize=15,
                bbox_to_anchor=(0.5, 1.02), columnspacing=1.2)
    right.legend(loc="lower center", frameon=False, fontsize=15,
                 bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(w_pad=2.5)
    out = args.results_dir / "insitu_vs_tomog_with_gate_errors"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"wrote {out}.pdf")


if __name__ == "__main__":
    main()
