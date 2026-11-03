"""Plot local conditioning, peak gate squeezing, and the final residual for each sweep."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..io_utils import ROOT

plt.rcParams.update({"font.size": 16, "axes.labelsize": 17, "xtick.labelsize": 15, "ytick.labelsize": 15})
SWEEPS = [("N", "modes $N$", False), ("alpha", "path parameter $\\alpha$", True),
          ("db", "squeezing $s_{\\max}$ (dB)", False)]


def quartiles(values):
    return np.quantile(values, [0.5, 0.25, 0.75])


def errorbar(ax, x, stats, **style):
    med, lo, hi = np.asarray(stats).T
    ax.errorbar(x, med, yerr=[med - lo, hi - med], ms=5, lw=1.8, capsize=3, **style)


def plot(rows, top_column, top_label, top_log, out):
    """Two rows of panels: top_column over each sweep, then the residual at eps=1e-8."""
    fig, axes = plt.subplots(2, 3, figsize=(13.6, 6.0))
    for j, (sweep, xlabel, logx) in enumerate(SWEEPS):
        values = sorted({r["value"] for r in rows if r["sweep"] == sweep})
        groups = [[r for r in rows if r["sweep"] == sweep and r["value"] == v] for v in values]
        errorbar(axes[0, j], values, [quartiles([r[top_column] for r in g]) for g in groups], fmt="^-", color="#d62728")
        errorbar(axes[1, j], values, [quartiles([r["residual_e8"] for r in g]) for g in groups], fmt="o-", color="#ff7f0e")
        for ax in axes[:, j]:
            if logx:
                ax.set_xscale("log")
                ax.invert_xaxis()
            ax.grid(False, which="both")
        if top_log:
            axes[0, j].set_yscale("log")
        else:
            axes[0, j].set_ylim(bottom=0)
        axes[1, j].set_yscale("log")
        axes[1, j].set_xlabel(xlabel, fontsize=17)
    axes[0, 0].set_ylabel(top_label)
    axes[1, 0].set_ylabel("residual error $\\mathcal{E}$")
    fig.tight_layout(h_pad=1.3)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results/conditioning")
    args = parser.parse_args()
    with (args.results_dir / "trials.csv").open() as stream:
        rows = [{k: (v if k == "sweep" else float(v)) for k, v in r.items()} for r in csv.DictReader(stream)]
    plot(rows, "kappa_max", "condition score\n$\\kappa_{\\max}$", True, args.results_dir / "nulling_conditioning")
    plot(rows, "max_gate_db", "max gate\nsqueezing (dB)", False, args.results_dir / "nulling_conditioning_gate_squeezing")


if __name__ == "__main__":
    main()
