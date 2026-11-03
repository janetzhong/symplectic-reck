"""Plot representative output matrices and loss curves for the feedback methods."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..io_utils import ROOT

METHODS = [("measured_closed_form", "measured\nclosed-form"), ("finite_diff", "finite\ndifference"),
           ("dither_lockin", "dithering"), ("genetic_algorithm", "genetic\nalgorithm")]
FEEDBACK = [("finite_diff", "finite difference"), ("dither_lockin", "dithering"),
            ("genetic_algorithm", "genetic algorithm")]


def plot_heatmaps(results_dir, ns, q, matrix):
    fig, axes = plt.subplots(len(ns), len(METHODS), figsize=(1.15 * len(METHODS), 3.75))
    fig.subplots_adjust(left=0.125, right=0.865, top=0.85, bottom=0.015, wspace=0.095, hspace=0.12)
    axes = np.asarray(axes).reshape(len(ns), len(METHODS))
    mats, errors = {}, {}
    for N in ns:
        with np.load(results_dir / f"N{N}/median_matrices.npz") as data:
            for method, _ in METHODS:
                raw = data[f"{method}__{q}__{matrix}"]
                mats[N, method] = np.abs(raw)
                error_matrix = raw - np.eye(2 * N) if matrix == "final" else raw
                errors[N, method] = np.linalg.norm(error_matrix) / np.sqrt(2 * N)
    vmax = max(1e-12, max(m.max() for m in mats.values()))
    for row, N in enumerate(ns):
        for col, (method, label) in enumerate(METHODS):
            ax = axes[row, col]
            image = ax.imshow(mats[N, method], cmap="viridis", vmin=0.0, vmax=vmax, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            coefficient, exponent = f"{errors[N, method]:.2e}".split("e")
            error_text = rf"$\mathcal{{E}}={coefficient}\!\times\!10^{{{int(exponent)}}}$"
            ax.text(0.5, 1.015, error_text, transform=ax.transAxes, ha="center", va="bottom", fontsize=5.6)
            if row == 0:
                title = label if "\n" in label else label + "\n"  # pad single-line titles so they top-align
                ax.set_title(title, fontsize=7.2, pad=15, linespacing=1.05)
            if col == 0:
                ax.set_ylabel(f"N={N}", fontsize=7.5, rotation=0, labelpad=16, va="center")
    cbar = fig.colorbar(image, cax=fig.add_axes([0.885, 0.12, 0.012, 0.68]), label="$|S_\\theta S_{\\mathrm{unk}}|$")
    cbar.ax.tick_params(labelsize=7.5, length=1.5, pad=1)
    cbar.ax.yaxis.label.set_size(7.5)
    out = results_dir / "finite_shots_heatmaps"
    fig.savefig(out.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}.pdf")


def plot_loss_curves(results_dir, N):
    """Plot median losses and quartiles for each shot count, one panel per feedback method."""
    path = results_dir / f"N{N}/loss_history.npz"
    if not path.exists():
        print(f"no loss histories for N={N}")
        return
    curves = {}  # (method, shots) -> list of two-mode loss curves over all trials and gates
    with np.load(path) as data:
        for key in data.files:
            method, shots, _trial, kind, _gate = key.split("__")
            if kind == "loss_history":
                curves.setdefault((method, shots), []).append(data[key])
    panels = [(m, t) for m, t in FEEDBACK if any(key[0] == m for key in curves)]
    fig, axes = plt.subplots(1, len(panels), figsize=(8 * len(panels), 6.8), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, (method, title) in zip(axes, panels):
        items = sorted(((s, c) for (m, s), c in curves.items() if m == method), key=lambda kv: float(kv[0]))
        for shots, histories in items:
            T = min(len(c) for c in histories)
            arr = np.array([c[:T] for c in histories])
            x = np.arange(1, T + 1)
            (line,) = ax.semilogy(x, np.median(arr, 0), label=f"q={shots}", linewidth=2.6)
            ax.fill_between(x, np.quantile(arr, 0.25, 0), np.quantile(arr, 0.75, 0), color=line.get_color(), alpha=0.18, linewidth=0)
        ax.set_title(title, fontsize=26)
        ax.set_xlabel("iterations", fontsize=24)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(loc="lower left", fontsize=20)
        ax.tick_params(labelsize=20)
    axes[0].set_ylabel("nulling step loss $\\|M_{j,k}\\|_F^2$", fontsize=24)
    fig.tight_layout()
    out = (results_dir if N == 3 else results_dir / f"N{N}") / "finite_shots_loss_curves"
    fig.savefig(out.with_suffix(".png"), dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    plt.close(fig)
    print(f"wrote {out}.pdf")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results/finite_shots")
    parser.add_argument("--Ns", default="3,15,30")
    parser.add_argument("--q", type=int, default=1000)
    parser.add_argument("--matrix", choices=["final", "error"], default="final")
    args = parser.parse_args()
    ns = [int(n) for n in args.Ns.split(",")]
    plot_heatmaps(args.results_dir, ns, args.q, args.matrix)
    for N in ns:
        plot_loss_curves(args.results_dir, N)


if __name__ == "__main__":
    main()
