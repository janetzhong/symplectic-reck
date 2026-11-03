"""Plot the matrix magnitude after each gate and mark the block being nulled."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from ..io_utils import ROOT


def red_box(ax, block_row, block_col, label):
    x0, y0 = 2 * block_col - 0.5, 2 * block_row - 0.5
    ax.add_patch(mpatches.Rectangle((x0, y0), 2, 2, linewidth=2.0, edgecolor="red", facecolor="none", zorder=3))
    ax.text(x0 + 1, y0 + 1, str(label), color="red", ha="center", va="center", fontsize=24, fontweight="bold", zorder=4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results/four_mode_nulling")
    args = parser.parse_args()
    settings = json.loads((args.results_dir / "settings.json").read_text())
    with np.load(args.results_dir / "matrices.npz") as arrays:
        snapshots = arrays["snapshots"]
    N = settings["N"]
    blocks = settings["nulled_blocks"]
    shown = list(snapshots[: len(blocks) + 1]) + [snapshots[-1]]
    fig, axes = plt.subplots(1, len(shown), figsize=(2.2 * len(shown), 2.2))
    label = f"$(x_1,p_1 \\ldots x_{N},p_{N})$"
    for idx, (ax, M) in enumerate(zip(axes, shown)):
        ax.imshow(np.abs(M), cmap="viridis", vmin=0.0, vmax=max(np.abs(M).max(), 1e-12))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel(label + "\ninput probe", fontsize=20)
        if idx == 0:
            ax.set_ylabel("output port\n" + label, fontsize=20)
        elif idx <= len(blocks):
            red_box(ax, *blocks[idx - 1], idx)
        else:
            for mode in range(N):
                red_box(ax, mode, mode, idx)
    fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.995, wspace=0.05)
    for ext in ("pdf", "png"):
        fig.savefig(args.results_dir / f"four_mode_nulling.{ext}", dpi=150, bbox_inches="tight", pad_inches=0.2)
    print(f"wrote {args.results_dir}/four_mode_nulling.pdf")


if __name__ == "__main__":
    main()
