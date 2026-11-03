"""Animate four-mode nulling: the circuit, the output matrix |M|, and the losses.

Six two-mode gates train one after another, then the four single-mode gates
train together. Matrix frames are sampled from the actual Adam iterations;
only the red outline is interpolated between stages. Requires ffmpeg on PATH."""

import argparse
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import FancyBboxPatch, Rectangle

from ..io_utils import ROOT, save_json
from ..matrix_utils import normalized_residual
from ..run_nulling_sequence import run_autodiff_sequence
from ..targets import random_bm_target

COLORS = ["#FFCD7D", "#FFBAFF", "#73E5F8", "#96E4A0"]  # fill per layer
EDGES = ["#8B5C0E", "#8E3290", "#14738A", "#2E7235"]  # edge per layer
POSITIONS = [(2.3, 3.5), (4.6, 2.5), (6.9, 1.5), (6.9, 3.5), (9.2, 2.5), (11.5, 3.5)]  # two-mode gate centres
RED = "#D62728"


def circuit(ax):
    """Draw the mesh and return the gate boxes, the nulled/not-nulled marks, and the moving outline."""
    ax.set(xlim=(-4.0, 16.6), ylim=(0.3, 4.7))
    ax.set_aspect("auto")
    ax.axis("off")
    for mode in range(1, 5):
        ax.plot([-2.7, 15], [mode, mode], color="#bbbbbb", lw=1, zorder=0)
        color = EDGES[mode - 1] if mode < 4 else "#333333"
        ax.text(-2.8, mode, f"In {mode}", ha="right", va="center", color=color, weight="bold", fontsize=21)
        ax.text(15.05, mode, f"Out {mode}", va="center", color=color, weight="bold", fontsize=21)
    ax.add_patch(FancyBboxPatch((-1.8, 0.65), 2.8, 3.7, boxstyle="round,pad=0.02", fc="#eaeaea", ec="#5c5c5c"))
    ax.text(-0.4, 2.5, r"$\mathbf{S}_{\rm unk}$", ha="center", va="center", fontsize=27)
    boxes, marks = [], []
    for k, (x, y) in enumerate(POSITIONS):
        layer = 0 if k < 3 else 1 if k < 5 else 2
        boxes.append((x - 0.835, y - 0.76, 1.67, 1.52))
        ax.add_patch(FancyBboxPatch((x - 0.775, y - 0.7), 1.55, 1.4, boxstyle="round,pad=0.02",
                                    fc=COLORS[layer], ec=EDGES[layer]))
        ax.text(x, y + 0.02, r"$\mathbf{S}$", ha="center", va="center", fontsize=20)
        ax.text(x - 0.675, y + 0.58, f"$L_{layer + 1}$", va="top", fontsize=18)
        ax.text(x, y - 0.59, f"step {k + 1}", ha="center", color="#8b1515", fontsize=18)
        cross, = ax.plot(x + 0.925, y + 0.5, marker="x", color=EDGES[layer], ms=7, visible=False)
        tick, = ax.plot([x + 0.845, x + 0.915, x + 1.025], [y - 0.45, y - 0.55, y - 0.39],
                        color=EDGES[layer], lw=1.3, visible=False)
        marks.append((cross, tick))
    for mode in range(1, 5):
        x, y = 13.8, mode
        boxes.append((x - 0.835, y - 0.5, 1.67, 1.0))
        ax.add_patch(FancyBboxPatch((x - 0.775, y - 0.45), 1.55, 0.9, boxstyle="round,pad=0.01",
                                    fc=COLORS[3], ec=EDGES[3]))
        ax.text(x, y, r"$\mathbf{S}$", ha="center", va="center", fontsize=18)
        ax.text(x - 0.695, y + 0.39, "$L_4$", va="top", fontsize=14)
        ax.text(x, y - 0.35, "step 7", ha="center", fontsize=14, color="#8b1515")
    highlight = Rectangle(boxes[0][:2], *boxes[0][2:], ec=RED, fc="none", lw=2.6, zorder=10)
    ax.add_patch(highlight)
    return boxes, marks, highlight


def set_fonts():
    """Use Latin Modern Sans if a TeX installation provides it, otherwise DejaVu Sans."""
    family = "DejaVu Sans"
    for name in ("lmsans10-regular.otf", "lmsans10-bold.otf"):
        try:
            path = subprocess.check_output(["kpsewhich", name], text=True).strip()
        except (FileNotFoundError, subprocess.CalledProcessError):
            continue
        if path:
            font_manager.fontManager.addfont(path)
            family = font_manager.FontProperties(fname=path).get_name()
    plt.rcParams.update({"font.family": family, "font.size": 18, "mathtext.fontset": "cm"})


def scientific(value):
    mantissa, exponent = f"{value:.2e}".split("e")
    return mantissa + r"\times 10^{" + str(int(exponent)) + "}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/four_mode_nulling_animation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps-per-gate", type=int, default=600)
    parser.add_argument("--singlemode-steps", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--frames-per-stage", type=int, default=36)
    args = parser.parse_args()
    if min(args.steps_per_gate, args.singlemode_steps, args.fps, args.frames_per_stage) < 1:
        parser.error("iteration counts, fps and frames-per-stage must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    target, _ = random_bm_target(4, seed=args.seed, r_min=0.3, r_max=0.4)
    trace = []  # (stage, iteration, matrix, loss, gradient norm) before each update

    def observe(stage, iteration, matrix, loss, gradient):
        trace.append((stage, iteration, matrix.numpy().copy(), loss, gradient))

    result = run_autodiff_sequence(target, 4, args.steps_per_gate, 0.02, args.singlemode_steps, 0.05,
                                   trace_callback=observe, parallel_single_modes=True)
    final = result["final_matrix"].numpy()
    residual = normalized_residual(result["final_matrix"])
    print(f"Autodiff finished: residual = {residual:.3e}", flush=True)
    matrices = np.stack([t[2] for t in trace])
    np.savez_compressed(args.out_dir / "optimization_trace.npz", matrices=matrices,
                        stages=[t[0] for t in trace], iterations=[t[1] for t in trace],
                        losses=[t[3] for t in trace], gradient_norms=[t[4] for t in trace], final_matrix=final)

    # Frames are (trace index, previous stage or None, outline interpolation weight); index -1 is the final result.
    frames = [(0, None, 1.0)] * args.fps
    for stage in range(7):
        indices = [i for i, t in enumerate(trace) if t[0] == stage]
        if stage:
            frames.extend((indices[0], stage - 1, float(a)) for a in np.linspace(0, 1, 8))
        frames.extend((indices[int(i)], None, 1.0) for i in np.linspace(0, len(indices) - 1, args.frames_per_stage))
        frames.extend([(indices[-1], None, 1.0)] * 6)
    frames.extend([(-1, None, 1.0)] * (args.fps * 3))

    set_fonts()
    fig = plt.figure(figsize=(20.7, 6.2), facecolor="white")
    ax = fig.add_axes([0.01, 0.235, 0.68, 0.605])
    boxes, marks, highlight = circuit(ax)
    boxes[6] = (12.965, 0.5, 1.67, 4.0)  # the final stage outlines all four single-mode gates
    hm = fig.add_axes([0.7546, 0.24, 0.17826, 0.59516])
    vmax = max(1.0, float(np.abs(matrices).max()))
    heat = hm.imshow(np.abs(matrices[0]), cmap="viridis", vmin=0, vmax=vmax, interpolation="nearest")
    labels = [f"${q}_{m}$" for m in range(1, 5) for q in ("x", "p")]
    hm.set_xticks(range(8), labels, fontsize=25)
    hm.set_yticks(range(8), labels, fontsize=25)
    cax = fig.add_axes([0.9433, 0.24, 0.00696, 0.59516])
    fig.colorbar(heat, cax=cax).ax.tick_params(labelsize=18)
    hm.set_title(r"$|M|=|S_\theta S_{\mathrm{unk}}|$", fontsize=22, pad=18)
    blockbox = Rectangle((-0.5, 5.5), 2, 2, ec=RED, fc="none", lw=2.5)
    hm.add_patch(blockbox)
    diagonal_boxes = []
    for mode in range(1, 4):
        box = Rectangle((2 * mode - 0.5, 2 * mode - 0.5), 2, 2, ec=RED, fc="none", lw=2.5, visible=False)
        hm.add_patch(box)
        diagonal_boxes.append(box)
    metrics = fig.text(0.21, 0.095, "", ha="center", fontsize=27)
    residual_metric = fig.text(0.56, 0.095, "", ha="center", fontsize=27)

    def block_corner(stage):
        """Lower-left corner of the highlighted matrix block for a stage."""
        if stage < 6:
            step = result["schedule"][stage]
            return np.array([2 * step.probe - 0.5, 2 * step.target - 0.5])
        return np.array([-0.5, -0.5])

    def draw(frame):
        idx, previous, alpha = frame
        done = idx == -1
        stage, iteration, matrix, loss, grad = trace[idx]
        if done:
            matrix = final
            loss = sum(np.sum((matrix[2 * m:2 * m + 2, 2 * m:2 * m + 2] - np.eye(2)) ** 2) for m in range(4))
        heat.set_data(np.abs(matrix))
        highlight.set_visible(not done)
        blockbox.set_visible(not done)
        for box in diagonal_boxes:
            box.set_visible(not done and stage == 6 and (previous is None or alpha == 1.0))
        rect = np.array(boxes[stage])
        xy = block_corner(stage)
        if previous is not None:
            rect = (1 - alpha) * np.array(boxes[previous]) + alpha * rect
            xy = (1 - alpha) * block_corner(previous) + alpha * xy
        highlight.set_bounds(*rect)
        blockbox.set_xy(xy)
        for k, pair in enumerate(marks):
            for mark in pair:
                mark.set_visible(done or stage > k)
        if stage < 6:
            step = result["schedule"][stage]
            local_block = matrix[2 * step.target:2 * step.target + 2, 2 * step.probe:2 * step.probe + 2]
            shown_loss = float(np.sum(local_block ** 2))
            formula = rf"Loss $=\|M_{{{step.target + 1},{step.probe + 1}}}\|_F^2"
        else:
            shown_loss = loss
            formula = r"Loss $=\sum_{m=1}^{4}\|M_{m,m}-I_2\|_F^2"
        metrics.set_text(formula + " = " + scientific(shown_loss) + "$,")
        current_residual = np.linalg.norm(matrix - np.eye(8)) / np.sqrt(8)
        residual_metric.set_text(r"Residual $=\|M-I_8\|_F/\sqrt{8} = " + scientific(current_residual) + "$")

    movie = args.out_dir / "fourmode_nulling_animation.mp4"
    writer = FFMpegWriter(fps=args.fps, codec="libx264", bitrate=2400,
                          extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    with writer.saving(fig, str(movie), dpi=100):
        for frame in frames:
            draw(frame)
            writer.grab_frame()
    draw((next(i for i, t in enumerate(trace) if t[0] == 3 and t[1] >= args.steps_per_gate // 3), None, 1.0))
    fig.savefig(args.out_dir / "preview.png", dpi=120)
    draw((next(i for i, t in enumerate(trace) if t[0] == 6 and t[1] >= args.singlemode_steps // 10), None, 1.0))
    fig.savefig(args.out_dir / "parallel_single_modes.png", dpi=120)
    draw((-1, None, 1.0))
    fig.savefig(args.out_dir / "final_frame.png", dpi=120)
    plt.close(fig)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(movie), "-filter_complex",
                    "fps=15,scale=1200:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse",
                    "-loop", "0", str(args.out_dir / "fourmode_nulling_animation.gif")], check=True)
    save_json(args.out_dir / "settings.json", dict(
        **vars(args), learning_rate=0.02, singlemode_learning_rate=0.05, parallel_single_modes=True,
        final_residual=residual, frame_count=len(frames), duration_seconds=len(frames) / args.fps,
        quadrature_order="xpxp", matrix_scale=[0, vmax],
    ))
    print(f"Saved MP4, GIF, previews and numerical trace in {args.out_dir}")


if __name__ == "__main__":
    main()
