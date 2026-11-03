"""Test mode separation under optical loss, using exact response measurements."""

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from ..io_utils import ROOT, save_csv, save_json
from ..nulling_analytical import analytical_nulling_gate, projected_single_mode_inverse
from ..nulling_gateorder import nulling_schedule
from ..targets import random_bm_target


def off_block_fraction(matrix):
    """||M - blockdiag(M)||_F / ||M||_F, with 2x2 mode blocks."""
    off = matrix.clone()
    for mode in range(matrix.shape[0] // 2):
        block = slice(2 * mode, 2 * mode + 2)
        off[block, block] = 0
    return float(torch.linalg.norm(off) / torch.linalg.norm(matrix))


@torch.no_grad()
def train_lossy_response(target, model="none", power_loss=0.0, final_layer=True,
                        loss_plan=None):
    """Null the current response using exact measurements, including optical loss.

    Uniform output loss commutes through the mesh, so we include it from the
    start. Single-mode corrections leave this attenuation in place.
    loss_plan can specify power loss for each gate and mode, including bypass
    modes, instead of using the same loss on every active pair."""
    if model not in ("none", "uniform", "per_gate"):
        raise ValueError(f"unknown loss model: {model}")
    if not math.isfinite(power_loss) or not 0 <= power_loss < 1:
        raise ValueError("power_loss must be finite and in [0, 1)")
    if model == "none" and power_loss != 0:
        raise ValueError("the no-loss model requires zero power_loss")
    N = target.shape[0] // 2
    schedule = nulling_schedule(N)
    if loss_plan is not None:
        loss_plan = torch.as_tensor(loss_plan, dtype=target.dtype, device=target.device)
        if (model != "per_gate" or loss_plan.shape != (len(schedule), N)
                or not torch.isfinite(loss_plan).all()
                or torch.any(loss_plan < 0) or torch.any(loss_plan >= 1)):
            raise ValueError("loss_plan requires per_gate model and finite [0,1) losses of shape (gates, modes)")
    amplitude = math.sqrt(1 - power_loss)
    M = target.clone()
    if model == "uniform":
        M *= amplitude
    for step in schedule:
        column = slice(2 * step.probe, 2 * step.probe + 2)
        rows = [2 * step.target, 2 * step.target + 1,
                2 * step.anchor, 2 * step.anchor + 1]
        gate = analytical_nulling_gate(M[rows[:2], column], M[rows[2:], column])
        M[rows, :] = gate @ M[rows, :]
        if model == "per_gate":
            if loss_plan is None:
                M[rows, :] *= amplitude
            else:
                factors = torch.sqrt(1 - loss_plan[step.index]).repeat_interleave(2)
                M *= factors[:, None]
    if final_layer:
        for mode in range(M.shape[0] // 2):
            block = slice(2 * mode, 2 * mode + 2)
            gate = projected_single_mode_inverse(M[block, block], context=f"mode {mode}")
            M[block, :] = gate @ M[block, :]
    return M


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", type=int, default=10)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=5040)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/loss_simulations")
    args = parser.parse_args()
    if args.N < 2 or args.trials < 1 or args.seed < 0:
        parser.error("N >= 2, trials >= 1 and seed >= 0 are required")
    torch.set_num_threads(1)
    cases = [("none", 0., True, False)]
    cases += [("uniform", loss, True, False) for loss in (.1, .2, .4)]
    cases += [("per_gate", loss, True, False) for loss in (.01, .02, .05, .1)]
    cases += [("per_gate", .05, False, False)]
    cases += [("per_gate", loss, True, True) for loss in (.01, .02, .05, .1)]
    targets = [random_bm_target(args.N, seed=args.seed + t, r_min=0., r_max=.5)[0]
               for t in range(args.trials)]
    rows, summary, matrices = [], [], []
    for model, loss, final_layer, padded in cases:
        values, responses, transmissions = [], [], []
        # Give bypass modes the same loss as the active pair at each step.
        plan = np.full((args.N * (args.N - 1) // 2, args.N), loss) if padded else None
        for trial, target in enumerate(targets):
            response = train_lossy_response(target, model, loss, final_layer, loss_plan=plan)
            value = off_block_fraction(response)
            transmission = float(torch.sum(response.square()) / response.shape[1])
            values.append(value)
            transmissions.append(transmission)
            responses.append(response.numpy())
            rows.append(dict(model=model, power_loss=loss, final_layer=final_layer,
                             padded=padded, trial=trial, seed=args.seed + trial, C_off=value,
                             coherent_power_transmission=transmission))
        matrices.append(responses)
        q25, median, q75 = np.percentile(values, [25, 50, 75])
        summary.append(dict(model=model, power_loss=loss, final_layer=final_layer, padded=padded,
                            mean=float(np.mean(values)), median=median, q25=q25,
                            q75=q75, maximum=max(values),
                            mean_coherent_power_transmission=float(np.mean(transmissions))))
    save_csv(args.out_dir / "trials.csv", rows)
    save_csv(args.out_dir / "summary.csv", summary)
    save_json(args.out_dir / "settings.json", dict(
        **vars(args), r_min=0., r_max=.5, measurements="exact", quadrature_order="xpxp",
        C_off="||M - blockdiag_2x2(M)||_F / ||M||_F",
        coherent_power_transmission="||M||_F^2 / (2N)",
        uniform_loss="applied once at the output",
        per_gate_loss="applied to both modes of each two-mode gate; the single-mode layer is lossless",
        padded="bypass modes receive the same loss as the active pair after each two-mode gate",
        cases=[dict(model=m, power_loss=l, final_layer=f, padded=p) for m, l, f, p in cases]))
    np.savez_compressed(args.out_dir / "matrices.npz",
                        targets=np.stack([t.numpy() for t in targets]),
                        final_matrices=np.asarray(matrices))
    lines = ["| Model | Power loss | Final layer | Dummy padding | Mean C_off (%) | Median [Q25, Q75] (%) | Max (%) | Mean coherent transmission (%) |",
             "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for s in summary:
        lines.append(f"| {s['model']} | {100*s['power_loss']:g}% | {s['final_layer']} | {s['padded']} | "
                     f"{100*s['mean']:.6g} | {100*s['median']:.6g} "
                     f"[{100*s['q25']:.6g}, {100*s['q75']:.6g}] | {100*s['maximum']:.6g} | "
                     f"{100*s['mean_coherent_power_transmission']:.6g} |")
    table = "\n".join(lines) + "\n"
    (args.out_dir / "summary.md").write_text(table)
    print(table)
    print(f"Saved to {args.out_dir}")


if __name__ == "__main__":
    main()
