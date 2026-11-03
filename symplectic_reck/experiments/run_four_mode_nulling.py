"""Train a four-mode mesh with autodiff and save the matrix after each gate."""

import argparse
from pathlib import Path

import numpy as np
import torch

from ..io_utils import ROOT, save_csv, save_json
from ..matrix_utils import normalized_residual
from ..run_nulling_sequence import run_autodiff_sequence
from ..targets import random_bm_target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/four_mode_nulling")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--r-min", type=float, default=0.3)
    parser.add_argument("--r-max", type=float, default=0.4)
    parser.add_argument("--steps-per-gate", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--singlemode-steps", type=int, default=500)
    parser.add_argument("--singlemode-lr", type=float, default=0.05)
    args = parser.parse_args()
    N = 4
    S, r = random_bm_target(N, seed=args.seed, r_min=args.r_min, r_max=args.r_max)
    result = run_autodiff_sequence(
        S, N, args.steps_per_gate, args.lr, args.singlemode_steps, args.singlemode_lr
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_dir / "matrices.npz",
        target=S.numpy(),
        snapshots=torch.stack(result["snapshots"]).numpy(),
        params=torch.stack(result["params"]).numpy(),
        single_params=torch.stack(result["single_params"]).numpy(),
    )
    np.savez_compressed(
        args.out_dir / "loss_history.npz",
        **{f"gate_{i}": np.asarray(h) for i, h in enumerate(result["loss_history"])},
    )
    residual = normalized_residual(result["final_matrix"])
    save_json(
        args.out_dir / "settings.json",
        dict(
            **vars(args), N=N, quadrature_order="xpxp", r_vals=r,
            gates=[(s.target, s.anchor) for s in result["schedule"]],
            nulled_blocks=[(s.target, s.probe) for s in result["schedule"]],
            residual="frobenius / sqrt(2N)", final_residual=residual,
        ),
    )
    save_csv(args.out_dir / "summary.csv", [dict(seed=args.seed, final_residual=residual)])
    print(f"saved to {args.out_dir} (final_residual={residual:.3e})")


if __name__ == "__main__":
    main()
