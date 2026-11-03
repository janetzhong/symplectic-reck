"""Compare in-situ and tomography-based nulling with fixed gate errors.

Tomography plans gates from one measured matrix, projected onto Sp(2N).
In-situ assumes sequential measurement access before the remaining unprogrammed
gates. Their identity-setting errors are excluded from each measurement. Both use
the same fixed errors. Noiseless runs show the effect of gate errors alone."""

import argparse
from pathlib import Path

import numpy as np

from ..gate_errors import draw_gate_errors
from ..io_utils import ROOT, save_csv, save_json
from ..matrix_utils import normalized_residual
from ..nulling_gateorder import nulling_schedule
from ..run_nulling_sequence import run_calibration_sequence
from ..targets import random_bm_target

METHODS = (
    ("tomo_10k", "tomo", True),
    ("insitu_equalq", "insitu", True),
    ("tomo_exact", "tomo", False),
    ("insitu_exact", "insitu", False),
)
DEFAULT_EPS = ",".join(f"{eps:.12g}" for eps in np.logspace(-3, -1, 13))


def run_experiment(N, trials, shots, eps_values, target_seed, error_seed, shot_seed):
    rows, matrices = [], {}
    for trial in range(trials):
        S, _ = random_bm_target(N, seed=target_seed + trial, r_min=0.0, r_max=0.5)
        matrices[f"trial{trial}__target"] = S.numpy()
        for eps in eps_values:
            offsets = draw_gate_errors(eps, error_seed + trial, len(nulling_schedule(N)), N)
            row = dict(trial=trial, eps=eps)
            for key, arm, measured in METHODS:
                try:
                    final = run_calibration_sequence(
                        arm, S, shots if measured else None, shot_seed + trial, offsets
                    )
                    row[key] = normalized_residual(final)
                    matrices[f"trial{trial}__eps{eps:g}__{key}"] = final.numpy()
                except ValueError as exc:
                    row[key] = None
                    row[key + "_error"] = str(exc)
            rows.append(row)
        print(f"N={N}: completed trial {trial + 1}/{trials}", flush=True)
    return rows, matrices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--Ns", default="4,10")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--shots", type=int, default=10000)
    parser.add_argument("--eps", default=DEFAULT_EPS,
                        help="comma-separated gate-error bounds (default: 13 log-spaced points from 1e-3 to 1e-1)")
    parser.add_argument("--target-seed", type=int, default=5040)
    parser.add_argument("--error-seed", type=int, default=10090)
    parser.add_argument("--shot-seed", type=int, default=13000)
    parser.add_argument("--out-root", type=Path, default=ROOT / "results/insitu_vs_tomog_with_gate_errors")
    args = parser.parse_args()
    eps_values = [float(x) for x in args.eps.split(",")]
    for N in map(int, args.Ns.split(",")):
        out = args.out_root / f"N{N}"
        save_json(out / "settings.json", dict(
            N=N, quadrature_order="xpxp", trials=args.trials, shots=args.shots, eps_values=eps_values,
            target_seed=args.target_seed, error_seed=args.error_seed, shot_seed=args.shot_seed,
            error_model="fixed independent uniform parameter error, left-multiplied",
            tomography="raw estimate projected onto Sp(2N) before planning",
            r_min=0.0, r_max=0.5, residual="Frobenius / sqrt(2N)",
        ))
        rows, matrices = run_experiment(
            N, args.trials, args.shots, eps_values, args.target_seed, args.error_seed, args.shot_seed
        )
        save_json(out / "trials.json", {"rows": rows})
        save_csv(out / "trials.csv", rows)
        np.savez_compressed(out / "matrices.npz", **matrices)
        summary = []
        for eps in eps_values:
            for key, _, _ in METHODS:
                values = [r[key] for r in rows if r["eps"] == eps and r[key] is not None]
                summary.append(dict(
                    eps=eps, method=key, valid_trials=len(values),
                    median=float(np.median(values)) if values else None,
                    q25=float(np.quantile(values, 0.25)) if values else None,
                    q75=float(np.quantile(values, 0.75)) if values else None,
                ))
        save_csv(out / "summary.csv", summary)


if __name__ == "__main__":
    main()
