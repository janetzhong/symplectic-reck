"""Test how small block errors affect the final nulling residual.

Sweep mode count N, distance alpha from a zero-pivot target, and squeezing
in dB. Each measured block gets independent noise of norm eps, with the same
noise directions for every eps. The alpha sweep uses N=3 and the dB sweep
uses N=4. Each trial also records the largest squeezing of any implemented
gate, 20 log10(sigma_max), in dB."""

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from ..gates import DTYPE, embed_single_mode, embed_two_mode, mode_block
from ..io_utils import ROOT, save_csv, save_json
from ..matrix_utils import normalized_residual
from ..nulling_analytical import analytical_nulling_gate, local_condition_number, projected_single_mode_inverse
from ..nulling_gateorder import nulling_schedule
from ..targets import isotropic_path_generator, isotropic_path_target, random_bm_target

EPSILONS = (1e-8, 1e-12)


def block_noise(gen, eps, per_block=False):
    """Draw noise with norm eps per block, or across both blocks if per_block=False."""
    noise = torch.randn(2, 2, 2, generator=gen, dtype=DTYPE)
    norm = torch.linalg.vector_norm(noise, dim=(1, 2), keepdim=True) if per_block else torch.linalg.vector_norm(noise)
    return noise * (eps / norm)


def gate_squeezing_db(gate):
    """Largest singular value of a gate in dB."""
    return 20 * math.log10(float(torch.linalg.svdvals(gate).max()))


def perturbation_trial(S, eps, seed, per_block=False):
    """Null S with noisy blocks and return the residual, conditioning and peak gate squeezing."""
    N = S.shape[0] // 2
    M = S.clone()
    gen = torch.Generator().manual_seed(seed)
    kappas, gates = [], []
    for step in nulling_schedule(N):
        anchor = mode_block(M, step.anchor, step.probe)
        target = mode_block(M, step.target, step.probe)
        kappas.append(local_condition_number(anchor, target))
        noise = block_noise(gen, eps, per_block=per_block)
        gate = analytical_nulling_gate(target + noise[0], anchor + noise[1])
        gates.append(gate)
        M = embed_two_mode(gate, step.target, step.anchor, N) @ M
    for mode in range(N):
        gate = projected_single_mode_inverse(mode_block(M, mode, mode), f"mode {mode}")
        gates.append(gate)
        M = embed_single_mode(gate, mode, N) @ M
    return dict(
        residual=normalized_residual(M), sqrt_sum_kappa=math.sqrt(sum(kappas)), kappa_max=max(kappas),
        max_gate_db=max(gate_squeezing_db(g) for g in gates), matrix=M,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=30, help="Trials per alpha/dB point.")
    parser.add_argument("--N-trials", type=int, default=60, help="Trials per mode-count point.")
    parser.add_argument("--seed", type=int, default=5040)
    parser.add_argument("--noise-seed", type=int, default=13000)
    parser.add_argument("--Ns", default="2,5,8,11,14,17,20,25,30")
    parser.add_argument("--alphas", default="1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2")
    parser.add_argument("--dbs", default="0,2,4,6,8,10")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/conditioning")
    args = parser.parse_args()
    sweeps = dict(
        N=[int(x) for x in args.Ns.split(",")],
        alpha=[float(x) for x in args.alphas.split(",")],
        db=[float(x) for x in args.dbs.split(",")],
    )
    if min(args.trials, args.N_trials) < 1 or min(sweeps["N"]) < 2:
        parser.error("trials must be positive and N must be >=2")
    if any(not math.isfinite(a) or a <= 0 for a in sweeps["alpha"]):
        parser.error("alphas must be finite and positive; alpha=0 is the singular endpoint")
    if any(not math.isfinite(db) or db < 0 for db in sweeps["db"]):
        parser.error("dbs must be finite and nonnegative")
    generators = [isotropic_path_generator(args.seed + t) for t in range(args.trials)]
    rows, summary, matrices = [], [], {}
    for sweep, values in sweeps.items():
        for value in values:
            N = int(value) if sweep == "N" else 3 if sweep == "alpha" else 4
            r_max = value * math.log(10) / 20 if sweep == "db" else 0.5
            trial_count = args.N_trials if sweep == "N" else args.trials
            group = []
            for trial in range(trial_count):
                if sweep == "alpha":
                    S = isotropic_path_target(value, generators[trial])
                else:
                    S, _ = random_bm_target(N, seed=args.seed + trial, r_min=0.0, r_max=r_max)
                e8 = perturbation_trial(S, EPSILONS[0], args.noise_seed + trial, per_block=True)
                e12 = perturbation_trial(S, EPSILONS[1], args.noise_seed + trial, per_block=True)
                group.append(dict(sweep=sweep, value=value, trial=trial, residual_e8=e8["residual"],
                                  residual_e12=e12["residual"], sqrt_sum_kappa=e8["sqrt_sum_kappa"],
                                  kappa_max=e8["kappa_max"], max_gate_db=e8["max_gate_db"]))
                prefix = f"{sweep}_{value}_trial{trial}"
                matrices[prefix + "__target"] = S.numpy()
                matrices[prefix + "__final_e8"] = e8["matrix"].numpy()
                matrices[prefix + "__final_e12"] = e12["matrix"].numpy()
            rows.extend(group)
            summary.append(dict(sweep=sweep, value=value, trials=trial_count, **{
                f"median_{key}": float(np.median([r[key] for r in group]))
                for key in ("residual_e8", "residual_e12", "kappa_max", "max_gate_db")
            }))
            print(f"conditioning {sweep}={value}: {trial_count} trials", flush=True)
    save_json(args.out_dir / "settings.json", dict(
        **vars(args), quadrature_order="xpxp", epsilons=EPSILONS, alpha_N=3, db_N=4,
        r_max="db*ln(10)/20 in the dB sweep, 0.5 otherwise",
        alpha_path="S = exp(alpha*L) @ S_iso with L the Hamiltonian part of log(S_random @ inv(S_iso))",
        noise="independent 2x2 block noise of norm eps on the target and anchor blocks of each step",
        max_gate_db="20*log10(sigma_max) over all two-mode nullers and single-mode corrections, eps=1e-8",
    ))
    save_csv(args.out_dir / "trials.csv", rows)
    save_csv(args.out_dir / "summary.csv", summary)
    np.savez_compressed(args.out_dir / "matrices.npz", **matrices)


if __name__ == "__main__":
    main()
