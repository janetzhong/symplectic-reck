"""Compare closed-form, finite-difference, lock-in dithering and genetic training
with finite shots.

Each trial uses the same target for all methods, plus an exact closed-form
reference. Every feedback method gets 2000 loss evaluations per gate: FD and
dithering take 20 per two-mode update, GA measures a population of 20. Lock-in
dithering uses 20-sample windows on two-mode gates and 8-sample windows on
single-mode gates, with amplitude 0.05. The closed-form method gets the same
shot budget through q_eff = q * evals_per_gate * (N + 1) / (2N)."""

import argparse
import csv
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from ..io_utils import ROOT, save_csv, save_json
from ..matrix_utils import identity_residual
from ..measurements import stable_seed
from ..nulling_feedback import dither_tones, optimize_dither, optimize_fd, optimize_ga
from ..run_nulling_sequence import run_analytical_sequence, run_feedback_sequence
from ..targets import random_bm_target

METHODS = ["analytical", "measured_closed_form", "finite_diff", "dither_lockin", "genetic_algorithm"]


def shots_label(shots):
    return "inf" if shots is None else str(shots)


def experiment_settings(args, N):
    steps = args.steps_per_gate
    _, dither_two_T = dither_tones(10)
    _, dither_one_T = dither_tones(3)
    return dict(
        N=N, trials=args.trials, seed_start=args.seed_start, noise_seed_start=args.noise_seed_start,
        r_min=0.0, r_max=0.5, steps_per_gate=steps, population=20, fd_evals_per_update=20,
        singlemode_fd_steps=math.ceil(steps * 20 / 6), singlemode_ga_steps=steps,
        singlemode_fd_evals_per_update=6, budget_evals_per_gate=steps * 20,
        mutation_sigma=0.04, mutation_decay=0.995, fd_eps=0.08, fd_lr=0.02,
        ga_elite_fraction=0.3, ga_parent_fraction=0.5, grad_clip=10.0, param_bound=5.0,
        dither_amp=args.dither_amp, dither_twomode_window=dither_two_T,
        dither_twomode_steps=math.ceil(steps * 20 / dither_two_T), dither_singlemode_window=dither_one_T,
        dither_singlemode_steps=math.ceil(steps * 20 / dither_one_T), max_workers=args.max_workers,
    )


def effective_shots(shots, settings):
    """Choose shots per setting to match the feedback measurement budget."""
    if shots is None:
        return None
    N = settings["N"]
    return max(1, round(shots * settings["budget_evals_per_gate"] * (N + 1) / (2 * N)))


def make_optimizer(method, shots, seed_base, settings):
    """Set the budget and seeds for optimize(problem, M, label)."""
    N = settings["N"]
    label_q = shots_label(shots)

    def optimize(problem, M, label):
        single = label[0] == "single_mode"
        if method == "dither_lockin":
            seed = stable_seed(seed_base, method, label_q, N, *label)
            steps = settings["dither_singlemode_steps" if single else "dither_twomode_steps"]
            return optimize_dither(problem, M, shots, seed, steps, settings["dither_amp"], settings["fd_lr"],
                                   settings["grad_clip"], settings["param_bound"])
        if single:
            seed = stable_seed(seed_base, "single_mode", method, label_q, N, label[1])
        else:
            seed = stable_seed(seed_base, method, label_q, N, *label)
        if method == "finite_diff":
            steps = settings["singlemode_fd_steps"] if single else settings["steps_per_gate"]
            return optimize_fd(problem, M, shots, seed, steps, settings["fd_eps"], settings["fd_lr"],
                               settings["grad_clip"], settings["param_bound"])
        steps = settings["singlemode_ga_steps"] if single else settings["steps_per_gate"]
        return optimize_ga(problem, M, shots, stable_seed(seed, "direction"), stable_seed(seed, "measurement"),
                           steps, settings["population"], settings["mutation_sigma"], settings["mutation_decay"],
                           settings["ga_elite_fraction"], settings["ga_parent_fraction"], settings["param_bound"])

    return optimize


def run_method(S, method, shots, trial, settings):
    """Return (final matrix, step losses, two-mode histories, one-mode histories)."""
    N = settings["N"]
    seed_base = stable_seed(settings["noise_seed_start"], "trial", trial, method, shots_label(shots))
    if method == "analytical":
        M, step_losses = run_analytical_sequence(S, N)
        return M, step_losses, [], []
    if method == "measured_closed_form":
        M, step_losses = run_analytical_sequence(S, N, effective_shots(shots, settings), seed_base)
        return M, step_losses, [], []
    return run_feedback_sequence(S, N, make_optimizer(method, shots, seed_base, settings))


def run_trial(trial, settings, specs):
    torch.set_num_threads(1)  # Use one thread per trial.
    N = settings["N"]
    target_seed = settings["seed_start"] + trial
    S, _ = random_bm_target(N, target_seed, settings["r_min"], settings["r_max"])
    rows = []
    for method, shots in specs:
        row = dict(trial=trial, target_seed=target_seed, method=method, shots=shots, failure=None,
                   target_matrix=S.numpy())
        try:
            M, step_losses, histories, single_histories = run_method(S, method, shots, trial, settings)
            row.update(frobenius=identity_residual(M), final_matrix=M.numpy(), step_losses=step_losses,
                       loss_history=histories, singlemode_loss_history=single_histories)
        except Exception as exc:
            nan = np.full((2 * N, 2 * N), np.nan)
            row.update(failure=str(exc), frobenius=float("inf"), final_matrix=nan, step_losses=[],
                       loss_history=[], singlemode_loss_history=[])
        print(f"N={N} trial {trial:02d} {method:>20s} q={shots_label(shots):>6s} fro={row['frobenius']:.3e}"
              + (f"  FAILED: {row['failure']}" if row["failure"] else ""), flush=True)
        rows.append(row)
    return rows


def statistics(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return dict(
        valid=len(finite), failed=len(values) - len(finite),
        median=float(np.median(finite)) if len(finite) else None,
        q25=float(np.quantile(finite, 0.25)) if len(finite) else None,
        q75=float(np.quantile(finite, 0.75)) if len(finite) else None,
    )


def measurement_budgets(settings, specs):
    """Count total shots as shots per setting x settings per loss x loss evaluations."""
    N = settings["N"]
    gates = N * (N - 1) // 2
    rows = []
    for method, q in specs:
        effective = effective_shots(q, settings) if method == "measured_closed_form" else q
        if method == "analytical" or q is None:
            two = one = 0
            samples = None
        elif method == "measured_closed_form":
            two, one = 16, 8
            samples = effective * (two * gates + one * N)
        else:
            two = 8 * settings["budget_evals_per_gate"]
            single_evals = {"finite_diff": 6 * settings["singlemode_fd_steps"],
                            "dither_lockin": settings["dither_singlemode_window"] * settings["dither_singlemode_steps"],
                            "genetic_algorithm": 20 * settings["singlemode_ga_steps"]}[method]
            one = 8 * single_evals
            samples = q * (two * gates + one * N)
        rows.append(dict(N=N, method=method, q=q, effective_q=effective, twomode_gates=gates, singlemode_gates=N,
                         settings_per_twomode_gate=two, settings_per_singlemode_gate=one,
                         total_raw_homodyne_samples=samples))
    return rows


def save_run(out, settings, specs, records):
    N = settings["N"]
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "settings.json", {**settings, "method_specs": specs, "quadrature_order": "xpxp"})
    save_csv(out / "trials.csv", [{k: r[k] for k in ("trial", "target_seed", "method", "shots", "frobenius", "failure")}
                                  for r in records])
    summaries, local_losses, matrices, median_matrices, median_trials, histories = [], [], {}, {}, [], {}
    for r in records:
        prefix = f"{r['method']}__{shots_label(r['shots'])}__trial{r['trial']}"
        for kind in ("loss_history", "singlemode_loss_history"):
            for i, values in enumerate(r[kind]):
                histories[f"{prefix}__{kind}__gate{i}"] = np.asarray(values)
    for method, q in specs:
        selected = [r for r in records if r["method"] == method and r["shots"] == q]
        prefix = f"{method}__{shots_label(q)}"
        summaries.append(dict(N=N, method=method, shots=q,
                              **statistics([r["frobenius"] / math.sqrt(2 * N) for r in selected])))
        local_losses.append(dict(N=N, method=method, shots=q,
                                 **statistics([x for r in selected for x in r["step_losses"]])))
        matrices[prefix + "__target"] = np.stack([r["target_matrix"] for r in selected])
        matrices[prefix + "__final"] = np.stack([r["final_matrix"] for r in selected])
        valid = [r for r in selected if np.isfinite(r["frobenius"])]
        if valid:
            median = np.median([r["frobenius"] for r in valid])
            rep = min(valid, key=lambda r: abs(r["frobenius"] - median))
            median_trials.append(dict(method=method, shots=q, trial=rep["trial"]))
            median_matrices[prefix + "__target"] = rep["target_matrix"]
            median_matrices[prefix + "__final"] = rep["final_matrix"]
            median_matrices[prefix + "__error"] = rep["final_matrix"] - np.eye(2 * N)
    save_csv(out / "summary.csv", summaries)
    save_csv(out / "local_nulling_losses.csv", local_losses)
    np.savez_compressed(out / "matrices.npz", **matrices)
    np.savez_compressed(out / "median_matrices.npz", **median_matrices)
    np.savez_compressed(out / "loss_history.npz", **histories)
    save_json(out / "median_trials.json", median_trials)
    save_csv(out / "finite_shots_measurement_budgets.csv", measurement_budgets(settings, specs))


def merge_tables(out_root, selected_q):
    """Combine results from all N folders into the paper tables."""
    mesh, budgets = [], []
    for n_dir in sorted(out_root.glob("N*")):
        with (n_dir / "summary.csv").open() as f:
            mesh.extend(r for r in csv.DictReader(f) if r["method"] == "analytical" or r["shots"] == str(selected_q))
        with (n_dir / "finite_shots_measurement_budgets.csv").open() as f:
            budgets.extend(csv.DictReader(f))
    save_csv(out_root / "finite_shots_full_mesh_residuals.csv", mesh)
    save_csv(out_root / "finite_shots_measurement_budgets.csv", budgets)
    n3 = out_root / "N3/local_nulling_losses.csv"
    if n3.exists():
        (out_root / "finite_shots_local_nulling_losses.csv").write_text(n3.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--Ns", default="3,5,7,9,11,13,15,20,25,30")
    parser.add_argument("--q", type=int, default=1000, help="shots per setting for N>3")
    parser.add_argument("--table1-shots", default="100,1000,10000,100000,inf", help="shot sweep at N=3")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--steps-per-gate", type=int, default=100)
    parser.add_argument("--dither-amp", type=float, default=0.05)
    parser.add_argument("--seed-start", type=int, default=20)
    parser.add_argument("--noise-seed-start", type=int, default=1000)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--out-root", type=Path, default=ROOT / "results/finite_shots")
    args = parser.parse_args()
    # Run the largest trials first. Each trial has its own seed.
    runs = []
    with ProcessPoolExecutor(max_workers=args.max_workers) as pool:
        for N in sorted(map(int, args.Ns.split(",")), reverse=True):
            settings = experiment_settings(args, N)
            settings["selected_q"] = args.q
            qs = [None if s == "inf" else int(s) for s in (args.table1_shots if N == 3 else str(args.q)).split(",")]
            specs = [(m, q) for m in args.methods.split(",") for q in ([None] if m == "analytical" else qs)]
            print(f"N={N}: {args.trials} trials, q={qs}, {settings['budget_evals_per_gate']} evaluations/gate", flush=True)
            futures = [pool.submit(run_trial, t, settings, specs) for t in range(args.trials)]
            runs.append((N, settings, specs, futures))
        for N, settings, specs, futures in runs:
            save_run(args.out_root / f"N{N}", settings, specs, [r for f in futures for r in f.result()])
            print(f"N={N}: saved", flush=True)
    merge_tables(args.out_root, args.q)


if __name__ == "__main__":
    main()
