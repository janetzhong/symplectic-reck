"""Compare the squeezing needed with and without extra passive gates.

Before each nulling step, passive gates move the x_probe response towards
the anchor mode. This makes each pivot one in exact arithmetic. Use exact
measurements and minimum-squeezing nullers. Report peak two-mode squeezing
as 10 * acosh(kappa_max) / ln(10) dB."""

import argparse
import math
from pathlib import Path

import numpy as np
import torch

from ..gates import DTYPE, embed_two_mode, mode_block, passive_from_unitary
from ..io_utils import ROOT, save_csv, save_json
from ..matrix_utils import normalized_residual
from ..nulling_analytical import analytical_nulling_gate, local_condition_number
from ..nulling_gateorder import nulling_schedule
from ..run_nulling_sequence import finish_single_modes
from ..targets import (isotropic_failure_target, isotropic_path_generator,
                       isotropic_path_target, random_bm_target)


def extra_passive_gate(M, first, second, probe, N):
    """Zero the first mode’s x_probe response with a passive gate on the pair."""
    z1 = complex(M[2 * first, 2 * probe], M[2 * first + 1, 2 * probe])
    z2 = complex(M[2 * second, 2 * probe], M[2 * second + 1, 2 * probe])
    n = math.hypot(abs(z1), abs(z2))
    # Treat roundoff-sized responses as zero.
    if n < 1e-15:
        return torch.eye(2 * N, dtype=DTYPE)
    U = torch.tensor([[z2 / n, -z1 / n],
                      [z1.conjugate() / n, z2.conjugate() / n]], dtype=torch.complex128)
    return embed_two_mode(passive_from_unitary(U), first, second, N)


@torch.no_grad()
def run_sequence(S, extra_gates):
    """Run closed-form nulling and return (kappas, pivots, normalized residual)."""
    N = S.shape[0] // 2
    M = S.clone()
    kappas, pivots = [], []
    for step in nulling_schedule(N):
        if extra_gates:
            for a in range(step.probe, step.anchor):
                M = extra_passive_gate(M, a, a + 1, step.probe, N) @ M
        P = mode_block(M, step.anchor, step.probe)
        E = mode_block(M, step.target, step.probe)
        kappas.append(local_condition_number(P, E))
        pivots.append(float(torch.linalg.det(P) + torch.linalg.det(E)))
        gate = analytical_nulling_gate(E, P)
        M = embed_two_mode(gate, step.target, step.anchor, N) @ M
    M = finish_single_modes(M, N)
    return kappas, pivots, normalized_residual(M)


def squeezing_db(kappa):
    return 10 / math.log(10) * math.acosh(max(kappa, 1.0))


def trial(S, extra_gates):
    try:
        kappas, pivots, residual = run_sequence(S, extra_gates)
        k = max(kappas)
        return dict(kappa_max=k, squeezing_db=squeezing_db(k),
                    min_abs_pivot=min(abs(p) for p in pivots),
                    max_unit_pivot_error=max(abs(p - 1) for p in pivots),
                    residual=residual, failure=None)
    except ValueError as exc:
        return dict(kappa_max=math.inf, squeezing_db=math.inf, min_abs_pivot=0.0,
                    max_unit_pivot_error=math.inf, residual=math.inf, failure=str(exc))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=5040)
    parser.add_argument("--Ns", default="2,5,8,11,14,17,20,25,30")
    parser.add_argument("--alphas", default="1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/universal_architecture")
    args = parser.parse_args()
    Ns = [int(v) for v in args.Ns.split(",")]
    alphas = [float(v) for v in args.alphas.split(",")]
    if args.trials < 1 or args.seed < 0 or any(N < 2 for N in Ns):
        parser.error("trials >= 1, seed >= 0 and all N >= 2 are required")
    if any(not math.isfinite(a) or a <= 0 for a in alphas):
        parser.error("alphas must be finite and positive; S_iso is tested separately")
    torch.set_num_threads(1)
    rows = []
    for N in Ns:
        for t in range(args.trials):
            S, _ = random_bm_target(N, seed=args.seed + t, r_min=0.0, r_max=0.5)
            for extra in (False, True):
                rows.append(dict(sweep="N", value=N, trial=t, seed=args.seed + t, extra_gates=extra, **trial(S, extra)))
        print(f"N={N} done", flush=True)
    generators = [isotropic_path_generator(args.seed + t) for t in range(args.trials)]
    for alpha in alphas:
        for t, L in enumerate(generators):
            S = isotropic_path_target(alpha, L)
            for extra in (False, True):
                rows.append(dict(sweep="alpha", value=alpha, trial=t, seed=args.seed + t, extra_gates=extra, **trial(S, extra)))
        print(f"alpha={alpha:g} done", flush=True)
    for extra in (False, True):
        rows.append(dict(sweep="S_iso", value=0.0, trial=0, seed=None, extra_gates=extra,
                         **trial(isotropic_failure_target(), extra)))

    summary = []
    keys = sorted({(r["sweep"], r["value"], r["extra_gates"]) for r in rows},
                  key=lambda k: (k[0], k[1], k[2]))
    for sweep, value, extra in keys:
        sel = [r for r in rows if (r["sweep"], r["value"], r["extra_gates"]) == (sweep, value, extra)]
        # Count failed trials separately and exclude them from quartiles.
        valid = [r for r in sel if r["failure"] is None]
        k = np.array([r["kappa_max"] for r in valid])
        db = np.array([r["squeezing_db"] for r in valid])
        res = np.array([r["residual"] for r in valid])
        summary.append(dict(sweep=sweep, value=value, extra_gates=extra, trials=len(sel),
                            failures=int(sum(r["failure"] is not None for r in sel)),
                            median_kappa_max=float(np.median(k)) if len(valid) else math.nan,
                            median_squeezing_db=float(np.median(db)) if len(valid) else math.nan,
                            q25_squeezing_db=float(np.quantile(db, .25)) if len(valid) else math.nan,
                            q75_squeezing_db=float(np.quantile(db, .75)) if len(valid) else math.nan,
                            median_residual=float(np.median(res)) if len(valid) else math.nan))
    save_csv(args.out_dir / "trials.csv", rows)
    save_csv(args.out_dir / "summary.csv", summary)
    save_json(args.out_dir / "settings.json", dict(
        **vars(args), quadrature_order="xpxp", measurements="exact", r_min=0.0, r_max=0.5, alpha_N=3,
        squeezing_db="10*acosh(kappa_max)/ln(10) over the two-mode nulling steps",
        residual="||M-I||_F / sqrt(2N)",
        statistics="median and quartiles over successful trials; failures counted separately",
        extra_passive_gate_count="N*(N-1)*(N-2)/6"))
    lines = []
    lines.append("| sweep | value | extra gates | median kappa_max | median squeezing (dB) [Q25, Q75] | failures |")
    lines.append("| --- | --- | --- | ---: | ---: | ---: |")
    for s in summary:
        if s["failures"] == s["trials"]:
            lines.append(f"| {s['sweep']} | {s['value']:g} | {s['extra_gates']} | -- | failed | {s['failures']} |")
            continue
        lines.append(f"| {s['sweep']} | {s['value']:g} | {s['extra_gates']} | {s['median_kappa_max']:.4g} | "
              f"{s['median_squeezing_db']:.2f} [{s['q25_squeezing_db']:.2f}, {s['q75_squeezing_db']:.2f}] | "
              f"{s['failures']} |")

    table = "\n".join(lines) + "\n"
    (args.out_dir / "summary.md").write_text(table)
    print(table)
    print(f"Saved to {args.out_dir}")


if __name__ == "__main__":
    main()
