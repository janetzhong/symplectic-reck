"""Train gates from measured block losses using finite differences, lock-in
dithering or a genetic algorithm.

Two-mode gates drive a response block to zero. Single-mode gates drive a
diagonal block to identity. Each loss evaluation uses a fresh measurement."""

import math
from dataclasses import dataclass
from typing import Callable

import torch

from .gates import DTYPE, apply_single_mode_gate, apply_two_mode_gate, mode_block
from .measurements import estimate_block, stable_seed


@dataclass
class GateTrainingProblem:
    """The gate to train, the response block to change, and its target value."""

    apply: Callable  # apply(M, params) -> M with the gate applied
    out_mode: int
    in_mode: int
    goal: torch.Tensor
    width: int  # number of gate parameters
    fd_tag: int  # seed tags that keep the FD and GA noise streams separate
    ga_tag: int


def two_mode_problem(N, target, anchor, probe):
    return GateTrainingProblem(
        lambda M, p: apply_two_mode_gate(M, N, target, anchor, p),
        target, probe, torch.zeros(2, 2, dtype=DTYPE), 10, fd_tag=1, ga_tag=10,
    )


def single_mode_problem(N, mode):
    return GateTrainingProblem(
        lambda M, p: apply_single_mode_gate(M, N, mode, p),
        mode, mode, torch.eye(2, dtype=DTYPE), 3, fd_tag=6, ga_tag=11,
    )


def exact_loss(problem, M, params):
    """||block - goal||_F^2 from the exact matrix."""
    block = mode_block(problem.apply(M, params), problem.out_mode, problem.in_mode)
    return float(torch.sum((block - problem.goal).square()))


def measured_loss(problem, M, params, shots, seed):
    """Measure ||block - goal||_F^2. Return inf on failure or a non-finite result."""
    try:
        block = estimate_block(
            problem.apply(M, params), M.shape[0] // 2, problem.out_mode, problem.in_mode, shots, seed
        )
        loss = float(torch.sum((block - problem.goal).square()))
        return loss if math.isfinite(loss) else float("inf")
    except Exception:
        return float("inf")


def _eval_seed(base, *indices):
    return stable_seed("loss_eval", base, *indices)


def optimize_fd(problem, M, shots, seed, steps, eps=0.08, lr=0.02, grad_clip=10.0, param_bound=5.0):
    """Train with Adam using central finite differences.

    Each step uses two measurements per parameter. Return the parameters
    and the exact block loss after each update."""
    params = torch.zeros(problem.width, dtype=DTYPE)
    m = torch.zeros_like(params)
    v = torch.zeros_like(params)
    beta1, beta2 = 0.9, 0.99
    losses = []
    for step in range(steps):
        grad = torch.zeros_like(params)
        for q in range(problem.width):
            delta = torch.zeros_like(params)
            delta[q] = eps
            lp = measured_loss(problem, M, params + delta, shots, _eval_seed(seed, problem.fd_tag, step, q, 0, 0))
            lm = measured_loss(problem, M, params - delta, shots, _eval_seed(seed, problem.fd_tag, step, q, 0, 1))
            grad[q] = (lp - lm) / (2 * eps)
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        norm = float(torch.norm(grad))
        if norm > grad_clip:
            grad = grad * (grad_clip / norm)
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * grad.square()
        m_hat = m / (1.0 - beta1 ** (step + 1))
        v_hat = v / (1.0 - beta2 ** (step + 1))
        params = torch.clamp(params - lr * m_hat / (torch.sqrt(v_hat) + 1e-8), -param_bound, param_bound)
        losses.append(exact_loss(problem, M, params))
    return params, losses


def optimize_ga(
    problem, M, shots, seed, noise_seed, steps, population=20,
    mutation_sigma=0.04, mutation_decay=0.995, elite_fraction=0.3, parent_fraction=0.5, param_bound=5.0,
):
    """Train with a genetic search, measuring each candidate once per generation.

    seed controls the population and noise_seed controls measurement noise.
    Return the final winner and its exact loss from each generation."""
    gen = torch.Generator().manual_seed(seed)
    candidates = torch.randn(population, problem.width, dtype=DTYPE, generator=gen) * mutation_sigma
    candidates[0].zero_()
    candidates = torch.clamp(candidates, -param_bound, param_bound)
    elite_count = max(1, min(population, round(elite_fraction * population)))
    parent_count = max(2, min(population, round(parent_fraction * population)))
    losses = []
    for step in range(steps):
        fitness = torch.tensor(
            [measured_loss(problem, M, p, shots, _eval_seed(noise_seed, problem.ga_tag, step, i)) for i, p in enumerate(candidates)],
            dtype=DTYPE,
        )
        order = torch.argsort(torch.nan_to_num(fitness, nan=float("inf"), posinf=float("inf")))
        best = candidates[order[0]].clone()
        losses.append(exact_loss(problem, M, best))
        if step + 1 == steps:
            break
        ranked = candidates[order]
        next_pop = [ranked[i].clone() for i in range(elite_count)]
        parents = ranked[:parent_count]
        while len(next_pop) < population:
            i = int(torch.randint(parent_count, (), generator=gen))
            j = int(torch.randint(parent_count, (), generator=gen))
            alpha = torch.rand(problem.width, dtype=DTYPE, generator=gen)
            child = alpha * parents[i] + (1.0 - alpha) * parents[j]
            child = child + mutation_sigma * torch.randn(problem.width, dtype=DTYPE, generator=gen)
            next_pop.append(torch.clamp(child, -param_bound, param_bound))
        candidates = torch.stack(next_pop)
        mutation_sigma *= mutation_decay
    return best, losses


# Lock-in dithering: parameter q is modulated as theta_q + a * r_q(t), where r_q
# is a sine or cosine at an odd harmonic of the lock-in window. Odd tones keep
# second-order mixing products (f_i +/- f_j, 2 f_i) on even bins, so curvature
# does not leak into the gradient estimate. With T samples per window,
#
#     L(t) ~ L0 + a sum_q g_q r_q(t),   sum_t r_p(t) r_q(t) = (T/2) delta_pq,
#
# so g_q = 2 / (a T) * sum_t L(t) r_q(t). The DC level L0 drops out.
#
# A ten-parameter gate uses tones (1, 3, 5, 7, 9) x (sin, cos) over T = 20
# samples, the same 20 loss evaluations per update as central finite differences
# (2 x 10). Each sample is a fresh noisy measurement.


def dither_tones(width):
    """Return reference waveforms (width x T) and the window length T."""
    pairs = math.ceil(width / 2)
    max_freq = 2 * pairs - 1
    T = 2 * max_freq + 2  # Even, above Nyquist for all second-order products
    t = torch.arange(T, dtype=DTYPE)
    refs = []
    for q in range(width):
        f = 2 * (q // 2) + 1
        phase = 0.0 if q % 2 == 0 else math.pi / 2
        refs.append(torch.sin(2 * math.pi * f * t / T + phase))
    return torch.stack(refs), T


def lockin_gradient(problem, M, params, refs, amp, shots, seed):
    """Measure one dither window and demodulate it into a gradient estimate."""
    T = refs.shape[1]
    readings = torch.tensor(
        [measured_loss(problem, M, params + amp * refs[:, t], shots, stable_seed("lockin", seed, t))
         for t in range(T)],
        dtype=DTYPE,
    )
    if not torch.all(torch.isfinite(readings)):
        return torch.zeros_like(params)
    return 2.0 / (amp * T) * refs @ readings


def optimize_dither(problem, M, shots, seed, steps, amp=0.08, lr=0.02, grad_clip=10.0, param_bound=5.0):
    """Train with Adam using lock-in gradients, one dither window per update.

    Uses the same Adam settings as optimize_fd. Return the parameters and the
    exact block loss after each update."""
    refs, _ = dither_tones(problem.width)
    params = torch.zeros(problem.width, dtype=DTYPE)
    m = torch.zeros_like(params)
    v = torch.zeros_like(params)
    beta1, beta2 = 0.9, 0.99
    losses = []
    for step in range(steps):
        grad = lockin_gradient(problem, M, params, refs, amp, shots, stable_seed(seed, step))
        norm = float(torch.norm(grad))
        if norm > grad_clip:
            grad = grad * (grad_clip / norm)
        m = beta1 * m + (1.0 - beta1) * grad
        v = beta2 * v + (1.0 - beta2) * grad.square()
        m_hat = m / (1.0 - beta1 ** (step + 1))
        v_hat = v / (1.0 - beta2 ** (step + 1))
        params = torch.clamp(params - lr * m_hat / (torch.sqrt(v_hat) + 1e-8), -param_bound, param_bound)
        losses.append(exact_loss(problem, M, params))
    return params, losses
