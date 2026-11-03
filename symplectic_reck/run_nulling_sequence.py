"""Run nulling with closed-form gates, measured feedback, or autodiff."""

import torch

from .gate_errors import implement_gate
from .gates import DTYPE, embed_single_mode, embed_two_mode, mode_block, single_mode_gate, two_mode_gate
from .matrix_utils import project_symplectic
from .measurements import estimate_block, estimate_transfer_matrix, stable_seed
from .nulling_analytical import analytical_nulling_gate, projected_single_mode_inverse
from .nulling_feedback import single_mode_problem, two_mode_problem
from .nulling_gateorder import nulling_schedule


def finish_single_modes(M, N, shots=None, seed=0):
    """Correct each mode using the projected inverse of its measured diagonal block."""
    for mode in range(N):
        block = estimate_block(M, N, mode, mode, shots, stable_seed(seed, "single_mode_finish", mode))
        M = embed_single_mode(projected_single_mode_inverse(block, f"mode {mode}"), mode, N) @ M
    return M


def run_analytical_sequence(S, N, shots=None, seed=0):
    """Null each block with a closed-form gate. Use shots=None for exact blocks.

    Return the final matrix and each target block's loss just after nulling."""
    M = S.clone()
    step_losses = []
    for step in nulling_schedule(N):
        s = stable_seed(seed, "measured_closed_form", step.index, step.target, step.anchor, step.probe)
        target_block = estimate_block(M, N, step.target, step.probe, shots, stable_seed(s, "target_block"))
        anchor_block = estimate_block(M, N, step.anchor, step.probe, shots, stable_seed(s, "anchor_block"))
        gate = analytical_nulling_gate(target_block, anchor_block)
        M = embed_two_mode(gate, step.target, step.anchor, N) @ M
        step_losses.append(float(torch.sum(mode_block(M, step.target, step.probe).square())))
    M = finish_single_modes(M, N, shots, stable_seed(seed, "measured_closed_form_finish"))
    return M, step_losses


def run_feedback_sequence(S, N, optimize):
    """Train and apply one gate at a time using measured losses.

    Supply optimize(problem, M, label), which returns parameters and loss
    history. label identifies the gate for seeding. Return the final matrix,
    step losses, two-mode histories, and single-mode histories."""
    M = S.clone()
    step_losses, histories, single_histories = [], [], []
    for step in nulling_schedule(N):
        problem = two_mode_problem(N, step.target, step.anchor, step.probe)
        params, losses = optimize(problem, M, (step.target, step.anchor, step.probe))
        M = problem.apply(M, params)
        step_losses.append(losses[-1])
        histories.append(losses)
    for mode in range(N):
        problem = single_mode_problem(N, mode)
        params, losses = optimize(problem, M, ("single_mode", mode))
        M = problem.apply(M, params)
        single_histories.append(losses)
    return M, step_losses, histories, single_histories


def full_circuit_matrix(S, N, schedule, params, single_params):
    M = S
    for step, p in zip(schedule, params):
        M = embed_two_mode(two_mode_gate(p), step.target, step.anchor, N) @ M
    for mode, p in enumerate(single_params):
        M = embed_single_mode(single_mode_gate(p), mode, N) @ M
    return M


def run_autodiff_sequence(S, N, steps_per_gate=600, lr=0.02, singlemode_steps=2000, singlemode_lr=0.05, *, trace_callback=None, parallel_single_modes=False):
    """Train gates with Adam and exact gradients.

    At step k, train only gate k and minimize the total squared norm of blocks
    nulled so far. Finish by training the single-mode gates, together if
    parallel_single_modes=True. Return matrices, parameters, and two-mode losses.

    trace_callback(stage, iteration, matrix, loss, gradient_norm) receives each
    state before the update, with a detached matrix. Two-mode stages come first,
    then one stage per single mode (or one shared stage when trained together)."""
    schedule = nulling_schedule(N)
    params = [torch.zeros(10, dtype=DTYPE) for _ in schedule]
    single_params = [torch.zeros(3, dtype=DTYPE) for _ in range(N)]
    snapshots = [full_circuit_matrix(S, N, schedule, params, single_params).detach()]
    loss_history = []
    for k, _ in enumerate(schedule):
        params[k] = params[k].clone().requires_grad_(True)
        opt = torch.optim.Adam([params[k]], lr=lr)
        losses = []
        for iteration in range(steps_per_gate):
            opt.zero_grad()
            M = full_circuit_matrix(S, N, schedule, params, single_params)
            loss = sum(torch.sum(mode_block(M, s.target, s.probe).square()) for s in schedule[: k + 1])
            loss.backward()
            if trace_callback is not None:
                trace_callback(k, iteration, M.detach().clone(), float(loss.detach()), float(params[k].grad.norm()))
            opt.step()
            losses.append(float(loss))
        loss_history.append(losses)
        params[k] = params[k].detach()
        snapshots.append(full_circuit_matrix(S, N, schedule, params, single_params).detach())
    eye2 = torch.eye(2, dtype=DTYPE)
    mode_groups = [list(range(N))] if parallel_single_modes else [[mode] for mode in range(N)]
    for modes in mode_groups:
        for mode in modes:
            p = single_params[mode].clone()
            p[0] = 0.001  # Start with a small rotation.
            single_params[mode] = p.requires_grad_(True)
        active_params = [single_params[mode] for mode in modes]
        opt = torch.optim.Adam(active_params, lr=singlemode_lr)
        for iteration in range(singlemode_steps):
            opt.zero_grad()
            M = full_circuit_matrix(S, N, schedule, params, single_params)
            loss = sum(torch.sum((mode_block(M, mode, mode) - eye2).square()) for mode in modes)
            loss.backward()
            if trace_callback is not None:
                gradient_norm = torch.cat([p.grad for p in active_params]).norm()
                trace_callback(len(schedule) + modes[0], iteration, M.detach().clone(), float(loss.detach()), float(gradient_norm))
            opt.step()
        for mode in modes:
            single_params[mode] = single_params[mode].detach()
        snapshots.append(full_circuit_matrix(S, N, schedule, params, single_params).detach())
    return dict(
        schedule=schedule, snapshots=snapshots, final_matrix=snapshots[-1],
        params=params, single_params=single_params, loss_history=loss_history,
    )


def run_calibration_sequence(arm, S, shots, seed, offsets):
    """Apply gates with fixed errors, using tomography or in-situ measurements.

    "tomo" plans gates from a single estimate of S projected onto Sp(2N).
    "insitu" measures the current response before each gate.
    shots=None uses exact measurements. Return the physical output matrix."""
    N = S.shape[0] // 2
    two_offsets, one_offsets = offsets
    physical = S.clone()
    if arm == "tomo":
        model = project_symplectic(estimate_transfer_matrix(S, N, shots, seed))
    elif arm != "insitu":
        raise ValueError(arm)
    for step, offset in zip(nulling_schedule(N), two_offsets):
        if arm == "insitu":
            s = seed + 1000 * step.index
            target_block = estimate_block(physical, N, step.target, step.probe, shots, s)
            anchor_block = estimate_block(physical, N, step.anchor, step.probe, shots, s + 101)
        else:
            target_block = mode_block(model, step.target, step.probe)
            anchor_block = mode_block(model, step.anchor, step.probe)
        gate = analytical_nulling_gate(target_block, anchor_block)
        physical = embed_two_mode(implement_gate(gate, offset), step.target, step.anchor, N) @ physical
        if arm == "tomo":
            model = embed_two_mode(gate, step.target, step.anchor, N) @ model
    for mode, offset in enumerate(one_offsets):
        if arm == "insitu":
            block = estimate_block(physical, N, mode, mode, shots, seed + 100000 + 10 * mode)
        else:
            block = mode_block(model, mode, mode)
        gate = projected_single_mode_inverse(block, f"{arm} mode {mode}")
        physical = embed_single_mode(implement_gate(gate, offset), mode, N) @ physical
        if arm == "tomo":
            model = embed_single_mode(gate, mode, N) @ model
    return physical
