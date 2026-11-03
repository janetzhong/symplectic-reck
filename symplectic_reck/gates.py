"""Symplectic gates in (x1, p1, x2, p2, ...) order.

Mode indices start at zero. Gate parameters support PyTorch gradients."""

import torch

DTYPE = torch.float64


def phase_shift(phi):
    """2x2 phase-space rotation by phi radians."""
    phi = torch.as_tensor(phi, dtype=DTYPE)
    c, s = torch.cos(phi), torch.sin(phi)
    return torch.stack((torch.stack((c, -s)), torch.stack((s, c))))


def squeeze(r):
    """diag(exp(r), exp(-r)) for one mode."""
    r = torch.as_tensor(r, dtype=DTYPE)
    return torch.diag(torch.stack((torch.exp(r), torch.exp(-r))))


def single_mode_gate(params):
    """R(phi1) S(r) R(phi2) from three real parameters (phi1, r, phi2)."""
    phi1, r, phi2 = params
    return phase_shift(phi1) @ squeeze(r) @ phase_shift(phi2)


def beam_splitter(theta, phi):
    """Mix two modes by theta, then rotate the second mode by phi."""
    theta = torch.as_tensor(theta, dtype=DTYPE)
    c, s = torch.cos(theta), torch.sin(theta)
    eye = torch.eye(2, dtype=DTYPE)
    rot = phase_shift(phi)
    return torch.cat(
        (torch.cat((c * eye, -s * eye), 1), torch.cat((s * rot, c * rot), 1)), 0
    )


def passive_two_mode_gate(params):
    """Passive gate from (phi1, phi2, theta, output_phase)."""
    phi1, phi2, theta, output_phase = params
    return beam_splitter(theta, output_phase) @ torch.block_diag(
        phase_shift(phi1), phase_shift(phi2)
    )


def two_mode_gate(params):
    """Build O_left S(r1, r2) O_right from ten parameters.

    The first four set O_left, the next two set the squeezing, and the last
    four set O_right. Each passive part uses passive_two_mode_gate."""
    return (
        passive_two_mode_gate(params[:4])
        @ torch.block_diag(squeeze(params[4]), squeeze(params[5]))
        @ passive_two_mode_gate(params[6:])
    )


def passive_from_unitary(unitary):
    """Convert an NxN unitary into its real 2Nx2N quadrature matrix."""
    N = unitary.shape[0]
    result = torch.empty(2 * N, 2 * N, dtype=unitary.real.dtype)
    result[0::2, 0::2] = unitary.real
    result[0::2, 1::2] = -unitary.imag
    result[1::2, 0::2] = unitary.imag
    result[1::2, 1::2] = unitary.real
    return result


def embed_single_mode(gate, mode, N):
    """Place a 2x2 gate on one mode of an N-mode identity matrix."""
    result = torch.eye(2 * N, dtype=gate.dtype)
    result[2 * mode : 2 * mode + 2, 2 * mode : 2 * mode + 2] = gate
    return result


def embed_two_mode(gate, first, second, N):
    """Place a 4x4 gate on the ordered pair (first, second)."""
    result = torch.eye(2 * N, dtype=gate.dtype)
    idx = torch.tensor([2 * first, 2 * first + 1, 2 * second, 2 * second + 1])
    result[idx[:, None], idx[None, :]] = gate
    return result


def mode_block(matrix, output_mode, input_mode):
    """2x2 response from an input mode to an output mode."""
    return matrix[
        2 * output_mode : 2 * output_mode + 2, 2 * input_mode : 2 * input_mode + 2
    ]


def apply_two_mode_gate(matrix, N, target, anchor, params):
    """Apply a two-mode gate to the output modes (target, anchor)."""
    return embed_two_mode(two_mode_gate(params), target, anchor, N) @ matrix


def apply_single_mode_gate(matrix, N, mode, params):
    """Apply a single-mode gate to one output mode."""
    return embed_single_mode(single_mode_gate(params), mode, N) @ matrix
