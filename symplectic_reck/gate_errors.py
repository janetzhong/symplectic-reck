"""Fixed calibration errors for each gate.

Each gate's error parameters are drawn once from Uniform[-eps, eps] and the
implemented gate is error_gate @ requested_gate."""

import torch

from .gates import DTYPE, single_mode_gate, two_mode_gate


def draw_gate_errors(eps, seed, n_twomode, n_singlemode):
    """Return (two-mode error gates, single-mode error gates) in gate order."""
    gen = torch.Generator().manual_seed(seed)
    two = [
        two_mode_gate((2.0 * torch.rand(10, generator=gen, dtype=DTYPE) - 1.0) * eps)
        for _ in range(n_twomode)
    ]
    one = [
        single_mode_gate((2.0 * torch.rand(3, generator=gen, dtype=DTYPE) - 1.0) * eps)
        for _ in range(n_singlemode)
    ]
    return two, one


def implement_gate(requested, offset=None):
    """Apply the fixed error after the requested gate."""
    return requested if offset is None else offset @ requested
