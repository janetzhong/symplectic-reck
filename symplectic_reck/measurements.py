"""Estimate response matrices using coherent probes and homodyne measurements.

For displacement d, the mean output is M @ d. With q shots, the noise
variance of quadrature i is (M M^T)_ii / (2q), using vacuum covariance I/2."""

import hashlib
import math

import torch

from .gates import DTYPE


def stable_seed(*parts):
    """Make a repeatable seed from the trial, gate, and other run labels."""
    h = hashlib.blake2b(digest_size=8)
    for part in parts:
        h.update(str(part).encode("utf-8"))
        h.update(b"\x00")
    return int.from_bytes(h.digest(), "little") & (1 << 63) - 1


def homodyne_mean(M, displacement, out_index, shots=None, seed=0):
    """Measure one output quadrature. Use shots=None for its exact mean."""
    mean = M @ displacement
    if shots is None:
        return mean[out_index]
    gen = torch.Generator().manual_seed(seed)
    z = torch.randn(M.shape[1], dtype=DTYPE, generator=gen)
    # The first N draws are the x noises and the next N the p noises; interleave to (x1, p1, ...).
    z = z.reshape(2, -1).T.reshape(-1)
    return mean[out_index] + torch.dot(M[out_index], z) / math.sqrt(2 * shots)


def estimate_block(M, N, out_mode, in_mode, shots=None, seed=0):
    """Measure a 2x2 response block using +/-1 input displacements.

    Uses eight homodyne settings, with shots samples per setting.
    With shots=None, returns the exact block."""
    block = torch.zeros(2, 2, dtype=DTYPE)
    for col, input_index in enumerate((2 * in_mode, 2 * in_mode + 1)):
        for row, output_index in enumerate((2 * out_mode, 2 * out_mode + 1)):
            readings = []
            for sign in (1.0, -1.0):
                displacement = torch.zeros(2 * N, dtype=DTYPE)
                displacement[input_index] = sign
                readings.append(homodyne_mean(M, displacement, output_index, shots, seed))
                seed += 1
            block[row, col] = (readings[0] - readings[1]) / 2.0
    return block


def estimate_transfer_matrix(M, N, shots=None, seed=0):
    """Measure all entries of the 2Nx2N response matrix (tomography)."""
    estimated = torch.zeros(2 * N, 2 * N, dtype=DTYPE)
    # Probe and read out all x quadratures first, then all p. Entries are stored in xpxp order.
    indices = list(range(0, 2 * N, 2)) + list(range(1, 2 * N, 2))
    for col in indices:
        for row in indices:
            readings = []
            for sign in (1.0, -1.0):
                displacement = torch.zeros(2 * N, dtype=DTYPE)
                displacement[col] = sign
                readings.append(homodyne_mean(M, displacement, row, shots, seed))
                seed += 1
            estimated[row, col] = (readings[0] - readings[1]) / 2.0
    return estimated
