"""Symplectic form, matrix residuals, and projection onto the symplectic group."""

import math

import torch

from .gates import DTYPE


def omega(N):
    """Symplectic form blockdiag([[0, 1], [-1, 0]], ...) in interleaved order."""
    return torch.kron(
        torch.eye(N, dtype=DTYPE), torch.tensor([[0.0, 1.0], [-1.0, 0.0]], dtype=DTYPE)
    )


def identity_residual(M):
    """||M - I||_F."""
    return float(torch.linalg.norm(M - torch.eye(M.shape[0], dtype=M.dtype)))


def normalized_residual(M):
    """||M - I||_F / sqrt(2N)."""
    return identity_residual(M) / math.sqrt(M.shape[0])


def project_symplectic(M):
    """Project M onto Sp(2N) as M (Omega^T M^T Omega M)^(-1/2).

    Leaves symplectic matrices unchanged. To first order, it removes the
    noise component that violates the symplectic condition. Raises ValueError
    if the inverse square root has a significant imaginary part."""
    J = omega(M.shape[0] // 2)
    G = J.T @ M.T @ J @ M
    vals, vecs = torch.linalg.eig(G)
    inv_sqrt = vecs @ torch.diag(vals.rsqrt()) @ torch.linalg.inv(vecs)
    if float(inv_sqrt.imag.abs().max()) > 1e-8:
        raise ValueError("symplectic projection failed: estimate too far from Sp(2N)")
    return M @ inv_sqrt.real
