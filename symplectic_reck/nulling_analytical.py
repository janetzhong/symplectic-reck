"""Closed-form gates that null a response block or correct a single mode."""

import math

import torch

from .gates import DTYPE, passive_from_unitary


def analytical_nulling_gate(target_block, anchor_block):
    """Build the minimum-squeezing gate that zeros target_block.

    Both blocks refer to the same input probe. The gate acts in the order
    (x_target, p_target, x_anchor, p_anchor). Raises ValueError if a nonzero
    target cannot be nulled because the frame or symplectic pivot is singular."""
    if torch.norm(target_block).item() < 1e-14:
        return torch.eye(4, dtype=DTYPE)
    gate = _anchor_first_nuller(anchor_block, target_block)
    idx = [2, 3, 0, 1]
    return gate[idx, :][:, idx]


def local_condition_number(anchor_block, target_block):
    """Return kappa = frame area / |det(anchor) + det(target)|."""
    V = torch.cat((anchor_block, target_block), 0).to(DTYPE)
    delta = torch.linalg.det(anchor_block) + torch.linalg.det(target_block)
    if abs(float(delta)) < 1e-14:
        return math.inf
    area = torch.sqrt(torch.clamp(torch.linalg.det(V.T @ V), min=0.0))
    return float(torch.clamp(area / torch.abs(delta), min=1.0))


def _complexify(v):
    return torch.stack((v[0] + 1j * v[1], v[2] + 1j * v[3])).to(torch.complex128)


def _unitary_with_first_column(z):
    z = z / torch.linalg.norm(z)
    return torch.stack((z, torch.stack((-z[1].conj(), z[0].conj()))), dim=1)


def _polar_unitary(U):
    W, _, Vh = torch.linalg.svd(U)
    return W @ Vh


def _anchor_first_nuller(anchor_block, target_block):
    """Nuller acting on (x_anchor, p_anchor, x_target, p_target)."""
    P = anchor_block.to(DTYPE)
    E = target_block.to(DTYPE)
    V = torch.cat((P, E), 0)  # 4x2 response frame of the probe column
    gram = V.T @ V
    eigvals, eigvecs = torch.linalg.eigh(gram)
    if float(eigvals[0]) <= 1e-28:
        raise ValueError("minimal nuller failed: local response frame is rank deficient")
    inv_sqrt = eigvecs @ torch.diag(torch.rsqrt(torch.clamp(eigvals, min=1e-28))) @ eigvecs.T
    X = V @ inv_sqrt  # orthonormal frame
    delta = torch.linalg.det(P) + torch.linalg.det(E)
    area = torch.sqrt(torch.clamp(torch.linalg.det(gram), min=0.0))
    if abs(float(delta)) < 1e-14:
        raise ValueError("minimal nuller failed: symplectic pivot is singular")
    kappa = torch.clamp(area / torch.abs(delta), min=1.0)
    zeta1 = _complexify(X[:, 0])
    zeta2 = _complexify(X[:, 1])
    if float(delta) < 0.0:
        zeta1, zeta2 = zeta2, zeta1
    if float(kappa - 1.0) < 1e-14:  # At kappa=1, a passive gate is enough.
        U = _unitary_with_first_column(zeta1)
        return passive_from_unitary(U.conj().T)
    a = torch.arccosh(kappa)  # The required squeezing is a/2.
    denom = torch.sqrt(2.0 * torch.cosh(a)).to(torch.complex128)
    eta = torch.zeros(2, 2, dtype=torch.complex128)
    eta[:, 0] = torch.stack((torch.exp(a / 2), torch.exp(-a / 2))).to(torch.complex128) / denom
    eta[:, 1] = 1j * torch.stack((torch.exp(-a / 2), torch.exp(a / 2))).to(torch.complex128) / denom
    zeta = torch.stack((zeta1, zeta2), dim=1)
    U = _polar_unitary(zeta @ torch.linalg.inv(eta))
    B = torch.tensor([[1.0, -1.0], [1.0, 1.0]], dtype=torch.complex128) / math.sqrt(2.0)
    K = torch.diag(torch.stack((torch.exp(-a / 2), torch.exp(a / 2), torch.exp(a / 2), torch.exp(-a / 2))))
    return passive_from_unitary(B.T) @ K @ passive_from_unitary(U.conj().T)


def projected_single_mode_inverse(block, context="single mode"):
    """Project a measured 2x2 block onto Sp(2,R), then invert it.

    Clip small singular values, correct any reflection along the weakest
    singular direction, and normalize the determinant to one."""
    if not torch.isfinite(block).all():
        raise ValueError(f"{context}: block contains non-finite values")
    u, singular, vh = torch.linalg.svd(block.to(DTYPE))
    singular = torch.clamp(singular, min=1e-10)
    projected = u @ torch.diag(singular) @ vh
    if float(torch.linalg.det(projected)) < 0.0:
        u = u.clone()
        u[:, -1] *= -1.0
        projected = u @ torch.diag(singular) @ vh
    return torch.linalg.inv(projected / torch.sqrt(torch.linalg.det(projected)))
