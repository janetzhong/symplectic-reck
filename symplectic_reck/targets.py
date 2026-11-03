"""Target matrices for random Bloch–Messiah transformations and zero-pivot examples."""

import torch
from scipy.linalg import logm

from .gates import DTYPE, passive_from_unitary
from .matrix_utils import omega


def random_orthosymplectic(N, seed):
    """Passive transformation from the eigenvectors of a random Hermitian matrix."""
    gen = torch.Generator().manual_seed(seed)
    A = torch.randn(N, N, dtype=torch.complex128, generator=gen)
    _, U = torch.linalg.eigh(A + A.conj().T)
    phases = torch.exp(2j * torch.pi * torch.rand(N, dtype=DTYPE, generator=gen))
    return passive_from_unitary(U @ torch.diag(phases))


def random_bm_target(N, seed=0, r_min=0.0, r_max=0.5):
    """Return (S, r) with squeezing strengths r ~ Uniform[r_min, r_max]."""
    gen = torch.Generator().manual_seed(10000 * seed + 101)
    r = r_min + torch.rand(N, dtype=DTYPE, generator=gen) * (r_max - r_min)
    r, _ = torch.sort(r, descending=True)
    O1 = random_orthosymplectic(N, 10000 * seed + 201)
    O2 = random_orthosymplectic(N, 10000 * seed + 301)
    K = torch.diag(torch.stack((torch.exp(r), torch.exp(-r)), dim=1).flatten())
    return O1 @ K @ O2, r


def isotropic_failure_target():
    """Three-mode target with a zero pivot, stored in xpxp order."""
    return torch.tensor([
        [1., 0.,  0.,  0.,  0., -1.],
        [0., 1.,  0.,  1.,  0.,  0.],
        [1., 0., -1.,  0.,  0.,  0.],
        [0., 0.,  0., -1.,  0.,  0.],
        [0., 0.,  0.,  0.,  0.,  1.],
        [0., 1.,  0.,  1., -1.,  0.],
    ], dtype=DTYPE)


def isotropic_path_generator(seed=0):
    """Build L for the path exp(alpha L) S_iso.

    Project real(log(S_random @ inv(S_iso))) onto the Hamiltonian matrices.
    Taking the real part means alpha=1 need not recover S_random."""
    S_iso = isotropic_failure_target()
    S_random, _ = random_bm_target(3, seed=seed)
    L = torch.tensor(logm((S_random @ torch.linalg.inv(S_iso)).numpy()).real, dtype=DTYPE)
    J = omega(3)
    return 0.5 * (L + J @ L.T @ J)


def isotropic_path_target(alpha, generator):
    """Return exp(alpha L) S_iso. Use alpha=0 for the zero-pivot target."""
    return torch.linalg.matrix_exp(alpha * generator) @ isotropic_failure_target()
