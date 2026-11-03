"""Check that extra passive gates remove zero pivots."""

import unittest

import torch

from symplectic_reck.experiments.run_universal_architecture import (
    extra_passive_gate, run_sequence, squeezing_db,
)
from symplectic_reck.matrix_utils import omega
from symplectic_reck.targets import isotropic_failure_target, random_bm_target


class UniversalArchitectureTests(unittest.TestCase):
    def test_zero_pivot_target_fails_without_and_inverts_with_extra_gates(self):
        target = isotropic_failure_target()
        with self.assertRaises(ValueError):
            run_sequence(target, extra_gates=False)
        kappas, pivots, residual = run_sequence(target, extra_gates=True)
        self.assertLess(residual, 1e-12)
        self.assertAlmostEqual(squeezing_db(max(kappas)), 5.0, delta=.1)
        for pivot in pivots:
            self.assertAlmostEqual(pivot, 1.0, places=10)

    def test_random_targets_have_unit_pivots(self):
        for seed in range(3):
            with self.subTest(seed=seed):
                target, _ = random_bm_target(6, seed=seed, r_max=.5)
                _, pivots, residual = run_sequence(target, extra_gates=True)
                self.assertLess(residual, 1e-12)
                for pivot in pivots:
                    self.assertAlmostEqual(pivot, 1.0, places=10)

    def test_extra_gate_is_passive_and_clears_first_response(self):
        target, _ = random_bm_target(3, seed=5040, r_max=.5)
        gate = extra_passive_gate(target, 0, 1, 0, 3)
        identity = torch.eye(6, dtype=target.dtype)
        torch.testing.assert_close(gate.T @ gate, identity, atol=1e-13, rtol=0)
        torch.testing.assert_close(gate.T @ omega(3) @ gate, omega(3), atol=1e-13, rtol=0)
        self.assertLess(float(torch.linalg.norm((gate @ target)[:2, 0])), 1e-13)
        zero = torch.zeros_like(target)
        torch.testing.assert_close(extra_passive_gate(zero, 0, 1, 0, 3), identity)


if __name__ == "__main__":
    unittest.main()
