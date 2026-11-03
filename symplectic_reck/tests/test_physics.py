"""Check the physics and compare fixed-seed runs with saved reference results.

The reference arrays use xpxp order with N=3, target seed 20, r in [0, 0.5],
noise seed 1000, and three optimizer updates. Calibration uses eps=0.001,
error seed 10090, shot seed 13000, and q=10000."""

import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from symplectic_reck.experiments.run_finite_shots import (
    experiment_settings, measurement_budgets, run_trial,
)
from symplectic_reck.gate_errors import draw_gate_errors
from symplectic_reck.gates import DTYPE, embed_two_mode, mode_block, single_mode_gate, two_mode_gate
from symplectic_reck.matrix_utils import omega, project_symplectic
from symplectic_reck.measurements import estimate_block, estimate_transfer_matrix, homodyne_mean
from symplectic_reck.nulling_analytical import analytical_nulling_gate, projected_single_mode_inverse
from symplectic_reck.nulling_feedback import dither_tones, exact_loss, lockin_gradient, two_mode_problem
from symplectic_reck.nulling_gateorder import nulling_schedule
from symplectic_reck.run_nulling_sequence import (
    run_analytical_sequence, run_autodiff_sequence, run_calibration_sequence,
)
from symplectic_reck.targets import random_bm_target


def assert_symplectic(M, atol=1e-11):
    J = omega(M.shape[0] // 2)
    torch.testing.assert_close(M.T @ J @ M, J, atol=atol, rtol=0)


class PhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with np.load(Path(__file__).with_name("fixed_seed_reference.npz")) as arrays:
            cls.ref = {name: arrays[name] for name in arrays.files}
        cls.S, _ = random_bm_target(3, seed=20, r_min=0.0, r_max=0.5)

    def test_gates_are_symplectic(self):
        gates = {
            "two_mode": two_mode_gate(torch.linspace(-0.2, 0.3, 10, dtype=DTYPE)),
            "single_mode": single_mode_gate(torch.linspace(-0.2, 0.3, 3, dtype=DTYPE)),
        }
        for name, gate in gates.items():
            with self.subTest(gate=name):
                assert_symplectic(gate)

    def test_random_targets_are_symplectic(self):
        for N in (2, 3, 4, 9):
            with self.subTest(N=N, seed=20):
                S, _ = random_bm_target(N, seed=20, r_min=0.0, r_max=0.5)
                assert_symplectic(S)

    def test_each_nulling_gate_clears_its_target_block(self):
        for N in (2, 3, 4, 9):
            M, _ = random_bm_target(N, seed=20, r_min=0.0, r_max=0.5)
            for step in nulling_schedule(N):
                with self.subTest(N=N, step=step.index):
                    target = mode_block(M, step.target, step.probe)
                    anchor = mode_block(M, step.anchor, step.probe)
                    gate = analytical_nulling_gate(target, anchor)
                    assert_symplectic(gate)
                    M = embed_two_mode(gate, step.target, step.anchor, N) @ M
                    remaining = mode_block(M, step.target, step.probe)
                    self.assertLess(float(torch.linalg.norm(remaining)), 1e-10)

    def test_full_nulling_returns_identity(self):
        for N in (2, 3, 4, 9):
            with self.subTest(N=N, seed=20):
                S, _ = random_bm_target(N, seed=20, r_min=0.0, r_max=0.5)
                final, _ = run_analytical_sequence(S, N)
                torch.testing.assert_close(final, torch.eye(2 * N, dtype=DTYPE), atol=1e-10, rtol=0)

    def test_projection_preserves_symplectic_matrices(self):
        S, _ = random_bm_target(5, seed=7, r_min=0.0, r_max=0.5)
        torch.testing.assert_close(project_symplectic(S), S, atol=1e-12, rtol=0)

    def test_projection_corrects_small_matrix_errors(self):
        S, _ = random_bm_target(5, seed=7, r_min=0.0, r_max=0.5)
        gen = torch.Generator().manual_seed(3)
        for scale in (1e-3, 1e-2):
            with self.subTest(scale=scale):
                noisy = S + scale * torch.randn(10, 10, dtype=DTYPE, generator=gen)
                projected = project_symplectic(noisy)
                assert_symplectic(projected)
                self.assertLess(float(torch.linalg.norm(projected - S)), float(torch.linalg.norm(noisy - S)))

    def test_single_mode_projection_handles_reflections(self):
        block = torch.tensor([[1.1, 0.2], [0.1, 0.8]], dtype=DTYPE)
        assert_symplectic(projected_single_mode_inverse(block), atol=1e-12)
        reflected = torch.tensor([[1.0, 0.0], [0.0, -0.9]], dtype=DTYPE)
        assert_symplectic(projected_single_mode_inverse(reflected), atol=1e-12)

    def test_homodyne_vacuum_variance(self):
        displacement = torch.tensor([0.7, -0.2], dtype=DTYPE)
        examples = {
            "identity": torch.eye(2, dtype=DTYPE),
            "squeezing": torch.tensor([[2., 0.], [0., .5]], dtype=DTYPE),
            "shear": torch.tensor([[1., .7], [0., 1.]], dtype=DTYPE),
        }
        for name, M in examples.items():
            for shots in (1, 100):
                with self.subTest(transformation=name, shots=shots):
                    expected_mean = float((M @ displacement)[0])
                    samples = np.array([float(homodyne_mean(M, displacement, 0, shots, seed))
                                        for seed in range(6000)])
                    expected_variance = float((M @ M.T)[0, 0]) / (2 * shots)
                    self.assertAlmostEqual(samples.var(ddof=1) / expected_variance, 1., delta=.06)
                    self.assertLess(abs(samples.mean()-expected_mean), 5 * np.sqrt(expected_variance / len(samples)))
                    self.assertEqual(float(homodyne_mean(M, displacement, 0, None)), expected_mean)

    def test_measurements_match_reference(self):
        S = self.S
        np.testing.assert_allclose(S.numpy(), self.ref["target"], atol=1e-14)
        for shots in (None, 1000):
            with self.subTest(shots=shots):
                np.testing.assert_allclose(estimate_block(S, 3, 2, 0, shots, 104).numpy(), self.ref[f"block_{shots}"], atol=1e-13)
        np.testing.assert_allclose(estimate_transfer_matrix(S, 3, 1000, 104).numpy(), self.ref["tomography"], atol=1e-13)
        torch.testing.assert_close(estimate_transfer_matrix(S, 3, None, 104), S, atol=0, rtol=0)

    def test_fixed_seed_methods_match_reference(self):
        args = Namespace(trials=1, seed_start=20, noise_seed_start=1000, steps_per_gate=3, max_workers=1,
                         dither_amp=0.05)
        specs = [("analytical", None), ("measured_closed_form", 1000), ("finite_diff", 1000), ("genetic_algorithm", 1000)]
        for row in run_trial(0, experiment_settings(args, 3), specs):
            with self.subTest(method=row["method"]):
                self.assertIsNone(row["failure"])
                np.testing.assert_allclose(row["final_matrix"], self.ref[row["method"]], atol=2e-10)

    def test_lockin_gradient_matches_exact_gradient(self):
        for width in (3, 10):
            refs, T = dither_tones(width)
            with self.subTest(width=width):
                torch.testing.assert_close(refs @ refs.T, T / 2 * torch.eye(width, dtype=DTYPE), atol=1e-12, rtol=0)
        step = nulling_schedule(3)[0]
        problem = two_mode_problem(3, step.target, step.anchor, step.probe)
        params = torch.linspace(-0.2, 0.3, 10, dtype=DTYPE)
        grad = lockin_gradient(problem, self.S, params, refs, 1e-4, None, 0)
        eps = 1e-6
        exact = torch.tensor([(exact_loss(problem, self.S, params + eps * e) - exact_loss(problem, self.S, params - eps * e))
                              / (2 * eps) for e in torch.eye(10, dtype=DTYPE)], dtype=DTYPE)
        torch.testing.assert_close(grad, exact, atol=1e-6, rtol=1e-6)

    def test_autodiff_matches_reference(self):
        result = run_autodiff_sequence(self.S, 3, steps_per_gate=2, singlemode_steps=2)
        np.testing.assert_allclose(result["final_matrix"].numpy(), self.ref["autodiff"], atol=1e-9)

    def test_calibration_matches_reference(self):
        offsets = draw_gate_errors(0.001, 10090, 3, 3)
        insitu = run_calibration_sequence("insitu", self.S, 10000, 13000, offsets)
        np.testing.assert_allclose(insitu.numpy(), self.ref["insitu"], atol=2e-10)
        tomo = run_calibration_sequence("tomo", self.S, 10000, 13000, offsets)
        self.assertTrue(torch.isfinite(tomo).all())

    def test_calibration_without_errors_returns_identity(self):
        zero = draw_gate_errors(0.0, 10090, 3, 3)
        for arm in ("tomo", "insitu"):
            with self.subTest(method=arm):
                final = run_calibration_sequence(arm, self.S, None, 13000, zero)
                torch.testing.assert_close(final, torch.eye(6, dtype=DTYPE), atol=1e-10, rtol=0)

    def test_parallel_single_modes_update_together(self):
        N = 4
        target, _ = random_bm_target(N, seed=42, r_min=0.3, r_max=0.4)
        states = []
        final_stage = len(nulling_schedule(N))

        def record(stage, iteration, matrix, loss, gradient_norm):
            if stage == final_stage:
                states.append(matrix)

        settings = dict(steps_per_gate=4, singlemode_steps=8)
        sequential = run_autodiff_sequence(target, N, **settings)
        parallel = run_autodiff_sequence(
            target, N, **settings, parallel_single_modes=True, trace_callback=record,
        )
        # Independent modes should reach the same result in either order.
        torch.testing.assert_close(
            parallel["final_matrix"], sequential["final_matrix"], atol=1e-12, rtol=1e-12,
        )
        self.assertEqual(len(states), settings["singlemode_steps"])
        for mode in range(N):
            with self.subTest(mode=mode):
                before = mode_block(states[0], mode, mode)
                after = mode_block(states[1], mode, mode)
                self.assertGreater(float(torch.linalg.norm(after - before)), 1e-8)
        assert_symplectic(parallel["final_matrix"])

    def test_standard_measurement_budget(self):
        args = Namespace(trials=1, seed_start=20, noise_seed_start=1000, steps_per_gate=100, max_workers=1,
                         dither_amp=0.05)
        methods = ("measured_closed_form", "finite_diff", "genetic_algorithm", "dither_lockin")
        rows = measurement_budgets(experiment_settings(args, 3), [(m, 1000) for m in methods])
        self.assertEqual(rows[1]["total_raw_homodyne_samples"], 96096000)
        self.assertEqual(rows[2]["total_raw_homodyne_samples"], 96000000)
        self.assertEqual(rows[3]["total_raw_homodyne_samples"], 96000000)
        self.assertLess(abs(rows[0]["total_raw_homodyne_samples"] - 96000000), 72)


if __name__ == "__main__":
    unittest.main()
