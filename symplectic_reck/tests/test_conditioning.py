"""Check the zero-pivot target, nearby paths, and block-noise conventions."""

import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from symplectic_reck.experiments.run_conditioning import block_noise, perturbation_trial
from symplectic_reck.gates import DTYPE, mode_block
from symplectic_reck.matrix_utils import omega
from symplectic_reck.nulling_analytical import local_condition_number
from symplectic_reck.nulling_gateorder import nulling_schedule
from symplectic_reck.targets import (
    isotropic_failure_target, isotropic_path_generator, isotropic_path_target,
    random_bm_target,
)


class ConditioningTests(unittest.TestCase):
    def test_isotropic_target_is_symplectic_with_a_zero_pivot(self):
        S = isotropic_failure_target()
        J = omega(3)
        torch.testing.assert_close(S.T @ J @ S, J)
        step = nulling_schedule(3)[0]
        P = mode_block(S, step.anchor, step.probe)
        E = mode_block(S, step.target, step.probe)
        self.assertEqual(float(torch.linalg.det(P) + torch.linalg.det(E)), 0.)
        self.assertEqual(int(torch.linalg.matrix_rank(torch.cat((P, E)))), 2)

    def test_path_stays_symplectic_and_approaches_the_zero_pivot(self):
        S0 = isotropic_failure_target()
        J = omega(3)
        L = isotropic_path_generator(5040)
        torch.testing.assert_close(L.T @ J + J @ L, torch.zeros_like(J), atol=1e-12, rtol=0)
        torch.testing.assert_close(isotropic_path_target(0., L), S0)
        step = nulling_schedule(3)[0]
        distances, kappas = [], []
        alphas = (1e-4, 1e-5, 1e-6)
        for alpha in alphas:
            with self.subTest(alpha=alpha):
                S = isotropic_path_target(alpha, L)
                torch.testing.assert_close(S.T @ J @ S, J, atol=1e-12, rtol=0)
                distances.append(float(torch.linalg.norm(S - S0)))
                P = mode_block(S, step.anchor, step.probe)
                E = mode_block(S, step.target, step.probe)
                kappas.append(local_condition_number(P, E))
        for i in range(len(alphas) - 1):
            with self.subTest(alpha_pair=alphas[i:i+2]):
                self.assertAlmostEqual(distances[i] / distances[i+1], 10., delta=.02)
                self.assertAlmostEqual(kappas[i+1] / kappas[i], 10., delta=.02)

    def test_target_squeezing_is_in_the_requested_range(self):
        S, _ = random_bm_target(2, seed=5040, r_min=0.0, r_max=0.5)
        squeezing = torch.linalg.svdvals(S).log()[:2]
        self.assertTrue(bool(((squeezing >= 0) & (squeezing < 0.5)).all()))

    def test_each_noise_block_has_the_requested_norm(self):
        for eps in (1e-5, 1e-8, 1e-12):
            with self.subTest(eps=eps):
                generator = torch.Generator().manual_seed(13000)
                noise = block_noise(generator, eps, per_block=True)
                norms = torch.linalg.vector_norm(noise, dim=(1, 2))
                torch.testing.assert_close(norms, torch.full((2,), eps, dtype=DTYPE), atol=0, rtol=1e-12)

    def test_sweeps_save_the_expected_targets_and_perturbed_results(self):
        # Run a tiny real experiment and inspect its files, without patching functions.
        with tempfile.TemporaryDirectory() as out:
            command = [
                sys.executable, "-m", "symplectic_reck.experiments.run_conditioning",
                "--Ns", "2", "--N-trials", "1", "--trials", "1",
                "--alphas", "0.001", "--dbs", "4", "--out-dir", out,
            ]
            run = subprocess.run(
                command, cwd=Path(__file__).resolve().parents[2],
                capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
            targets = {
                "N_2_trial0": random_bm_target(2, seed=5040, r_max=.5)[0],
                "alpha_0.001_trial0": isotropic_path_target(.001, isotropic_path_generator(5040)),
                "db_4.0_trial0": random_bm_target(4, seed=5040, r_max=4 * math.log(10) / 20)[0],
            }
            with np.load(Path(out) / "matrices.npz") as matrices:
                for name, target in targets.items():
                    with self.subTest(sweep=name):
                        np.testing.assert_allclose(matrices[name + "__target"], target.numpy(), atol=1e-14, rtol=0)
                    for eps, suffix in ((1e-8, "e8"), (1e-12, "e12")):
                        with self.subTest(sweep=name, eps=eps):
                            expected = perturbation_trial(target, eps, 13000, per_block=True)["matrix"]
                            np.testing.assert_allclose(
                                matrices[name + "__final_" + suffix], expected.numpy(), atol=1e-14, rtol=1e-12,
                            )


if __name__ == "__main__":
    unittest.main()
