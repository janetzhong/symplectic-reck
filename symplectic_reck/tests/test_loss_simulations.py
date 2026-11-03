"""Check attenuation, mode separation, and the optical-loss reference results."""

import math
import unittest

import numpy as np
import torch

from symplectic_reck.experiments.run_loss_simulations import off_block_fraction, train_lossy_response
from symplectic_reck.targets import random_bm_target


class LossSimulationTests(unittest.TestCase):
    def test_uniform_loss_preserves_attenuation_and_separation(self):
        target, _ = random_bm_target(5, seed=5040, r_max=.5)
        original = target.clone()
        for loss in (0., .1, .2, .4):
            with self.subTest(power_loss=loss):
                response = train_lossy_response(target, "uniform", loss)
                expected = math.sqrt(1-loss) * torch.eye(10, dtype=target.dtype)
                torch.testing.assert_close(response, expected, atol=1e-11, rtol=1e-11)
        torch.testing.assert_close(target, original, atol=0, rtol=0)

    def test_loss_is_power_and_applies_only_to_participating_modes(self):
        # For an identity target all nullers are identities. N=3 visits
        # modes 0,1,2 respectively 1,3,2 times in the Reck schedule.
        target = torch.eye(6, dtype=torch.float64)
        response = train_lossy_response(target, "per_gate", .2)
        expected = torch.diag(torch.tensor(
            [math.sqrt(.8)**n for n in (1,1,3,3,2,2)], dtype=target.dtype))
        torch.testing.assert_close(response, expected, atol=1e-13, rtol=1e-13)
        self.assertEqual(off_block_fraction(response), 0.)

    def test_per_gate_loss_leaves_upper_block_mixing(self):
        target, _ = random_bm_target(10, seed=5040, r_max=.5)
        response = train_lossy_response(target, "per_gate", .05)
        for row in range(10):
            with self.subTest(output_mode=row):
                self.assertLess(float(torch.linalg.norm(response[2*row:2*row+2, :2*row])), 1e-11)
        self.assertGreater(off_block_fraction(response), .05)

    def test_mean_mode_mixing_matches_reference_values(self):
        targets = [random_bm_target(10, seed=5040+t, r_max=.5)[0] for t in range(30)]
        for loss, expected in ((.01, .0200408), (.05, .105829)):
            with self.subTest(power_loss=loss):
                actual = np.mean([off_block_fraction(train_lossy_response(t, "per_gate", loss))
                                  for t in targets])
                self.assertAlmostEqual(actual, expected, delta=5e-7)

    def test_invalid_loss(self):
        target = torch.eye(4, dtype=torch.float64)
        for loss in (-.1, 1., float("nan")):
            with self.subTest(power_loss=loss):
                with self.assertRaises(ValueError):
                    train_lossy_response(target, "per_gate", loss)

    def test_matched_dummy_padding_removes_mixing_with_expected_transmission(self):
        target, _ = random_bm_target(10, seed=5040, r_max=.5)
        for loss in (.01, .05):
            with self.subTest(power_loss=loss):
                plan = np.full((45, 10), loss)
                padded = train_lossy_response(target, "per_gate", loss, loss_plan=plan)
                expected = (1-loss)**(45/2) * torch.eye(20, dtype=target.dtype)
                torch.testing.assert_close(padded, expected, atol=1e-11, rtol=1e-11)
                self.assertLess(off_block_fraction(padded), 1e-11)
                self.assertAlmostEqual(float(torch.sum(padded.square()) / 20), (1-loss)**45)
                unpadded = train_lossy_response(target, "per_gate", loss)
                self.assertGreater(off_block_fraction(unpadded), .01)


if __name__ == "__main__":
    unittest.main()
