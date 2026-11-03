"""Check residual normalization and statistics of paired trial differences."""
import unittest
import numpy as np

from symplectic_reck.plotting.plot_insitu_vs_tomog_with_gate_errors import (
    displayed_eps, paired_difference_stats, residual_stats,
)


class CalibrationPlotTests(unittest.TestCase):
    def test_difference_quantiles_are_paired_and_signed(self):
        rows = [dict(eps=.01, tomo_10k=t, insitu_equalq=i)
                for t, i in [(0., 2.), (10., 1.), (11., 10.), (None, 5.), (4., None)]]
        actual = paired_difference_stats(rows, [.01])[:, 0]
        np.testing.assert_allclose(actual, np.quantile([-2., 9., 1.], [.5, .25, .75]))
        self.assertEqual(actual[0], 1.)
        self.assertLess(actual[1], 0.)
        self.assertNotEqual(actual[0], np.median([0., 10., 11.])-np.median([2., 1., 10.]))

    def test_range_and_no_second_normalization(self):
        rows = [dict(eps=e, tomo_10k=2., insitu_equalq=1.)
                for e in (.0003, .001, .005, .007, .015, .02, .1, .15, .2)]
        eps = displayed_eps(rows)
        self.assertEqual(eps, [.001, .005, .007, .015, .02, .1])
        np.testing.assert_array_equal(residual_stats(rows, eps, 'tomo_10k'), np.full((3, 6), 2.))
        np.testing.assert_array_equal(paired_difference_stats(rows, eps), np.ones((3, 6)))
        self.assertEqual(len(rows), 9)
