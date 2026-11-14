import sys
import unittest
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from kernel_experiment import box_sum, fraction_table, response, fit_alpha, alpha_for, output_path


class KernelTests(unittest.TestCase):
    def test_box_sum_has_no_spatial_or_temporal_wrap(self):
        x = np.zeros((2, 4, 5), bool); x[0, 0, 0] = True
        y = box_sum(x, 1)
        self.assertEqual(y[0, :2, :2].sum(), 4)
        self.assertEqual(y[0, -1, -1], 0)
        self.assertEqual(y[1].sum(), 0)

    def test_edge_denominators_are_observable_cells(self):
        d = box_sum(np.ones((4, 5), bool), 1)
        self.assertEqual(d[0, 0], 4)
        self.assertEqual(d[1, 1], 9)

    def test_exact_fraction_lookup(self):
        for radius in (1, 2, 4):
            f, lookup = fraction_table(radius)
            for n in range(1, (2*radius+1)**2+1):
                for k in range(n+1):
                    self.assertAlmostEqual(f[lookup[k, n]], k/n, places=11)

    def test_response_is_zero_at_no_signal_and_bounded_monotone(self):
        x = np.linspace(0, 1, 100)
        for b in (.5, .75, 1.):
            p = response(x, 2., b)
            self.assertEqual(p[0], 0)
            self.assertTrue(np.all((p >= 0) & (p <= 1)))
            self.assertTrue(np.all(np.diff(p) >= 0))

    def test_moment_fit_recovers_known_parameter(self):
        f = np.array([0., .1, .4, 1.]); h = np.array([10000., 20., 10., 5.])
        target = float(h @ response(f, 2.5, .75))
        fitted = fit_alpha(h, target, f, .75)
        self.assertAlmostEqual(fitted, 2.5, places=9)

    def test_fit_rejects_unreachable_activity(self):
        with self.assertRaises(ValueError):
            fit_alpha(np.array([100., 2.]), 3., np.array([0., 1.]), .75)

    def test_unseen_region_uses_pooled_response(self):
        m = {"pooled_alpha": 2., "regional_alpha": {"west": 8.}}
        self.assertEqual(alpha_for(m, "east", 1.), 2.)
        self.assertAlmostEqual(alpha_for(m, "west", .5), 4.)

    def test_output_guard(self):
        with self.assertRaises(ValueError):
            output_path("../../frontend/public/data/no.json")


if __name__ == "__main__":
    unittest.main()
