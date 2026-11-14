import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from experiment import (Archive, FEATURES, SHAPE, NumpyGLM, make_features,
                        neighbors, output_path, sample_mask, fit_baselines)
from probe_nasa import summarize_csv


class ExperimentTests(unittest.TestCase):
    def test_output_cannot_escape_research(self):
        with self.assertRaises(ValueError):
            output_path("../frontend/public/data/should_not_exist.json")

    def test_neighbors_do_not_wrap_across_grid_boundary(self):
        x = np.zeros((3, 3)); x[0, 0] = 4
        result = neighbors(x)
        self.assertEqual(result[0, 0], 0)
        self.assertEqual(result[0, 1], 4)
        self.assertEqual(result[2, 2], 0)

    def test_features_use_modis_and_past_context_only(self):
        m = np.zeros((92, *SHAPE), dtype=np.int32)
        m[0, 0, 0] = 3
        valid = np.ones(92, dtype=bool)
        before = make_features(m, 1, 2, valid)
        m[2:, :, :] = 100  # Future measurements must not change a day-2 prediction.
        after = make_features(m, 1, 2, valid)
        np.testing.assert_array_equal(before, after)
        self.assertEqual(before.shape[1], len(FEATURES))
        self.assertAlmostEqual(float(before[0, 5]), np.log1p(3), places=6)
        self.assertFalse(any("viirs" in name for name in FEATURES))

    def test_flagged_date_is_excluded_from_feature_window(self):
        m = np.zeros((92, *SHAPE), dtype=np.int32)
        m[0, 0, 0], m[1, 0, 0] = 100, 2
        valid = np.ones(92, dtype=bool); valid[0] = False
        x = make_features(m, 0, 2, valid)
        self.assertAlmostEqual(float(x[0, 0]), np.log1p(2), places=6)
        self.assertEqual(x[0, 1], 1)
        self.assertEqual(x[0, 8], 1)

    def test_oversampling_retains_positive_and_modis_only_cases_with_correct_weights(self):
        x = np.zeros((1000, len(FEATURES))); y = np.zeros(1000)
        x[0, 0] = 1; y[1] = 1
        selected, weights = sample_mask(x, y, np.ones(1000, bool),
                                        np.random.default_rng(42), .1)
        self.assertTrue(selected[0]); self.assertTrue(selected[1])
        np.testing.assert_array_equal(weights[:2], [1, 1])
        np.testing.assert_array_equal(weights[2:], np.full(len(weights) - 2, 10.))

    def test_partial_2026_overlap_rejected(self):
        with self.assertRaisesRegex(ValueError, "partial overlap"):
            Archive().valid_dates(2026)

    def test_static_mask_reads_only_supplied_training_years(self):
        archive = object.__new__(Archive)
        visited = []
        def arrays(year):
            visited.append(year)
            v = np.zeros((92, *SHAPE), dtype=np.int32)
            v[:6, 0, 0] = 1
            return np.zeros_like(v), v
        archive.arrays = arrays
        mask = archive.static_mask([2012, 2013, 2014])
        self.assertEqual(visited, [2012, 2013, 2014])
        self.assertTrue(mask[0, 0])
        allowed = np.ones(SHAPE, dtype=bool); allowed[0, 0] = False
        self.assertFalse(archive.static_mask([2012, 2013, 2014], allowed)[0, 0])

    def test_reference_direction_is_modis_to_viirs(self):
        rows = []
        for region in ["west", "central", "east", "canada_rectangle"]:
            rows += [{"region": region, "observed_modis": x, "observed_viirs": 3 * x}
                     for x in [10, 20, 40, 80]]
        fits = fit_baselines(rows)
        self.assertAlmostEqual(fits["west"]["ratio"], 3)

    def test_logistic_model_learns_and_returns_expected_occupancy(self):
        x = np.linspace(-3, 3, 300).reshape(-1, 1)
        y = (x[:, 0] > 0).astype(float)
        model = NumpyGLM("binary").fit(x, y, np.ones(len(y)))
        p = model.predict(x)
        self.assertTrue(np.all((p >= 0) & (p <= 1)))
        self.assertLess(p[0], .1); self.assertGreater(p[-1], .9)
        self.assertAlmostEqual(float(p.mean()), .5, places=4)
        self.assertTrue(all(b["loss"] <= a["loss"] + 1e-9
                            for a, b in zip(model.trace, model.trace[1:])))

    def test_weighted_logistic_fit_preserves_population_prevalence(self):
        x = np.zeros((20, 1)); y = np.r_[np.zeros(10), np.ones(10)]
        weights = np.r_[np.full(10, 9.), np.ones(10)]
        model = NumpyGLM("binary").fit(x, y, weights)
        self.assertAlmostEqual(float(model.predict(x).mean()), .1, places=6)

    def test_poisson_fit_supports_zeros_and_nonnegative_predictions(self):
        x = np.arange(30).reshape(-1, 1) / 10
        y = np.round(np.exp(x[:, 0]) / 4)
        model = NumpyGLM("poisson").fit(x, y, np.ones(len(y)))
        self.assertTrue(np.all(np.isfinite(model.predict(x))))
        self.assertTrue(np.all(model.predict(x) >= 0))

    def test_api_probe_rejects_non_csv_or_incomplete_schema(self):
        with self.assertRaises(ValueError):
            summarize_csv('<html>Service unavailable</html>')

    def test_api_probe_accepts_header_only_and_real_rows(self):
        header = 'latitude,longitude,acq_date,acq_time,satellite,instrument,confidence,frp,scan,track,daynight\n'
        self.assertEqual(summarize_csv(header)["rows"], 0)
        result = summarize_csv(header + '55,-123,2023-07-01,0100,N20,VIIRS,n,1.2,0.4,0.4,N\n')
        self.assertEqual(result["rows"], 1)
        self.assertEqual(result["satellites"], ["N20"])


if __name__ == "__main__":
    unittest.main()
