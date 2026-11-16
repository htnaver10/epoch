"""Scientific-integrity checks runnable with Python's built-in unittest runner."""
from __future__ import annotations

import unittest
import numpy as np

from dataset import TrainingData, check_extra_rows


class DataIsolationTests(unittest.TestCase):
    def row(self, split, source, group, window=(0, 0, 96, 96)):
        return {'split': split, 'event_group': group, 'source_ids': [source],
                'dataset_kind': 'nasa_native', 'source_window': list(window), 'interval_time_bound_seconds': 600}

    def test_heldout_source_is_rejected_across_directory_aliases(self):
        parent = [self.row('test', 'old/MYD021KM.same.hdf', 'test_group')]
        with self.assertRaisesRegex(ValueError, 'held-out'):
            check_extra_rows([self.row('train', 'new/MYD021KM.same.hdf', 'new_group')], parent)

    def test_heldout_cohort_is_rejected_even_with_different_sources(self):
        parent = [self.row('validation', 'MYD021KM.a.hdf', 'validation_group')]
        with self.assertRaisesRegex(ValueError, 'held-out'):
            check_extra_rows([self.row('train', 'MYD021KM.b.hdf', 'validation_group')], parent)

    def test_alternate_processing_of_same_heldout_acquisition_is_rejected(self):
        parent = [self.row('test', 'MYD021KM.A2024157.1845.061.old.hdf', 'test_group')]
        with self.assertRaisesRegex(ValueError, 'held-out'):
            check_extra_rows([self.row('train', 'MYD021KM.A2024157.1845.061.new.hdf', 'new_group')], parent)

    def test_duplicate_native_windows_are_rejected(self):
        row = self.row('train', 'MYD021KM.a.hdf', 'training')
        with self.assertRaisesRegex(ValueError, 'Duplicated'):
            check_extra_rows([row], [row])

    def test_independent_source_and_cohort_are_admitted(self):
        self.assertTrue(check_extra_rows([self.row('train', 'MYD021KM.b.hdf', 'new')],
                                         [self.row('test', 'MYD021KM.a.hdf', 'test')]))

    def test_importance_sampling_recovers_each_cohorts_natural_prevalence(self):
        data = TrainingData.__new__(TrainingData)
        data.y = np.r_[np.ones(2), np.zeros(98), np.ones(10), np.zeros(90)].astype(np.float32)
        data.pools = [(np.arange(2), np.arange(2, 100)), (np.arange(100, 110), np.arange(110, 200))]
        data.offsets = np.array([0, 100, 200])
        data.shards = [{'X': np.zeros((100, 1), np.float32)}, {'X': np.zeros((100, 1), np.float32)}]
        data.mean, data.scale = np.zeros(1, np.float32), np.ones(1, np.float32)
        _, y, correction, groups = data.draw(np.random.default_rng(7), 200000, .25)
        for group, expected_prevalence in enumerate((.02, .10)):
            selected = groups == group
            self.assertAlmostEqual(float((y[selected] * correction[selected]).mean()), expected_prevalence, delta=.002)
            self.assertAlmostEqual(float(correction[selected].mean()), 1., delta=.01)

    def test_three_shard_draw_preserves_sample_feature_label_alignment(self):
        data = TrainingData.__new__(TrainingData)
        data.offsets = np.array([0, 4, 9, 15])
        values = np.arange(15, dtype=np.float32) / 20
        data.y = (np.arange(15) % 2).astype(np.float32)
        data.shards = [{'X': values[a:b, None]} for a, b in zip(data.offsets[:-1], data.offsets[1:])]
        data.pools = [(np.flatnonzero(data.y == 1), np.flatnonzero(data.y == 0))]
        data.mean = np.zeros(1, np.float32); data.scale = np.ones(1, np.float32)
        x, y, correction, groups = data.draw(np.random.default_rng(34), 1000, .25)
        recovered = np.rint(x[:, 0] * 20).astype(int)
        np.testing.assert_array_equal(y, data.y[recovered])
        self.assertTrue(all(np.any((recovered >= a) & (recovered < b)) for a, b in zip(data.offsets[:-1], data.offsets[1:])))


if __name__ == '__main__':
    unittest.main()
