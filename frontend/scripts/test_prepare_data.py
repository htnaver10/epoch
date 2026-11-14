import csv
import tempfile
import unittest
from pathlib import Path
from prepare_data import aggregate

class AggregationTests(unittest.TestCase):
    def test_shared_cell_preserves_raw_counts_and_separates_dates(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'observations.csv'
            with path.open('w') as stream:
                writer = csv.writer(stream)
                writer.writerow(['latitude', 'longitude', 'acq_date', 'instrument'])
                writer.writerows([
                    [50.01, -100.01, '2025-06-01', 'MODIS'],
                    [50.02, -100.02, '2025-06-01', 'VIIRS'],
                    [50.03, -100.03, '2025-06-01', 'VIIRS'],
                    [50.01, -100.01, '2025-06-02', 'MODIS'],
                    [50.25, -100, '2025-06-01', 'VIIRS'],
                    [95, -100, '2025-06-01', 'MODIS'],
                    [50, -100, '2025-06-01', 'UNKNOWN'],
                ])
            result = aggregate(path)
            self.assertEqual(result['total'], 5)
            self.assertEqual(len(result['records']), 3)
            self.assertIn(['2025-06-01', 50.125, -100.125, 1, 2], result['records'])
            self.assertIn(['2025-06-01', 50.375, -99.875, 0, 1], result['records'])
            self.assertIn(['2025-06-02', 50.125, -100.125, 1, 0], result['records'])

if __name__ == '__main__':
    unittest.main()
