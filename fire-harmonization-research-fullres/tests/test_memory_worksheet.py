import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
import memory_worksheet as w
from gpu_preflight import destination


class WorksheetTests(unittest.TestCase):
    def test_attention_components_show_quadratic_cost(self):
        self.assertEqual(w.attention_bytes(1,8,256**2,96**2,2)/w.GIB,9.)
        self.assertEqual(w.attention_bytes(1,8,256**2,384**2,2)/w.GIB,144.)

    def test_adam_lower_bound(self):
        self.assertAlmostEqual(w.report()['fp32_adam_weights_gradients_two_moments_mib']['10000000'],152.587890625)

    def test_coordinate_rounding_exposes_low_precision_error(self):
        r=w.report()['coordinate_cast_example']['rounding']
        error={x['dtype']:x['ground_displacement_m'] for x in r}
        self.assertEqual(error['float64'],0.)
        self.assertLess(error['float32'],1.)
        self.assertGreater(error['float16'],1000.)
        self.assertGreater(error['bfloat16'],10000.)

    def test_known_bfloat16_rounding(self):
        self.assertEqual(w.round_bfloat16(60.11),60.)
        self.assertEqual(w.round_bfloat16(-120.24),-120.)

    def test_input_payload_matches_plan_arithmetic(self):
        r=w.report()['input_payload']
        self.assertAlmostEqual(r['current_tile_mib'],2.5078125)
        self.assertAlmostEqual(r['with_two_extra_frames_mib'],2.9296875)

    def test_report_is_not_a_gpu_profile(self):
        r=w.report()
        self.assertIn('not_gpu_profile',r['kind'])
        self.assertTrue(r['no_native_model_trained'])

    def test_output_guard(self):
        with self.assertRaises(ValueError):destination('../../backend/accidental.json')


if __name__=='__main__':unittest.main()
