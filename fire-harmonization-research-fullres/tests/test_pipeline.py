import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from firefullres import core
from firefullres.geometry import inverse_geolocation,reference_targets
from firefullres.native import calibrated_values
from firefullres.metrics import BinaryMetrics
from firefullres.demo import make_demo
from firefullres.prepare import check_land_cover_metadata


class GeometryTests(unittest.TestCase):
    def test_forest_context_cannot_leak_future_even_in_all_land_mode(self):
        bundle={'land_cover':['data/raw/MCD12Q1/MCD12Q1.A2022001.h12v02.061.2023001000000.hdf'],
                'land_cover_year':2022,'land_cover_available_utc':'2023-06-01T00:00:00Z','modis_start_utc':'2023-07-01T00:00:00Z'}
        check_land_cover_metadata(bundle,False)
        bundle['land_cover_available_utc']='2024-01-01T00:00:00Z'
        with self.assertRaisesRegex(ValueError,'future'):check_land_cover_metadata(bundle,False)

    def test_forest_context_year_matches_file_and_is_required_for_forest_mode(self):
        with self.assertRaisesRegex(ValueError,'requires'):check_land_cover_metadata({},True)
        bundle={'land_cover':['data/raw/MCD12Q1/MCD12Q1.A2021001.h12v02.061.2023001000000.hdf'],
                'land_cover_year':2022,'land_cover_available_utc':'2023-06-01T00:00:00Z','modis_start_utc':'2023-07-01T00:00:00Z'}
        with self.assertRaisesRegex(ValueError,'disagree'):check_land_cover_metadata(bundle,False)
    def test_inverse_recovers_fractional_coordinates_without_future_or_label_inputs(self):
        y,x=np.indices((16,20));lat=50+.01*y+.0001*x;lon=-120+.02*x+.0002*y
        qr=np.array([[3.25,7.5]]);qc=np.array([[4.75,9.25]])
        grid,valid,error,row,col=inverse_geolocation(lat,lon,50+.01*qr+.0001*qc,-120+.02*qc+.0002*qr)
        self.assertTrue(valid.all());self.assertLess(error.max(),1e-5)
        np.testing.assert_allclose(row,qr,atol=1e-8);np.testing.assert_allclose(col,qc,atol=1e-8)
        self.assertEqual(grid.dtype,np.float32)

    def test_unknown_coordinates_and_extrapolation_have_no_support(self):
        y,x=np.indices((8,8));lat=50+.01*y;lon=-120+.02*x
        _,ok,_,_,_=inverse_geolocation(lat,lon,np.array([[np.nan,60.,50.03]]),np.array([[-120.,-120.,-119.94]]))
        np.testing.assert_array_equal(ok,[[False,False,True]])

    def test_labels_exclude_cloud_low_confidence_bowtie_and_halo(self):
        classes=np.array([[5,7,8,9,4]],np.uint8)
        grid=np.zeros((1,5,2),np.float32);qa=np.zeros((1,5),np.uint32);qa[0,3]=1<<22
        y,w,nt,nw=reference_targets(classes,qa,grid,np.ones(classes.shape,bool),(8,8),1)
        np.testing.assert_array_equal(w,[[1,0,1,0,0]])
        self.assertEqual(nw.sum(),2);self.assertEqual((nt*nw).sum(),1)

    def test_modis_scaled_integer_calibration_preserves_invalid_mask(self):
        raw=np.array([[[10,65535]],[[20,65533]]],np.uint16)
        meta={'radiance_scales':'0.1,0.2','radiance_offsets':'1,2','valid_range':'0,32767','_FillValue':'65535'}
        values,valid=calibrated_values(raw,meta,[0,1],'radiance')
        np.testing.assert_allclose(values[:,:,0],[ [.9],[3.6] ])
        np.testing.assert_array_equal(valid[:,:,1],0)
        with self.assertRaises(ValueError):calibrated_values(raw,{},[0,1],'radiance')


class MetricTests(unittest.TestCase):
    def test_weighted_losses_ranking_and_unknown_mask(self):
        m=BinaryMetrics();m.add([.1,.9,.5,.5,1.],[0,1,0,1,0],[1,1,1,1,0])
        r=m.result();self.assertAlmostEqual(r['brier'],.13)
        self.assertAlmostEqual(r['roc_auc'],.875);self.assertAlmostEqual(r['average_precision'],5/6)
        self.assertEqual(r['valid_query_weight'],4)


class SizedDemoTests(unittest.TestCase):
    def test_sized_fixture_retains_native_lattices_and_synthetic_guard(self):
        old=core.ROOT;(ROOT/'.cache').mkdir(exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(dir=ROOT/'.cache') as temp:
                core.ROOT=Path(temp)
                make_demo('sized.jsonl',size=48,query_size=96,train_tiles=2)
                with self.assertRaisesRegex(ValueError,'Synthetic'):
                    core.validate_tiles('sized.jsonl')
                rows=core.validate_tiles('sized.jsonl',True)
                self.assertEqual(len(rows),6)
                with np.load(core.path(rows[0]['tile'])) as arrays:
                    self.assertEqual(arrays['thermal'].shape,(12,48,48))
                    self.assertEqual(arrays['reflective_500'].shape,(10,96,96))
                    self.assertEqual(arrays['reflective_250'].shape,(4,192,192))
                    self.assertEqual(arrays['target'].shape,(96,96))
                    self.assertEqual(arrays['query_lat'].dtype,np.float64)
        finally:core.ROOT=old


class DatasetTests(unittest.TestCase):
    def setUp(self):
        (ROOT/'.cache').mkdir(exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=ROOT/'.cache');self.old=core.ROOT;core.ROOT=Path(self.tmp.name)
        make_demo('manifest.jsonl')
    def tearDown(self):core.ROOT=self.old;self.tmp.cleanup()
    def test_synthetic_default_rejection_and_hash_tampering(self):
        with self.assertRaisesRegex(ValueError,'Synthetic'):core.validate_tiles('manifest.jsonl')
        rows=core.validate_tiles('manifest.jsonl',True)
        self.assertEqual(len(rows),8)
        p=core.path(rows[0]['tile']);p.write_bytes(p.read_bytes()+b'tamper')
        with self.assertRaisesRegex(ValueError,'checksum'):core.validate_tiles('manifest.jsonl',True)
    def test_source_and_event_leakage_is_rejected(self):
        rows=core.read_jsonl('manifest.jsonl');rows[4]['event_group']=rows[0]['event_group']
        core.write_jsonl('bad.jsonl',rows)
        with self.assertRaisesRegex(ValueError,'leakage'):core.validate_tiles('bad.jsonl',True)
        rows[4]['event_group']='different';rows[4]['source_ids']=rows[0]['source_ids']
        core.write_jsonl('bad.jsonl',rows)
        with self.assertRaisesRegex(ValueError,'leakage'):core.validate_tiles('bad.jsonl',True)
    def test_statistics_use_training_only_and_preserve_precision(self):
        a=core.statistics('manifest.jsonl','stats.json',True)
        rows=core.read_jsonl('manifest.jsonl');r=rows[4]
        with np.load(core.path(r['tile'])) as z:arrays={k:z[k] for k in z.files}
        arrays['thermal'][:6]+=1000
        core.write_npz(r['tile'],**arrays);r['sha256']=core.sha256(r['tile']);core.write_jsonl('changed.jsonl',rows)
        b=core.statistics('changed.jsonl','stats2.json',True)
        self.assertEqual(a['channels'],b['channels'])
        self.assertEqual(len(a['training_groups']),4)
    def test_path_guard(self):
        with self.assertRaises(ValueError):core.path('../../backend/accidental.json',True)
    def test_nonfinite_auxiliary_or_wrong_target_semantics_fail_audit(self):
        row=core.read_jsonl('manifest.jsonl')[0]
        with np.load(core.path(row['tile'])) as z:arrays={k:z[k] for k in z.files}
        arrays['native_target'][0,0]=np.nan
        with self.assertRaisesRegex(ValueError,'targets'):core.validate_arrays(arrays)
        arrays['native_target'][0,0]=0;arrays['target'][0,0]=.5
        with self.assertRaisesRegex(ValueError,'binary'):core.validate_arrays(arrays)
    def test_utc_intervals_conservatively_bound_acquisitions(self):
        b={'modis_start_utc':'2023-01-01T00:00:00Z','modis_end_utc':'2023-01-01T00:05:00Z',
           'viirs_start_utc':'2023-01-01T00:04:00Z','viirs_end_utc':'2023-01-01T00:10:00Z'}
        self.assertEqual(core.time_bound_seconds(b),600)
        with self.assertRaises(ValueError):core.utc('2023-01-01T00:00:00')


if __name__=='__main__':unittest.main()
