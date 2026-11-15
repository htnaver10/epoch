"""Tests of blind spots, reference separation, alert policy and deployable training."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fireyes import common
from fireyes.features import coarse,fine,ring
from fireyes.inference import localize
from fireyes.policy import alert_cutoff,residual_score

def sample():
    rng=np.random.default_rng(5);h=48
    a={'thermal':rng.uniform(.1,2,(12,h,h)).astype(np.float32),
       'reflective_500':rng.uniform(.1,.8,(10,2*h,2*h)).astype(np.float32),
       'reflective_250':rng.uniform(.1,.8,(4,4*h,4*h)).astype(np.float32),
       'geometry':np.zeros((14,h,h),np.float32)}
    a['thermal'][6:]=1;a['reflective_500'][5:]=1;a['reflective_250'][2:]=1
    a['geometry'][1]=1;a['geometry'][5]=1;a['geometry'][8:12]=1
    return a

class PipelineTests(unittest.TestCase):
    def test_coarse_cannot_copy_thermal_center(self):
        a=sample();before=coarse(a);a['thermal'][[0,2],24,24]=100
        after=coarse(a)
        np.testing.assert_allclose(before[0][24,24],after[0][24,24],atol=1e-5)
        self.assertFalse(np.array_equal(before[1][24,24],after[1][24,24]))
        self.assertEqual(before[0].shape[-1],64)

    def test_fine_cannot_copy_swir_center(self):
        a=sample();before=fine(a);a['reflective_500'][4,48,48]=100
        after=fine(a)
        np.testing.assert_allclose(before[0][48,48],after[0][48,48],atol=1e-5)
        self.assertNotEqual(float(before[1][48,48,0]),float(after[1][48,48,0]))
        self.assertEqual(before[0].shape[-1],44)

    def test_rejects_fire_labels_as_inputs(self):
        a=sample();a['target']=np.zeros((48,48))
        with self.assertRaises(ValueError):coarse(a)

    def test_night_has_no_fine_localization(self):
        a=sample();a['geometry'][5]=-1
        self.assertTrue(coarse(a)[2].any());self.assertFalse(fine(a)[2].any())

    def test_invalid_radiance_is_excluded_from_context(self):
        a=np.ones((15,15));v=np.ones_like(a);v[3,3]=0
        before=ring(a,v,7);a[3,3]=1e9;after=ring(a,v,7)
        for x,y in zip(before,after):np.testing.assert_array_equal(x,y)

    def test_localization_retains_two_and_conserves_mass(self):
        parent=np.array([[True,False]]);scores=np.array([[1.,2.,50.,60.],[3.,4.,70.,80.]])
        detections,mass,support,entropy=localize(parent,scores,np.ones_like(scores,bool),3.)
        self.assertEqual(detections[:,:2].sum(),2);self.assertEqual(detections[:,2:].sum(),0)
        np.testing.assert_allclose(mass.reshape(1,2,2,2).sum((1,3)),1)
        np.testing.assert_array_equal(detections.reshape(1,2,2,2).any((1,3)),parent)
        self.assertTrue(support[0,0])

    def test_insufficient_evidence_remains_unresolved(self):
        parent=np.ones((1,1),bool);s=np.arange(4.).reshape(2,2);v=np.ones((2,2),bool);v[0,0]=False
        detect,mass,support,entropy=localize(parent,s,v,2.)
        self.assertTrue(detect.all());self.assertFalse(support.any());np.testing.assert_array_equal(mass,np.full((2,2),.25))
        np.testing.assert_allclose(entropy,1.)

    def test_tied_alerts_never_exceed_budget(self):
        scores=np.array([5.,5.,5.,2.,1.]);cut=alert_cutoff(scores,.4)
        self.assertLessEqual((scores>=cut).sum(),2)

    def test_float32_calibration_matches_json_inference(self):
        error=np.array([[1.,2.],[2.,3.],[2.,3.],[4.,5.]],np.float32)
        center=np.array([.1,.2],np.float32);scale=np.array([.3,.4],np.float32)
        calibration=residual_score(error,center,scale,'coarse')
        inference=residual_score(error,center.tolist(),scale.tolist(),'coarse')
        self.assertEqual(calibration.dtype,np.float64)
        np.testing.assert_array_equal(calibration,inference)
        cutoff=alert_cutoff(calibration,.5)
        self.assertLessEqual((calibration>=cutoff).sum(),2)

    def test_path_guard(self):
        with self.assertRaises(ValueError):common.path('../outside.npz')

    def test_modis_view_rejects_reference_array(self):
        from fireyes.data import load_inputs
        with tempfile.TemporaryDirectory() as tmp,patch.object(common,'ROOT',Path(tmp)):
            a=sample();a['weight']=np.ones((48,48));common.save_npz('bad.npz',**a)
            with self.assertRaises(ValueError):load_inputs({'input':'bad.npz'})

    def test_reconstruction_training_export_and_resume(self):
        from fireyes.model import train,predict
        cfg=common.config();cfg['training'].update(device='cpu',epochs=12,patience=4,batch_size=64,learning_rate=.01)
        rng=np.random.default_rng(8);x=rng.normal(size=(300,3)).astype(np.float32)
        y=np.column_stack([x[:,0]+x[:,1],x[:,1]-x[:,2]]).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp,patch.object(common,'ROOT',Path(tmp)):
            common.write('config.json',cfg);common.write('data/features/receipt.json',{'test_fixture':True})
            stats={'input_mean':[0.,0.,0.],'input_scale':[1.,1.,1.],'target_center':[0.,0.],'target_scale':[1.,1.],
                   'feature_names':['a','b','c'],'fit_split':'train'}
            common.write('data/features/coarse_stats.json',stats)
            for split,sl in [('train',slice(0,200)),('validation',slice(200,300))]:
                common.save_array(f'data/features/coarse_{split}_X.npy',x[sl]);common.save_array(f'data/features/coarse_{split}_measured.npy',y[sl])
            result=train('coarse');pred=predict('coarse',x[200:]);self.assertEqual(pred.shape,(100,2))
            self.assertLess(np.mean((pred-y[200:])**2),.2)
            self.assertFalse(result['uses_viirs_labels']);self.assertEqual(train('coarse'),result)

if __name__=='__main__':unittest.main()
