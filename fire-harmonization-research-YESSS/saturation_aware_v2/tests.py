"""Regression tests for the actual saturation, boundary and calibration failures."""
import json
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
import common
from features import coarse,fine,INPUT_KEYS
from inference import localize
from policy import residual_score,selected_channel,alert_cutoff

def sample():
    rng=np.random.default_rng(6);h=48
    a={'thermal':rng.uniform(.1,2,(12,h,h)).astype(np.float32),
       'reflective_500':rng.uniform(.1,.8,(10,2*h,2*h)).astype(np.float32),
       'reflective_250':rng.uniform(.1,.8,(4,4*h,4*h)).astype(np.float32),
       'geometry':np.zeros((14,h,h),np.float32)}
    a['thermal'][6:]=1;a['reflective_500'][5:]=1;a['reflective_250'][2:]=1
    a['geometry'][1]=1;a['geometry'][5]=1;a['geometry'][8:12]=1
    return a

class CorrectedTests(unittest.TestCase):
    def test_saturated_20_and_22_keep_available_21(self):
        a=sample();a['thermal'][6,24,24]=0;a['thermal'][8,24,24]=0
        x,y,valid,day,names,tv=coarse(a)
        self.assertTrue(valid[24,24]);np.testing.assert_array_equal(tv[24,24],[True,False])
        self.assertEqual(selected_channel(tv)[24,24],0)

    def test_22_available_when_21_unavailable(self):
        a=sample();a['thermal'][7,24,24]=0
        self.assertTrue(coarse(a)[2][24,24]);self.assertEqual(selected_channel(coarse(a)[5])[24,24],1)

    def test_both_channels_missing_are_unknown(self):
        a=sample();a['thermal'][[7,8],24,24]=0
        self.assertFalse(coarse(a)[2][24,24])

    def test_predictor_cannot_copy_center_or_center_validity(self):
        a=sample();before=coarse(a)[0][24,24].copy()
        a['thermal'][[1,2],24,24]=100;a['thermal'][8,24,24]=0
        np.testing.assert_allclose(before,coarse(a)[0][24,24],atol=1e-5)

    def test_original_query_boundary_has_context_support(self):
        a=sample();self.assertTrue(coarse(a)[2][32,32])
        self.assertFalse(coarse(a)[2][5,5])

    def test_channel_selection_does_not_require_two_positive_residuals(self):
        g={'residual_center':[0,0],'residual_scale':[1,1]}
        e=np.array([[8.,-99.],[10.,2.],[3.,4.]])
        tv=np.array([[1,0],[1,1],[0,1]],bool)
        np.testing.assert_array_equal(residual_score(e,g,'coarse',tv),[8,2,4])

    def test_unobserved_target_has_zero_gradient(self):
        from model import masked_loss
        prediction=torch.tensor([[2.,3.]],requires_grad=True)
        loss=masked_loss(prediction,torch.tensor([[1.,-999999.]]),torch.tensor([[1.,0.]]));loss.backward()
        self.assertEqual(float(prediction.grad[0,1]),0.);self.assertGreater(float(prediction.grad[0,0]),0.)

    def test_calibration_precision_and_tied_budget(self):
        e=np.array([[1,2],[2,3],[2,3],[4,5]],np.float32);tv=np.ones_like(e,bool)
        g={'residual_center':[.1,.2],'residual_scale':[.3,.4]}
        score=residual_score(e,g,'coarse',tv);self.assertEqual(score.dtype,np.float64)
        np.testing.assert_array_equal(score,residual_score(e,json.loads(json.dumps(g)),'coarse',tv))
        self.assertLessEqual((score>=alert_cutoff(score,.5)).sum(),2)

    def test_reference_arrays_rejected(self):
        a=sample();a['target']=np.ones((48,48))
        with self.assertRaises(ValueError):coarse(a)

    def test_no_central_swir_leak(self):
        a=sample();before=fine(a)[0][48,48].copy();a['reflective_500'][4,48,48]=100
        np.testing.assert_allclose(before,fine(a)[0][48,48],atol=1e-5)

    def test_night_keeps_coarse_and_has_no_fine_evidence(self):
        a=sample();a['geometry'][5]=-1
        self.assertTrue(coarse(a)[2].any());self.assertFalse(fine(a)[2].any())

    def test_localization_preserves_parent_alert_and_weights(self):
        parent=np.array([[True,False]]);s=np.array([[1.,2.,4.,5.],[3.,4.,6.,7.]])
        detection,w,supported,_=localize(parent,s,np.ones_like(s,bool),3.)
        np.testing.assert_array_equal(detection.reshape(1,2,2,2).any((1,3)),parent)
        np.testing.assert_allclose(w.reshape(1,2,2,2).sum((1,3)),1.)
        self.assertEqual(detection.sum(),2)

    def test_path_cannot_overwrite_original_model(self):
        with self.assertRaises(ValueError):common.path('../models/coarse/inference.pt')

    def test_metric_helper_does_not_redirect_report_or_verify_imports(self):
        import evaluation
        before=list(sys.path)
        evaluation.summarize(np.array([1,0],bool),np.array([1,0],bool),np.array([2.,1.]))
        self.assertEqual(sys.path,before)
        import report,verify
        self.assertEqual(Path(report.__file__).resolve().parent,Path(__file__).resolve().parent)
        self.assertEqual(Path(verify.__file__).resolve().parent,Path(__file__).resolve().parent)

    def test_masked_training_export_reload_and_completed_reuse(self):
        from model import train,predict
        cfg=common.config();cfg['training'].update(device='cpu',epochs=12,patience=4,batch_size=64,learning_rate=.01)
        rng=np.random.default_rng(8);x=rng.normal(size=(300,3)).astype(np.float32)
        y=np.column_stack([x[:,0]+x[:,1],x[:,1]-x[:,2]]).astype(np.float32)
        observed=np.ones_like(y,bool);observed[:50,1]=False;y[:50,1]=-1000
        with tempfile.TemporaryDirectory() as tmp,patch.object(common,'ROOT',Path(tmp)):
            common.write('config.json',cfg);common.write('data/features/receipt.json',{'fixture':True})
            stats={'input_mean':[0.,0.,0.],'input_scale':[1.,1.,1.],'target_center':[0.,0.],'target_scale':[1.,1.],'feature_names':['a','b','c']}
            common.write('data/features/coarse_stats.json',stats)
            for split,sl in [('train',slice(0,200)),('development',slice(200,300))]:
                for name,value in [('X',x),('measured',y),('target_valid',observed)]:common.save_array(f'data/features/coarse_{split}_{name}.npy',value[sl])
            result=train('coarse');prediction=predict('coarse',x[200:])
            self.assertLess(np.mean((prediction-y[200:])**2),.2);self.assertEqual(train('coarse'),result)
            self.assertFalse(result['uses_viirs_labels'])

if __name__=='__main__':unittest.main()
