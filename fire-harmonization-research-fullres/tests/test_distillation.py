"""Native calibration, split isolation, loss mathematics and student-only export."""
import copy
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from firefullres import core
from firefullres.demo import make_demo
from firefullres.distill import data
from firefullres.distill.history import reconcile_history
HAS_TORCH=importlib.util.find_spec('torch') is not None
if HAS_TORCH:
    import torch
    from firefullres import engine
    from firefullres.distill import teacher,student
    from firefullres.distill.models import VIIRSTeacher,frozen_teacher,bernoulli_kl_sum
    from firefullres.runtime import resolve_config


@unittest.skipUnless(HAS_TORCH,'PyTorch is not installed in the data-only environment')
class DistillationMathTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(7)

    def test_both_bernoulli_classes_temperature_gradient_and_frozen_target(self):
        logits=torch.tensor([-.7,.5,1.2],requires_grad=True)
        teach=torch.tensor([2.,-1.,.1],requires_grad=True);weights=torch.tensor([1.,0.,3.]);temperature=2.
        loss=bernoulli_kl_sum(logits,teach,weights,temperature)
        loss.backward()
        expected=temperature*(torch.sigmoid(logits.detach()/temperature)-torch.sigmoid(teach.detach()/temperature))*weights
        torch.testing.assert_close(logits.grad,expected)
        self.assertIsNone(teach.grad)
        self.assertGreater(float(loss.detach()),0.)
        self.assertEqual(float(logits.grad[1]),0.)

    def test_equal_logits_have_zero_kl_and_extreme_logits_stay_finite(self):
        a=torch.tensor([-100.,-2.,0.,5.,100.],requires_grad=True)
        equal=bernoulli_kl_sum(a,a.detach(),torch.ones(5),2.)
        self.assertAlmostEqual(float(equal.detach()),0.,places=5)
        loss=bernoulli_kl_sum(a,-a.detach(),torch.ones(5),2.);loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(a.grad).all())
        with self.assertRaises(ValueError):bernoulli_kl_sum(a,a.detach(),torch.ones(5),0.)
        with self.assertRaises(ValueError):bernoulli_kl_sum(a,a.detach(),-torch.ones(5),2.)

    def test_unequal_exposures_accumulate_to_same_distillation_gradient(self):
        y=torch.tensor([1.,0.,1.,0.]);hardw=torch.tensor([1.,7.,2.,3.]);softw=torch.tensor([0.,2.,1.,0.])
        t=torch.tensor([1.,-2.,.4,3.]);a=torch.tensor(.3,requires_grad=True);b=a.detach().clone().requires_grad_()
        hard=lambda z,s:(torch.nn.functional.binary_cross_entropy_with_logits(z.expand_as(y[s]),y[s],reduction='none')*hardw[s]).sum()/hardw.sum()
        full=hard(a,slice(None))+.5*bernoulli_kl_sum(a.expand(4),t,softw,2.)/softw.sum();full.backward()
        for s in (slice(0,1),slice(1,4)):
            (hard(b,s)+.5*bernoulli_kl_sum(b.expand(len(y[s])),t[s],softw[s],2.)/softw.sum()).backward()
        torch.testing.assert_close(a.grad,b.grad)

    def test_teacher_is_unpadded_four_channel_model_and_freezes_completely(self):
        model=VIIRSTeacher(base=4);model.initialize_prior(.01)
        image=torch.randn(1,4,32,48)
        out=model(image)
        self.assertEqual(out.shape,(1,1,32,48));torch.testing.assert_close(out.sigmoid(),torch.full_like(out,.01))
        torch.nn.functional.binary_cross_entropy_with_logits(out,torch.ones_like(out)).backward()
        self.assertTrue(any(p.grad is not None for p in model.parameters()))
        frozen_teacher(model)
        self.assertFalse(model.training);self.assertTrue(all(not p.requires_grad for p in model.parameters()))
        with self.assertRaises(ValueError):model(torch.randn(2,4,32,48))


class NativeTeacherCalibrationTests(unittest.TestCase):
    def attrs(self):
        return {'scale_factor':np.array([2.]),'add_offset':np.array([3.]),'valid_min':np.array([0]),
            'valid_max':np.array([65527]),'_FillValue':np.array([65535]),'units':b'Watts/meter^2/steradian/micrometer'}
    def qa(self):return {'flag_masks':np.array([2,4,256,512,1024,4096],np.uint16),
        'flag_meanings':b'Out_of_Range Saturation Bowtie_Deleted Missing_EV Cal_Fail Noisy_Detector'}

    def test_cf_scaling_adds_offset_and_all_special_values_are_masked(self):
        raw=np.array([[5,65527,65528,65532,65535]],np.uint16)
        values,valid=data.calibrate(raw,np.zeros_like(raw),self.attrs(),self.qa())
        self.assertEqual(float(values[0,0]),13.)
        np.testing.assert_array_equal(valid,[[True,True,False,False,False]])
        self.assertTrue((values[~valid]==0).all())

    def test_saturated_bowtie_missing_and_noisy_pixels_do_not_enter_radiance(self):
        raw=np.full((1,6),10,np.uint16);flags=np.array([[0,4,256,512,1024,4096]],np.uint16)
        values,valid=data.calibrate(raw,flags,self.attrs(),self.qa())
        self.assertEqual(int(valid.sum()),1);self.assertTrue((values[~valid]==0).all())
        with self.assertRaisesRegex(ValueError,'attribute'):data.calibrate(raw,flags,{},self.qa())
        with self.assertRaisesRegex(ValueError,'quality'):data.calibrate(raw,flags,self.attrs(),{})

    def test_coordinate_alignment_is_exact_not_shape_only(self):
        lat=np.array([[50.,51.]],np.float64);lon=np.array([[-120.,-119.]],np.float64)
        arrays={'query_lat':lat.copy(),'query_lon':lon.copy()};data.verify_crop(lat,lon,arrays)
        arrays['query_lat'][0,1]+=.000001
        with self.assertRaisesRegex(ValueError,'coordinates'):data.verify_crop(lat,lon,arrays)

    def test_teacher_input_schema_refuses_label_arrays(self):
        arrays={'radiance':np.ones((2,4,8),np.float32),'valid':np.ones((2,4,8),bool)}
        data.validate_teacher_arrays(arrays,(4,8))
        with self.assertRaisesRegex(ValueError,'imagery'):data.validate_teacher_arrays({**arrays,'fire_mask':np.zeros((4,8))},(4,8))


@unittest.skipUnless(HAS_TORCH,'PyTorch is not installed in the data-only environment')
class DistillationPipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(7)
        self.temp=tempfile.TemporaryDirectory(dir=ROOT/'.cache');self.patch=patch.object(core,'ROOT',Path(self.temp.name));self.patch.start()
        make_demo('parent.jsonl',size=16,query_size=16,train_tiles=2)
        parents=core.read_jsonl('parent.jsonl');pairs=[]
        for row in parents:
            with np.load(core.path(row['tile'])) as z:
                target=z['target'];weight=z['weight']
            radiance=np.stack([.3+3*target,.8+target]).astype(np.float32);valid=np.ones_like(radiance,bool)
            dest=f'teacher/{row["tile_id"]}.npz';core.write_npz(dest,radiance=radiance,valid=valid)
            pairs.append({**row,'teacher_schema':data.SCHEMA,'teacher_tile':dest,'teacher_sha256':core.sha256(dest),
                'viirs_l1b':'raw_'+row['source_ids'][0],'viirs_l1b_sha256':'fixture',
                'teacher_known_weight':float(weight.sum()),'teacher_positive_weight':float((target*weight).sum())})
        core.write_jsonl('pairs.jsonl',pairs)
        self.experiment={'name':'fixture','parent_manifest':'parent.jsonl','parent_manifest_sha256':core.sha256('parent.jsonl'),
            'student_statistics':'student_stats.json','teacher_config':'teacher_cfg.json','student_config':'student_cfg.json',
            'temperature':2.,'distillation_weight':.5}
        self.cfg={'device':'cpu','precision':'float32','seed':9,'model':{'base':4},'epochs':2,'microbatch':1,
            'accumulation_steps':2,'workers':0,'prefetch_factor':1,'pin_memory':False,'persistent_workers':False,
            'tile_cache_mib':1,'query_bucket_size':8,'learning_rate':.003,'weight_decay':0.,
            'gradient_clip':1.,'threshold':.5,'optimizer_impl':'default'}
        core.write_json('teacher_cfg.json',self.cfg)
        cfg={**self.cfg,'model':{'base':8,'branch':4},'microbatch':2,'accumulation_steps':1,
            'native_loss_weight':.2,'initialize_output_prior':False};core.write_json('student_cfg.json',cfg)
        core.statistics('parent.jsonl','student_stats.json',True)
    def tearDown(self):self.patch.stop();self.temp.cleanup()

    def statistics(self):return data.fit_statistics('pairs.jsonl',self.experiment,True)

    def test_teacher_statistics_do_not_use_validation_radiance(self):
        first=core.read_json(self.statistics())
        validation=next(r for r in core.read_jsonl('pairs.jsonl') if r['split']=='validation')
        with np.load(core.path(validation['teacher_tile'])) as z:valid=z['valid']
        core.write_npz(validation['teacher_tile'],radiance=np.full(valid.shape,9999.,np.float32),valid=valid)
        rows=core.read_jsonl('pairs.jsonl')
        for r in rows:
            if r['tile_id']==validation['tile_id']:r['teacher_sha256']=core.sha256(r['teacher_tile'])
        core.write_jsonl('changed_pairs.jsonl',rows)
        stats_config={**self.experiment,'name':'other'}
        second=core.read_json(data.fit_statistics('changed_pairs.jsonl',stats_config,True))
        self.assertEqual(first['mean'],second['mean']);self.assertEqual(first['scale'],second['scale'])
        self.assertEqual(first['fit_partition'],'train')

    def test_teacher_source_cannot_cross_a_parent_partition(self):
        rows=core.read_jsonl('pairs.jsonl');rows[-1]['viirs_l1b']=rows[0]['viirs_l1b'];core.write_jsonl('bad_pairs.jsonl',rows)
        with self.assertRaisesRegex(ValueError,'leakage'):data.validate_pairs('bad_pairs.jsonl',self.experiment,True)

    def test_teacher_quality_guard_rejects_uninformative_predictions(self):
        stats=core.read_json(self.statistics());cfg=resolve_config(self.cfg)
        saved={'stats':stats,'config':cfg};rows=core.read_jsonl('pairs.jsonl');model=VIIRSTeacher(base=4)
        score={'valid_query_weight':1000.,'positive_query_weight':10.,'average_precision':.01,'brier':.02}
        with patch.object(teacher,'score_teacher',return_value=score):
            with self.assertRaisesRegex(ValueError,'quality'):teacher.qualify_teacher(model,saved,rows,torch.device('cpu'),self.experiment)

    def test_recover_checkpoint_committed_before_epoch_log_append(self):
        saved={'epoch':0,'epoch_record':{'epoch':0,'validation':{'logloss':.1}}}
        reconcile_history('unit',saved);reconcile_history('unit',saved)
        self.assertEqual(len(core.read_jsonl('logs/unit/epochs.jsonl')),1)
        with self.assertRaisesRegex(ValueError,'history'):reconcile_history('unit',{'epoch':3,'epoch_record':{'epoch':3}})

    def fitted_cache(self):
        stats=self.statistics()
        with contextlib.redirect_stdout(io.StringIO()):
            teacher.train_teacher('pairs.jsonl',stats,'teacher_cfg.json','fixture_teacher',allow_synthetic=True)
        # This tiny fixture checks plumbing, not fire accuracy. Production quality
        # rejection is tested independently above and is never disabled by a CLI flag.
        receipt={'passed':True,'test_used':False,'fixture_only':True}
        with patch.object(teacher,'qualify_teacher',return_value=receipt),contextlib.redirect_stdout(io.StringIO()):
            cache=teacher.cache_teacher_targets('pairs.jsonl','models/fixture_teacher/best.pt',self.experiment,True)
        return cache

    def test_end_to_end_train_cache_student_export_needs_no_viirs(self):
        cache=self.fitted_cache();meta=core.read_json(cache)
        self.assertTrue(all(r['split']=='train' for r in meta['rows']))
        self.assertEqual(len(meta['rows']),2)
        before=core.sha256('models/fixture_teacher/best.pt')
        with contextlib.redirect_stdout(io.StringIO()):
            student.train_student(self.experiment,cache,'fixture_student',allow_synthetic=True)
            student.train_student(self.experiment,cache,'fixture_student',True,True)
        self.assertEqual(before,core.sha256('models/fixture_teacher/best.pt'))
        with patch.object(teacher,'load_teacher',side_effect=AssertionError('Student inference must not load teacher')):
            model,saved=engine.load_checkpoint('models/fixture_student/student_inference.pt',torch.device('cpu'))
            rows=core.read_jsonl('parent.jsonl');raw=next(iter(engine.loader([rows[-1]],saved['stats'],saved['config'])))
            with torch.inference_mode():output=model(**raw['inputs'])
        self.assertEqual(tuple(output['logits'].shape[-2:]),(16,16))
        self.assertFalse(saved['requires_viirs_at_inference']);self.assertEqual(saved['inference_input_keys'],list(core.INPUT_KEYS))
        self.assertNotIn('optimizer',saved);self.assertNotIn('teacher',model.state_dict())

    def test_foreign_or_heldout_soft_target_is_rejected(self):
        train=[r for r in core.read_jsonl('pairs.jsonl') if r['split']=='train']
        meta={'schema':'frozen_viirs_soft_targets_v1','training_only':True,'rows':[{'tile_id':train[0]['tile_id'],'split':'validation'}]}
        with self.assertRaisesRegex(ValueError,'Held-out'):teacher.validate_soft_targets(meta,train)

    def test_teacher_cache_tampering_cannot_silently_train_student(self):
        cache=self.fitted_cache();meta=core.read_json(cache)
        training=[r for r in core.read_jsonl('parent.jsonl') if r['split']=='train']
        store=student.DistilledTiles(training,core.read_json('student_stats.json'),meta)
        store[0]
        row=next(r for r in meta['rows'] if r['tile_id']==training[0]['tile_id'])
        core.path(row['path']).write_bytes(b'changed-cache')
        with self.assertRaisesRegex(ValueError,'checksum'):store[0]

    def test_student_refuses_distillation_weight_outside_known_labels(self):
        cache=self.fitted_cache();meta=core.read_json(cache)
        training=[r for r in core.read_jsonl('parent.jsonl') if r['split']=='train']
        row=next(r for r in meta['rows'] if r['tile_id']==training[0]['tile_id'])
        with np.load(core.path(row['path'])) as z:logits=z['teacher_logits'];weight=z['distill_weight']
        with np.load(core.path(training[0]['tile'])) as z:unknown=z['weight']==0
        self.assertTrue(unknown.any());weight[unknown]=1.
        core.write_npz(row['path'],teacher_logits=logits,distill_weight=weight);row['sha256']=core.sha256(row['path'])
        store=student.DistilledTiles(training,core.read_json('student_stats.json'),meta)
        with self.assertRaisesRegex(ValueError,'outside known'):store[0]

    @unittest.skipUnless(HAS_TORCH and torch.cuda.is_available(),'CUDA is unavailable')
    def test_cuda_bf16_student_update_has_no_teacher_gradients(self):
        self.statistics();rows=core.read_jsonl('pairs.jsonl');stats=core.read_json('data/manifests/fixture_teacher_stats.json')
        item=teacher.TeacherTiles([rows[0]],stats)[0]
        model=frozen_teacher(VIIRSTeacher(base=4).cuda())
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):soft=model(item['image'].cuda()).float().cpu()[0,0]
        source=engine.Tiles([rows[0]],core.read_json('student_stats.json'))[0]
        window=student.DistillCollate(1,8)([{**source,'teacher_logits':soft,'distill_weight':item['weight'][0,0]}])
        from firefullres.network import NativeFireNet
        config=resolve_config({**core.read_json('student_cfg.json'),'device':'cuda','precision':'bfloat16','cuda_prefetch':True,'pin_memory':True})
        network=NativeFireNet(base=8,branch=4).cuda();before=network.query_head[-1].weight.detach().clone()
        optimizer=torch.optim.AdamW(network.parameters(),lr=.001)
        scaler=torch.amp.GradScaler('cuda',enabled=False)
        result=student.student_window(network,window,optimizer,scaler,config,torch.device('cuda'),2.,.5)
        self.assertEqual(result['tiles'],1);self.assertTrue(np.isfinite(result['soft_sum']))
        self.assertFalse(torch.equal(before,network.query_head[-1].weight))
        self.assertTrue(all(p.grad is None for p in model.parameters()))


if __name__=='__main__':unittest.main()
