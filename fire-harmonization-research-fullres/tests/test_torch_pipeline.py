"""Runs on CPU after PyTorch installation; explicitly skipped when unavailable."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
HAS_TORCH=importlib.util.find_spec('torch') is not None


@unittest.skipUnless(HAS_TORCH,'PyTorch not installed on this review host')
class TorchPipelineTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(2);torch.manual_seed(3)
    def test_training_prior_initialization_is_constant_and_trainable(self):
        import torch
        from firefullres.network import NativeFireNet
        model=NativeFireNet(base=8,branch=4)
        model.initialize_output_priors(.001,.002)
        inputs={'thermal':torch.randn(1,12,32,32),'reflective_500':torch.randn(1,10,64,64),
                'reflective_250':torch.randn(1,4,128,128),'geometry':torch.randn(1,14,32,32),
                'query_grid':torch.zeros(1,16,16,2)}
        output=model(**inputs)
        torch.testing.assert_close(output['logits'].sigmoid(),torch.full_like(output['logits'],.001))
        torch.testing.assert_close(output['native_logits'].sigmoid(),torch.full_like(output['native_logits'],.002))
        torch.nn.functional.binary_cross_entropy_with_logits(output['logits'],torch.ones_like(output['logits'])).backward()
        self.assertNotEqual(float(model.query_head[-1].bias.grad.abs().sum()),0.)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        with self.assertRaises(ValueError):model.initialize_output_priors(0.,.1)
    def test_model_native_shapes_masked_loss_and_backward(self):
        import torch
        from firefullres.network import NativeFireNet
        from firefullres.engine import loss_sums
        model=NativeFireNet(base=8,branch=4)
        inputs={'thermal':torch.randn(1,12,32,32),'reflective_500':torch.randn(1,10,64,64),
                'reflective_250':torch.randn(1,4,128,128),'geometry':torch.randn(1,14,32,32),
                'query_grid':torch.zeros(1,48,64,2,dtype=torch.float32)}
        out=model(**inputs)
        self.assertEqual(out['logits'].shape,(1,1,48,64));self.assertEqual(out['native_logits'].shape,(1,1,32,32))
        batch={'target':torch.zeros_like(out['logits']),'weight':torch.ones_like(out['logits']),
               'native_target':torch.zeros_like(out['native_logits']),'native_weight':torch.ones_like(out['native_logits'])}
        a,b=loss_sums(out,batch);loss=a/batch['weight'].sum()+.2*b/batch['native_weight'].sum();loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
    def test_accumulation_uses_total_exposure_not_mean_of_means(self):
        import torch
        from torch.nn import functional as F
        initial=torch.tensor(.2)
        whole=initial.clone().requires_grad_()
        y=torch.tensor([0.,1.,1.]);weight=torch.tensor([1.,2.,7.])
        (F.binary_cross_entropy_with_logits(whole.expand(3),y,reduction='none')*weight).sum().div(weight.sum()).backward()
        accumulated=initial.clone().requires_grad_()
        for sl in (slice(0,1),slice(1,3)):
            (F.binary_cross_entropy_with_logits(accumulated.expand(len(y[sl])),y[sl],reduction='none')*weight[sl]).sum().div(weight.sum()).backward()
        torch.testing.assert_close(whole.grad,accumulated.grad)
    def test_query_padding_does_not_change_unpadded_predictions(self):
        import torch
        from firefullres.network import NativeFireNet
        model=NativeFireNet(base=8,branch=4).eval()
        source={'thermal':torch.randn(1,12,32,32),'reflective_500':torch.randn(1,10,64,64),
                'reflective_250':torch.randn(1,4,128,128),'geometry':torch.randn(1,14,32,32)}
        small=torch.zeros(1,16,16,2);large=torch.full((1,24,24,2),2.);large[:,:16,:16]=small
        with torch.no_grad():
            a=model(**source,query_grid=small)['logits'];b=model(**source,query_grid=large)['logits'][:,:,:16,:16]
        torch.testing.assert_close(a,b)

    def test_train_resume_evaluate_and_infer_roundtrip(self):
        import json
        from firefullres import core,engine
        from firefullres.demo import make_demo
        root=Path(__file__).resolve().parents[1];(root/'.cache').mkdir(exist_ok=True)
        config=json.loads((root/'configs/smoke_cpu.json').read_text());config['epochs']=1
        config.update(microbatch=2,accumulation_steps=2,query_bucket_size=32,memory_format='channels_last')
        old=core.ROOT
        try:
            with tempfile.TemporaryDirectory(dir=root/'.cache') as temp:
                core.ROOT=Path(temp)
                make_demo('manifest.jsonl')
                import numpy as np
                rows=core.read_jsonl('manifest.jsonl');last=rows[-1]
                with np.load(core.path(last['tile'])) as z:arrays={key:z[key] for key in z.files}
                for key in ('query_grid','target','weight','query_lat','query_lon','inference_valid','modis_baseline'):
                    arrays[key]=arrays[key][:57,:61]
                core.write_npz(last['tile'],**arrays);last.update(sha256=core.sha256(last['tile']),query_shape=[57,61])
                core.write_jsonl('manifest.jsonl',rows);core.statistics('manifest.jsonl','stats.json',True)
                shapes={r['tile_id']:tuple(r['query_shape']) for r in rows}
                core.write_json('config.json',config)
                engine.train('manifest.jsonl','stats.json','config.json','unit',True)
                engine.train('manifest.jsonl','stats.json','config.json','unit',True,True,epochs_override=2)
                result=engine.predict('manifest.jsonl','models/unit/best.pt','results/test','test','cpu',True,True)
                self.assertTrue(result['synthetic_smoke_only'])
                self.assertTrue(0<=result['model_metrics']['brier']<=1)
                result=engine.predict('manifest.jsonl','models/unit/best.pt','results/infer','test','cpu',True,False)
                self.assertEqual(len(result['predictions']),2)
                profile=engine.profile('manifest.jsonl','stats.json','config.json','profile.json',2,True,1)
                self.assertEqual(profile['measured_updates'],2)
                self.assertEqual(profile['processed_tiles'],8)
                self.assertGreater(profile['tiles_per_second_including_loader'],0)
                self.assertIsNone(profile['peak_allocated_bytes'])
                engine.profile('manifest.jsonl','stats.json','config.json','traced_profile.json',1,True,1,trace='trace.json')
                self.assertTrue(core.path('trace.json').is_file())
                for item in result['predictions']:
                    with np.load(core.path(item['prediction'])) as z:
                        self.assertEqual(z['probability'].shape,shapes[item['tile_id']])
        finally:core.ROOT=old

    def test_shared_training_step_matches_one_large_batch(self):
        import copy
        import torch
        from firefullres.engine import collate,WindowCollate,train_window
        from firefullres.runtime import resolve_config
        class Small(torch.nn.Module):
            def __init__(self):super().__init__();self.bias=torch.nn.Parameter(torch.tensor(.2))
            def forward(self,**inputs):
                n,_,h,w=inputs['thermal'].shape;qh,qw=inputs['query_grid'].shape[1:3]
                return {'logits':self.bias.expand(n,1,qh,qw),'native_logits':self.bias.expand(n,1,h,w)}
        def item(i):
            return {'inputs':{'thermal':torch.zeros(12,8,8),'reflective_500':torch.zeros(10,16,16),
                    'reflective_250':torch.zeros(4,32,32),'geometry':torch.zeros(14,8,8),'query_grid':torch.zeros(10+i,9,2)},
                    'target':torch.full((10+i,9),float(i%2)),'weight':torch.full((10+i,9),float(i+1)),
                    'native_target':torch.full((8,8),float(i%2)),'native_weight':torch.ones(8,8)*(i+1),'record':{}}
        samples=[item(i) for i in range(3)]
        a=Small();b=copy.deepcopy(a)
        config=resolve_config({'precision':'float32','microbatch':1,'accumulation_steps':3,'gradient_clip':10.,'native_loss_weight':.2})
        opta=torch.optim.SGD(a.parameters(),lr=.1);optb=torch.optim.SGD(b.parameters(),lr=.1)
        scaler=torch.amp.GradScaler('cuda',enabled=False)
        train_window(a,WindowCollate(1,16)(samples),opta,scaler,config,torch.device('cpu'))
        train_window(b,WindowCollate(3,16)(samples),optb,scaler,config,torch.device('cpu'))
        torch.testing.assert_close(a.bias,b.bias)

    def test_nonfinite_guard_rejects_update_before_optimizer(self):
        import torch
        from firefullres.engine import WindowCollate,train_window
        from firefullres.runtime import resolve_config
        class Bad(torch.nn.Module):
            def __init__(self):super().__init__();self.bias=torch.nn.Parameter(torch.tensor(1.))
            def forward(self,**inputs):
                logits=self.bias*torch.full((1,1,4,4),float('nan'))
                return {'logits':logits,'native_logits':logits}
        item={'inputs':{key:torch.zeros(1,4,4) for key in ('thermal','reflective_500','reflective_250','geometry')},
              'target':torch.zeros(4,4),'weight':torch.ones(4,4),'native_target':torch.zeros(4,4),
              'native_weight':torch.ones(4,4),'record':{}}
        item['inputs']['query_grid']=torch.zeros(4,4,2)
        model=Bad();optimizer=torch.optim.SGD(model.parameters(),lr=.1)
        config=resolve_config({'precision':'float32','gradient_clip':1.,'native_loss_weight':.2})
        with self.assertRaisesRegex(ValueError,'refused'):
            train_window(model,WindowCollate(1,0)([item]),optimizer,torch.amp.GradScaler('cuda',enabled=False),config,torch.device('cpu'))
        self.assertEqual(model.bias.item(),1.)

    def test_persistent_workers_keep_epoch_coverage_and_resume_order(self):
        import json
        from firefullres import core,engine
        from firefullres.demo import make_demo
        root=Path(__file__).resolve().parents[1];old=core.ROOT;(root/'.cache').mkdir(exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(dir=root/'.cache') as temp:
                core.ROOT=Path(temp);make_demo('manifest.jsonl')
                stats=core.statistics('manifest.jsonl','stats.json',True)
                rows=core.read_jsonl('manifest.jsonl')
                for row in rows:row['tile']=str(core.path(row['tile']))
                config=json.loads((root/'configs/smoke_cpu.json').read_text())
                config.update(workers=1,persistent_workers=True,microbatch=1,accumulation_steps=2,tile_cache_mib=2)
                data=engine.loader(rows,stats,config,True,accumulated=True)
                def ids():return [r['tile_id'] for window in data for batch in window['batches'] for r in batch['record']]
                first=ids();data.batch_sampler.set_epoch(1);second=ids()
                self.assertEqual(sorted(first),sorted(r['tile_id'] for r in rows))
                self.assertEqual(sorted(second),sorted(first));self.assertNotEqual(first,second)
                del data
        finally:core.ROOT=old

    @unittest.skipUnless(HAS_TORCH and __import__('torch').cuda.is_available(),'CUDA GPU unavailable')
    def test_cuda_prefetch_fused_step_matches_synchronous_step(self):
        import torch
        from firefullres.runtime import move,device_batches,resolve_config,optimizer_for
        config=resolve_config({'precision':'float32','pin_memory':True,'cuda_prefetch':True,
                               'memory_format':'channels_last','optimizer_impl':'fused','learning_rate':.01,'weight_decay':.01})
        raw={'inputs':{'thermal':torch.randn(1,12,8,8).pin_memory(),'query_grid':torch.zeros(1,8,8,2).pin_memory()},
             **{k:torch.zeros(1,1,8,8).pin_memory() for k in ('target','weight','native_target','native_weight')}}
        device=torch.device('cuda');a=move(raw,device,config)
        for b in device_batches([raw,raw,raw],device,config):
            torch.testing.assert_close(a['inputs']['thermal'],b['inputs']['thermal'])
            self.assertTrue(b['inputs']['thermal'].is_contiguous(memory_format=torch.channels_last))
            self.assertEqual(b['inputs']['query_grid'].dtype,torch.float32)
        first=torch.nn.Linear(2,1,device=device);second=torch.nn.Linear(2,1,device=device)
        second.load_state_dict(first.state_dict())
        fused=optimizer_for(first,config,device)
        plain=torch.optim.AdamW(second.parameters(),lr=.01,weight_decay=.01,foreach=False)
        x=torch.randn(3,2,device=device)
        for model,opt in [(first,fused),(second,plain)]:
            model(x).square().mean().backward();opt.step()
        for a,b in zip(first.parameters(),second.parameters()):torch.testing.assert_close(a,b)

    def test_compile_wrapper_leaves_canonical_checkpoint_keys(self):
        import torch
        from unittest.mock import patch
        from firefullres.runtime import configure_model,resolve_config
        original=torch.compile
        config=resolve_config({'precision':'float32','compile_mode':'default'})
        model=torch.nn.Linear(2,1)
        with patch('firefullres.runtime.torch.compile',side_effect=lambda m,**kw:original(m,backend='eager',dynamic=kw['dynamic'])):
            execution=configure_model(model,config,torch.device('cpu'))
            execution(torch.ones(1,2)).sum().backward()
        self.assertEqual(set(model.state_dict()),{'weight','bias'})
        self.assertIsNotNone(model.weight.grad)

    def test_channels_last_preserves_model_outputs_and_query_layout(self):
        import copy
        import torch
        from firefullres.network import NativeFireNet
        from firefullres.runtime import configure_model,move,resolve_config
        original=NativeFireNet(base=8,branch=4).eval();converted=copy.deepcopy(original)
        config=resolve_config({'precision':'float32','memory_format':'channels_last'})
        execution=configure_model(converted,config,torch.device('cpu'))
        raw={'inputs':{'thermal':torch.randn(2,12,32,32),'reflective_500':torch.randn(2,10,64,64),
                      'reflective_250':torch.randn(2,4,128,128),'geometry':torch.randn(2,14,32,32),
                      'query_grid':torch.rand(2,16,24,2)*1.8-.9}}
        moved=move(raw,torch.device('cpu'),config,labels=False)
        self.assertEqual(raw['inputs']['query_grid'].stride(),moved['inputs']['query_grid'].stride())
        with torch.no_grad():a=original(**raw['inputs']);b=execution(**moved['inputs'])
        for key in a:torch.testing.assert_close(a[key],b[key],rtol=1e-4,atol=1e-5)

    @unittest.skipUnless(HAS_TORCH and __import__('torch').cuda.is_available(),'CUDA GPU unavailable')
    def test_cuda_bf16_native_step_with_prefetch_and_fused_optimizer(self):
        import torch
        from firefullres.engine import WindowCollate,train_window
        from firefullres.network import NativeFireNet
        from firefullres.runtime import configure_model,optimizer_for,resolve_config,LABEL_KEYS
        if not torch.cuda.is_bf16_supported():self.skipTest('BF16 GPU unavailable')
        config=resolve_config({'precision':'bfloat16','pin_memory':True,'cuda_prefetch':True,
                              'memory_format':'channels_last','optimizer_impl':'fused','learning_rate':.001,
                              'weight_decay':.01,'gradient_clip':1.,'native_loss_weight':.2})
        samples=[]
        for i in range(4):
            samples.append({'inputs':{'thermal':torch.randn(12,32,32),'reflective_500':torch.randn(10,64,64),
                'reflective_250':torch.randn(4,128,128),'geometry':torch.randn(14,32,32),
                'query_grid':torch.rand(16,24,2)*1.8-.9},'target':torch.full((16,24),float(i%2)),
                'weight':torch.ones(16,24),'native_target':torch.full((32,32),float(i%2)),
                'native_weight':torch.ones(32,32),'record':{}})
        window=WindowCollate(2,32)(samples)
        for b in window['batches']:
            b['inputs']={key:value.pin_memory() for key,value in b['inputs'].items()}
            for key in LABEL_KEYS:b[key]=b[key].pin_memory()
        device=torch.device('cuda');model=NativeFireNet(base=8,branch=4).to(device)
        execution=configure_model(model,config,device);optimizer=optimizer_for(model,config,device)
        result=train_window(execution,window,optimizer,torch.amp.GradScaler('cuda',enabled=False),config,device)
        self.assertEqual(result,4)
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
        self.assertTrue(optimizer.state)


if __name__=='__main__':unittest.main()
