"""RTX compatibility smoke test, not a native fire model or benchmark.

Requires an independently configured CUDA PyTorch environment. No install,
download, account action, training-data access or writes outside this folder.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def destination(relative):
    p=(ROOT/relative).resolve()
    if not p.is_relative_to(ROOT):
        raise ValueError('Preflight output must remain in fire-harmonization-research-fullres')
    p.parent.mkdir(parents=True,exist_ok=True)
    return p


def run():
    result={'kind':'tiny_synthetic_cuda_compatibility_smoke_test_not_native_model_profile',
            'python':sys.version,'os':platform.platform(),'machine':platform.machine(),
            'native_model_trained':False,'full_model_memory_measured':False}
    try:
        import torch
    except ImportError as error:
        result.update(status='pytorch_not_installed',reason=str(error))
        return result,2
    result.update(torch_version=str(torch.__version__),wheel_cuda_version=torch.version.cuda,
                  cuda_available=torch.cuda.is_available())
    if not result['cuda_available']:
        result.update(status='cuda_not_available',reason='Run on the RTX host with a verified CUDA-enabled PyTorch build and driver.')
        return result,2
    try:
        torch.cuda.set_device(0)
        prop=torch.cuda.get_device_properties(0)
        result.update(gpu_name=prop.name,device_total_bytes=prop.total_memory,
                      compute_capability=list(torch.cuda.get_device_capability(0)),
                      wheel_architectures=torch.cuda.get_arch_list(),
                      bf16_supported=torch.cuda.is_bf16_supported(),cudnn_version=torch.backends.cudnn.version())
        result['user_target_gpu_name_matches']='RTX 5070' in prop.name and 'Ti' not in prop.name and 'Laptop' not in prop.name
        result['sm120_explicit_in_wheel']='sm_120' in result['wheel_architectures']
        result['gpu_memory_free_before_smoke_bytes']=torch.cuda.mem_get_info()[0]
        tests=[]
        for mode in ('float32','bfloat16'):
            if mode=='bfloat16' and not result['bf16_supported']:
                tests.append({'mode':mode,'passed':False,'reason':'BF16 unavailable'});continue
            model=torch.nn.Sequential(torch.nn.Conv2d(6,16,3,padding=1),torch.nn.GroupNorm(4,16),
                        torch.nn.SiLU(),torch.nn.Conv2d(16,1,1)).cuda()
            optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4)
            x=torch.randn(2,6,96,96,device='cuda')
            target=(torch.rand(2,1,96,96,device='cuda')>.98).float()
            def step():
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16,enabled=mode=='bfloat16'):
                    logits=model(x)
                with torch.autocast('cuda',enabled=False):
                    loss=torch.nn.functional.binary_cross_entropy_with_logits(logits.float(),target)
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                loss.backward()
                if not all(p.grad is None or torch.isfinite(p.grad).all().item() for p in model.parameters()):
                    raise RuntimeError('Nonfinite gradients')
                optimizer.step()
                return float(loss.detach())
            step()  # initialize Adam states and CUDA/cuDNN lazy state before profiling
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            started=time.perf_counter()
            losses=[step() for _ in range(3)]
            torch.cuda.synchronize()
            tests.append({'mode':mode,'passed':True,'losses':losses,
                          'three_tiny_steps_seconds':time.perf_counter()-started,
                          'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                          'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
            del model,optimizer,x,target
            torch.cuda.empty_cache()
        with torch.autocast('cuda',enabled=False):
            features=torch.randn(1,4,8,8,device='cuda',requires_grad=True,dtype=torch.float32)
            grid=torch.empty(1,16,16,2,device='cuda',dtype=torch.float32).uniform_(-1,1)
            sampled=torch.nn.functional.grid_sample(features,grid,align_corners=False)
            sampled.square().mean().backward()
            geometry_ok=bool(torch.isfinite(features.grad).all().item())
        result.update(smoke_tests=tests,float32_grid_sample_backward_passed=geometry_ok,
                      geometry_note='CUDA grid_sample backward may be nondeterministic; use numeric tolerances and repeated-seed validation.')
        passed=all(t['passed'] for t in tests) and geometry_ok
        result['status']='compatibility_smoke_passed' if passed else 'compatibility_smoke_failed'
        result['not_proven']=['native-model fit in 12GB','native-model throughput','NASA-data reader compatibility','fire detection/localization quality']
        return result,0 if passed else 1
    except Exception as error:
        result.update(status='compatibility_smoke_failed',error_type=type(error).__name__,reason=str(error))
        return result,1


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='results/gpu_preflight.json',help='Path relative to this full-resolution folder')
    args=parser.parse_args()
    target=destination(args.output)
    value,status=run()
    target.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    print(json.dumps(value,indent=2))
    raise SystemExit(status)
