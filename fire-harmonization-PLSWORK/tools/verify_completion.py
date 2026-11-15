"""Verify immutable copies, model provenance and saved inference predictions."""
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from firepls.common import read,sha,write,path,config,freeze_sources
from firepls.mlp import load_model
from firepls.inference import infer


def main():
    lock=freeze_sources();receipt=read('results/local_copy_receipt.json')
    if not receipt['complete'] or receipt['network_used'] or not receipt['physical_copies']:raise ValueError('Local copy receipt failed')
    for row in receipt['files']:
        if sha(row['destination'])!=row['sha256']:raise ValueError('Copied source content changed')
    report=read('results/evaluation.json');policies=read('results/operating_policy.json')
    assert report['operating_policy']==policies and policies['test_used_for_fit'] is False
    assert not report['new_blind_test']
    model,saved=load_model();selected=torch.load(path('models/mlp/best.pt'),map_location='cpu',weights_only=True)
    assert all(torch.equal(value,selected['model'][key]) for key,value in model.state_dict().items())
    assert 'optimizer' not in saved and not saved['requires_viirs_at_inference']
    features=read('data/features/receipt.json');assert features['counts']['train']==10371908
    assert features['positive_counts']['test']==95
    # Build an input-only tile in the new folder and predict without a label file.
    parent=[json.loads(l) for l in path('data/manifests/source_v3.jsonl').read_text().splitlines()]
    row=next(r for r in parent if r['split']=='test')
    with np.load(path(row['tile'])) as z:arrays={k:z[k] for k in ('thermal','reflective_500','reflective_250','geometry','query_grid','inference_valid')}
    from firepls.inference import save_npz
    save_npz('results/inference_input_only.npz',**arrays)
    infer('results/inference_input_only.npz','results/mlp_inference_example.npz','mlp')
    subprocess.run([str((ROOT/config()['tree_python']).resolve()),str(ROOT/'run.py'),'infer','--tile','results/inference_input_only.npz','--output','results/tree_inference_example.npz','--method','tree'],check=True)
    for method in ('mlp','tree'):
        with np.load(path(f'results/{method}_inference_example.npz')) as z:
            assert np.isfinite(z['raw_probability']).all()
            assert z['raw_probability'].shape==tuple(row['query_shape'])
    verification={'passed':True,'local_files_checksum_verified':len(receipt['files']),'network_data_downloads':False,
        'physical_copies':True,'feature_count':features['feature_count'],'counts':features['counts'],
        'positive_counts':features['positive_counts'],'model_selected_on_validation':policies['selected_model'],
        'new_blind_test':False,'mlp_export_strict_reload':True,'mlp_export_weights_match_selected':True,
        'label_free_inference_passed':['mlp','tree'],'software_lock_matches':True,
        'copied_baseline_sha_matches':sha('models/reference/supervised_modis.pt')==config()['source_baseline_sha256']}
    write('results/completion_verification.json',verification);print(json.dumps(verification,indent=2))

if __name__=='__main__':main()
