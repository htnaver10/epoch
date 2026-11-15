"""Frozen neural prior plus new, supervised logistic residual trees."""
import json
import shutil
import time
import numpy as np
from common import ROOT, HERE, path, config, read, write, sha, load, save_array
from firepls.prepare import load_split
from firefullres.calibration import exact_ranking

def prior_features(x):
    import torch
    from firepls.mlp import load_model, normalize_numpy
    torch.set_num_threads(4)
    # CPU float32 is used for both fitting caches and deployment, avoiding BF16 cutoff drift.
    model, saved = load_model('hybrid_v1/models/base_mlp.pt', device='cpu')
    out = np.empty(len(x), np.float64)
    with torch.inference_mode():
        for start in range(0,len(x),65536):
            values = normalize_numpy(x[start:start+65536,:96], saved['stats'])
            out[start:start+len(values)] = model(torch.from_numpy(values)).numpy().astype(np.float64)
    if not np.isfinite(out).all(): raise ValueError('Nonfinite neural prior')
    return out

def cache_prior(split):
    dest = 'models/base_mlp.pt'
    path('models').mkdir(parents=True, exist_ok=True)
    if not path(dest).exists(): shutil.copyfile(ROOT/config()['base_model'], path(dest))
    if sha(path(dest)) != config()['base_model_sha256']: raise ValueError('Copied neural prior changed')
    name = f'data/{split}_prior.npy'; receipt = f'data/{split}_prior.json'
    if path(receipt).exists():
        if read(receipt)['sha256'] != sha(path(name)): raise ValueError('Cached prior changed')
        return read(receipt)
    print(f'Frozen MLP float32 scores: {split}', flush=True)
    z = prior_features(load(f'data/{split}_X.npy'))
    save_array(name, z)
    result = {'split':split, 'count':len(z), 'sha256':sha(path(name)),
              'base_model_sha256':sha(path(dest)), 'precision':'CPU float32'}
    write(receipt, result); return result

def fit_residual(train_x, train_y, train_w, train_z, val_x, val_y, val_w, val_z, options, names):
    import lightgbm as lgb
    cfg = options.copy(); rounds = cfg.pop('num_boost_round'); patience = cfg.pop('stopping_rounds')
    a = lgb.Dataset(train_x, label=train_y, weight=train_w, init_score=train_z,
                    feature_name=names, free_raw_data=True)
    b = lgb.Dataset(val_x, label=val_y, weight=val_w, init_score=val_z,
                    feature_name=names, reference=a, free_raw_data=True)
    history = {}
    model = lgb.train(cfg, a, num_boost_round=rounds, valid_sets=[b], valid_names=['validation'],
                      callbacks=[lgb.early_stopping(patience, first_metric_only=True),
                                 lgb.record_evaluation(history), lgb.log_evaluation(10)])
    # Dataset init_score is not a deployable tree: explicitly add the neural score after reload.
    combined = np.asarray(val_z, np.float64) + model.predict(val_x, raw_score=True,
                            num_iteration=model.best_iteration, num_threads=cfg['num_threads'])
    ap = exact_ranking(combined, val_y, val_w)['average_precision_exact']
    np.testing.assert_allclose(ap, model.best_score['validation']['average_precision'], rtol=1e-10, atol=1e-10)
    return model, history, combined

def train():
    import lightgbm as lgb
    done = 'models/completed.json'
    if path(done).exists():
        saved = read(done)
        if saved['model_sha256'] != sha(path('models/residual.txt')): raise ValueError('Residual model changed')
        return saved
    started = time.perf_counter()
    a, b = load_split('train'), load_split('validation')
    names = read('data/receipt.json')['feature_names']
    train_x, val_x = load('data/train_X.npy'), load('data/validation_X.npy')
    print(f'Residual fit: {len(train_x):,} queries, {int(a["y"].sum()):,} positive labels, {len(names)} features', flush=True)
    model, history, z = fit_residual(train_x,a['y'],a['w'],load('data/train_prior.npy'),
                                   val_x,b['y'],b['w'],load('data/validation_prior.npy'),config()['residual'],names)
    dest = path('models/residual.txt'); tmp = dest.with_name(dest.name+'.partial')
    model.save_model(str(tmp), num_iteration=model.best_iteration); tmp.replace(dest)
    loaded = lgb.Booster(model_file=str(dest))
    check = load('data/validation_prior.npy') + loaded.predict(val_x, raw_score=True, num_threads=8)
    np.testing.assert_allclose(check,z,rtol=1e-12,atol=1e-12)
    save_array('results/validation_logits.npy',z)
    write('results/training_history.json',history)
    importance = sorted(zip(names,model.feature_importance(importance_type='gain').tolist()),key=lambda p:-p[1])
    write('results/feature_importance.json',[{'feature':k,'gain':v} for k,v in importance])
    result = {'selected_iteration':model.best_iteration,
              'trained_iterations':len(history['validation']['average_precision']),
              'best_validation_average_precision':model.best_score['validation']['average_precision'],
              'training_queries':len(train_x),'positive_training_queries':int(a['y'].sum()),
              'feature_count':len(names),'seconds':time.perf_counter()-started,
              'model_sha256':sha(dest),'base_model_sha256':sha(path('models/base_mlp.pt')),
              'feature_receipt_sha256':sha(path('data/receipt.json')),
              'combined_native_reload_passed':True,'lightgbm_version':lgb.__version__,
              'training_labels':'unmodified VIIRS reference labels','test_used_for_fit':False}
    write(done,result); print(json.dumps(result,indent=2),flush=True); return result

def predict(split):
    import lightgbm as lgb
    if split == 'test' and not path('results/operating_policy.json').exists():
        raise ValueError('Freeze validation policies before test prediction')
    model = lgb.Booster(model_file=str(path('models/residual.txt')))
    return np.asarray(load(f'data/{split}_prior.npy'),np.float64) + model.predict(
        load(f'data/{split}_X.npy'),raw_score=True,num_threads=8)

def infer_prior(tile, output):
    from preprocessing import features, INPUTS
    # Path confinement applies to inputs and outputs. No target or VIIRS array is read.
    with np.load(path(tile),allow_pickle=False) as z:
        inputs = {k:z[k] for k in INPUTS}
        modis = z['modis_baseline']
        if 'inference_valid' not in z: raise ValueError('Inference requires MODIS-only validity mask')
        valid = z['inference_valid'] > 0
    x, names, _ = features(inputs,modis)
    if names != read('data/receipt.json')['feature_names']: raise ValueError('Inference schema changed')
    save_array('data/infer_X.npy',x.reshape(-1,x.shape[-1]))
    save_array('data/infer_prior.npy',prior_features(x.reshape(-1,x.shape[-1])))
    save_array('data/infer_valid.npy',valid)
    write('data/infer_request.json',{'output':output,'shape':list(valid.shape),'tile':tile})

def infer_finish():
    import lightgbm as lgb
    from firefullres.calibration import sigmoid
    from evaluation import verify_policy
    policy = verify_policy(); request = read('data/infer_request.json')
    model = lgb.Booster(model_file=str(path('models/residual.txt')))
    z = load('data/infer_prior.npy') + model.predict(load('data/infer_X.npy'),raw_score=True,num_threads=8)
    shape = request['shape']; p = policy['query']; valid = np.asarray(load('data/infer_valid.npy'))
    dest = path(request['output']); dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('wb') as stream:
        np.savez_compressed(stream,raw_logit=z.reshape(shape),raw_probability=sigmoid(z).reshape(shape),
                            adjusted_probability=sigmoid(z+p['logit_offset']).reshape(shape),valid=valid,
                            detection=(z>=p['cutoff']).reshape(shape)&valid)
    return {'output':str(dest),'requires_viirs':False,'requires_modis_fire_mask':True}
