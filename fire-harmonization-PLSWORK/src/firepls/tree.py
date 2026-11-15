"""LightGBM contextual baseline with native validation early stopping."""
import json
import numpy as np
from .common import path,read,write,sha,config
from .prepare import load_split


def predict(split):
    import lightgbm as lgb
    model=lgb.Booster(model_file=str(path('models/tree/model.txt')))
    raw=load_split(split)
    z=model.predict(raw['X'],raw_score=True,num_threads=8)
    return np.asarray(z,np.float64)


def train():
    import lightgbm as lgb
    receipt=sha('data/features/receipt.json');done='models/tree/completed.json'
    if path(done).exists():
        saved=read(done)
        if saved['receipt_sha256']!=receipt or saved['model_sha256']!=sha('models/tree/model.txt'):raise ValueError('Completed tree artifacts changed')
        return saved
    cfg=config()['lightgbm'].copy();rounds=cfg.pop('num_boost_round');patience=cfg.pop('stopping_rounds')
    train_data=load_split('train');val=load_split('validation');names=read('data/features/statistics.json')['feature_names']
    a=lgb.Dataset(train_data['X'],label=train_data['y'],weight=train_data['w'],feature_name=names,free_raw_data=True)
    b=lgb.Dataset(val['X'],label=val['y'],weight=val['w'],reference=a,feature_name=names,free_raw_data=True)
    history={}
    model=lgb.train(cfg,a,num_boost_round=rounds,valid_sets=[b],valid_names=['validation'],callbacks=[
        lgb.early_stopping(patience,first_metric_only=True),lgb.record_evaluation(history),lgb.log_evaluation(10)])
    dest=path('models/tree/model.txt');dest.parent.mkdir(parents=True,exist_ok=True)
    temp=dest.with_name(dest.name+'.partial');model.save_model(str(temp),num_iteration=model.best_iteration);temp.replace(dest)
    loaded=lgb.Booster(model_file=str(dest));np.testing.assert_allclose(model.predict(val['X'][:1000],num_iteration=model.best_iteration),loaded.predict(val['X'][:1000]),rtol=1e-10)
    write('results/tree_history.json',history)
    result={'selected_iteration':model.best_iteration,'trained_iterations':len(history['validation']['average_precision']),
        'best_validation_average_precision':model.best_score['validation']['average_precision'],
        'receipt_sha256':receipt,'model_sha256':sha('models/tree/model.txt'),'native_reload_passed':True,'lightgbm':lgb.__version__}
    write(done,result);return result
