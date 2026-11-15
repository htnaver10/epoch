"""Validation-only intercept fitting and fixed false-alarm operating points."""
import numpy as np


def sigmoid(logits):return np.exp(-np.logaddexp(0.,-np.asarray(logits,dtype=np.float64)))


def fit_intercept(logits,target,weight):
    z,y,w=[np.asarray(a,dtype=np.float64).ravel() for a in (logits,target,weight)]
    if not (np.isfinite(z).all() and np.isfinite(y).all() and np.isfinite(w).all()):raise ValueError('Nonfinite calibration inputs')
    positive=float(np.sum(w*y));total=float(w.sum())
    if not 0<positive<total:raise ValueError('Calibration needs both positive and negative validation exposure')
    low,high=-float(z.max())-40.,-float(z.min())+40.
    for _ in range(36):
        middle=(low+high)/2
        if float(np.sum(w*sigmoid(z+middle)))>positive:high=middle
        else:low=middle
    return (low+high)/2


def matched_false_alarm_cutoff(logits,target,weight,baseline):
    z,y,w,b=[np.asarray(a).ravel() for a in (logits,target,weight,baseline)]
    negative=(y==0)&(w>0)
    if not negative.any():raise ValueError('Operating-point selection needs negative validation exposure')
    allowed=float(w[negative&(b>=.5)].sum())
    order=np.argsort(-z[negative],kind='stable');scores=z[negative][order].astype(np.float64)
    cumulative=np.cumsum(w[negative][order],dtype=np.float64)
    index=int(np.searchsorted(cumulative,allowed,side='right'))
    cut=float(np.nextafter(scores[index],np.inf)) if index<len(scores) else float(np.min(z)-1.)
    return cut,allowed


def exact_ranking(scores,target,weight):
    score,y,w=[np.asarray(a).ravel() for a in (scores,target,weight)]
    keep=w>0;score,y,w=score[keep],y[keep],w[keep]
    order=np.argsort(score,kind='stable');score,y,w=score[order],y[order],w[order]
    ends=np.r_[np.flatnonzero(score[1:]!=score[:-1]),len(score)-1]
    cp=np.cumsum(w*y,dtype=np.float64)[ends];cn=np.cumsum(w*(1-y),dtype=np.float64)[ends]
    pos=np.diff(np.r_[0.,cp]);neg=np.diff(np.r_[0.,cn]);p=float(cp[-1]);n=float(cn[-1])
    auc=float(np.sum(pos*(cn-.5*neg))/(p*n)) if p*n else None
    cumulative_positive=np.cumsum(pos[::-1]);cumulative_total=np.cumsum((pos+neg)[::-1])
    ap=float(np.sum(pos[::-1]/p*cumulative_positive/cumulative_total)) if p else None
    return {'roc_auc_exact':auc,'average_precision_exact':ap}


def metrics(probability,target,weight,prediction,ranking_score=None):
    p,y,w=[np.asarray(a,dtype=np.float64).ravel() for a in (probability,target,weight)]
    pred=np.asarray(prediction,dtype=bool).ravel();positive=y>.5;negative=~positive
    if not len(p) or not (np.isfinite(p).all() and np.isfinite(y).all() and np.isfinite(w).all()) or w.sum()<=0 or (w<0).any():
        raise ValueError('Metrics require finite scores/labels and positive known exposure')
    if (p<0).any() or (p>1).any():raise ValueError('Invalid probabilities')
    tp=float(w[pred&positive].sum());fp=float(w[pred&negative].sum())
    fn=float(w[~pred&positive].sum());tn=float(w[~pred&negative].sum())
    precision=tp/(tp+fp) if tp+fp else 0.;recall=tp/(tp+fn) if tp+fn else 0.
    total=float(w.sum());observed=float((w*y).sum())
    q=np.clip(p,1e-12,1-1e-12)
    result={'true_positives':tp,'false_positives':fp,'false_negatives':fn,'true_negatives':tn,
        'precision':precision,'recall':recall,'f1':2*precision*recall/max(precision+recall,1e-30),
        'false_positive_rate':fp/(fp+tn) if fp+tn else None,'brier':float(np.sum(w*(p-y)**2)/total),
        'logloss':float(-np.sum(w*(y*np.log(q)+(1-y)*np.log1p(-q)))/total),
        'observed_positive_weight':observed,'predicted_positive_weight':tp+fp,
        'score_sum_active_query_equivalents':float(np.sum(w*p)),
        'score_sum_bias_fraction':float((np.sum(w*p)-observed)/observed) if observed else None,
        'known_reference_exposure':total}
    result.update(exact_ranking(p if ranking_score is None else ranking_score,y,w));return result
