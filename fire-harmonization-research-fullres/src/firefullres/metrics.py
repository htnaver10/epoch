import numpy as np


class BinaryMetrics:
    """Exact exposure-weighted point losses; histogram ranking approximation."""
    def __init__(self,bins=4096,threshold=.5):
        self.bins=bins;self.threshold=threshold;self.pos=np.zeros(bins);self.neg=np.zeros(bins)
        self.n=self.brier=self.logloss=self.tp=self.fp=self.fn=0.

    def add(self,probability,target,weight):
        p,y,w=[np.asarray(a,dtype=float).ravel() for a in (probability,target,weight)]
        keep=w>0;p,y,w=p[keep],y[keep],w[keep]
        if not len(p):return
        if not np.isfinite(p).all() or np.any((p<0)|(p>1)):raise ValueError('Invalid probabilities')
        self.n+=w.sum();self.brier+=np.sum(w*(p-y)**2)
        q=np.clip(p,1e-12,1-1e-12);self.logloss-=np.sum(w*(y*np.log(q)+(1-y)*np.log1p(-q)))
        pred=p>=self.threshold
        self.tp+=np.sum(w*y*pred);self.fp+=np.sum(w*(1-y)*pred);self.fn+=np.sum(w*y*(~pred))
        level=np.minimum((p*(self.bins-1)).astype(int),self.bins-1)
        self.pos+=np.bincount(level,weights=w*y,minlength=self.bins)
        self.neg+=np.bincount(level,weights=w*(1-y),minlength=self.bins)

    def result(self):
        if self.n==0:raise ValueError('No valid reference exposure to evaluate')
        precision=self.tp/(self.tp+self.fp) if self.tp+self.fp else 0.
        recall=self.tp/(self.tp+self.fn) if self.tp+self.fn else 0.
        pos,neg=self.pos,self.neg
        auc=float(np.sum(pos*(np.cumsum(neg)-.5*neg))/(pos.sum()*neg.sum())) if pos.sum()*neg.sum() else None
        total=np.cumsum((pos+neg)[::-1]);tp=np.cumsum(pos[::-1])
        ap=float(np.sum(pos[::-1]/pos.sum()*np.divide(tp,total,out=np.zeros_like(tp),where=total>0))) if pos.sum() else None
        return {'valid_query_weight':float(self.n),'positive_query_weight':float(pos.sum()),'brier':float(self.brier/self.n),
                'logloss':float(self.logloss/self.n),'precision':float(precision),'recall':float(recall),
                'f1':float(2*precision*recall/(precision+recall)) if precision+recall else 0.,
                'roc_auc':auc,'average_precision':ap,'ranking_method':f'approximate_{self.bins}_bin_histogram',
                'threshold':self.threshold,'scope':'recorded VIIRS query fire status; not verified ground-truth wildfire'}
