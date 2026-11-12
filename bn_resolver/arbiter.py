"""Optional plan extension: NumPy logistic arbiter, development-only diagnostic.

Chooses between DNN and BN candidates; it supplies no calibrated acceptance rule.
Only disagreements where exactly one candidate is correct supervise fitting.
"""
import numpy as np
from .core import probabilities


def features(dnn,bn,evidence,prior):
    dnn,bn=probabilities(dnn),probabilities(bn)
    if dnn.shape!=bn.shape or dnn.shape[1]<2:
        raise ValueError('Need matching multiclass candidates')
    prior=np.asarray(prior,dtype=float)
    if prior.shape!=(dnn.shape[1],) or (prior<=0).any() or not np.isfinite(prior).all():
        raise ValueError('Positive prior required')
    ds=np.sort(dnn,axis=1); bs=np.sort(bn,axis=1)
    d,b=dnn.argmax(1),bn.argmax(1)
    row=np.arange(len(d))
    # Log evidence ratio removes the species prior from BN posterior odds.
    gap=(np.log(np.maximum(bn[row,b],1e-12))-np.log(prior[b])-
         np.log(np.maximum(bn[row,d],1e-12))+np.log(prior[d]))
    if not evidence: raise ValueError('Feature probabilities required')
    conf=[]
    for p in evidence.values():
        p=probabilities(p)
        if len(p)!=len(dnn): raise ValueError('Feature row count mismatch')
        conf.append(p.max(1))
    return np.column_stack((ds[:,-1]-ds[:,-2],bs[:,-1]-bs[:,-2],gap,np.min(conf,axis=0)))


def fit(x,y,dnn,bn):
    x=np.asarray(x,dtype=float); y=np.asarray(y)
    if x.shape!=(len(y),4) or not np.isfinite(x).all(): raise ValueError('Invalid arbiter features')
    d,b=dnn.argmax(1),bn.argmax(1)
    keep=(d!=b)&((d==y)^(b==y))
    target=(b[keep]==y[keep]).astype(float)
    if len(target)<20 or len(np.unique(target))<2:
        return dict(fitted=False,support=len(target),reason='Need 20 decisive disagreements and both target outcomes',fallback='dnn')
    mean=x[keep].mean(0); scale=np.maximum(x[keep].std(0),1e-8)
    z=np.column_stack((np.ones(len(target)),(x[keep]-mean)/scale))
    w=np.zeros(5)
    for _ in range(2000):
        p=1/(1+np.exp(-np.clip(z@w,-40,40)))
        grad=z.T@(p-target)/len(target)+.01*np.r_[0.,w[1:]]
        w-=.05*grad
    return dict(fitted=True,support=len(target),mean=mean.tolist(),scale=scale.tolist(),weights=w.tolist(),
                feature_order=['dnn_margin','bn_margin','log_evidence_gap','minimum_feature_confidence'],
                l2=.01,iterations=2000,step=.05)


def choose(x,dnn,bn,model):
    d,b=dnn.argmax(1),bn.argmax(1)
    if not model['fitted']: return d.copy()
    x=np.asarray(x,dtype=float)
    if x.shape!=(len(d),4) or not np.isfinite(x).all(): raise ValueError('Invalid arbiter features')
    z=np.column_stack((np.ones(len(d)),(x-np.asarray(model['mean']))/np.asarray(model['scale'])))
    return np.where(z@np.asarray(model['weights'])>=0,b,d)
