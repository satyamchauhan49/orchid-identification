"""Selected-value binary virtual evidence on the project's simplified BN.

For observed category j, assert only its detector J_j=yes. Its likelihood is
TPR_j if true F=j and FPR_j otherwise. Do not multiply K redundant one-v-rest
observations from a single argmax. This reproduces the paper's binary detector
CPT mechanism, not its learned graph or historical dataset/parameters.
"""
import numpy as np
from .core import probabilities, labels, softmax, EPS


def fit_binary_rates(q, truth, alpha=1.):
    q = probabilities(q)
    y = labels(truth, len(q), q.shape[1], missing=True)
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('Positive smoothing required')
    valid = y >= 0
    pred = q.argmax(1)
    rates = []
    for k in range(q.shape[1]):
        tp = int(((pred==k)&(y==k)&valid).sum())
        fn = int(((pred!=k)&(y==k)&valid).sum())
        tn = int(((pred!=k)&(y!=k)&valid).sum())
        fp = int(((pred==k)&(y!=k)&valid).sum())
        rates.append(dict(tp=tp,fn=fn,tn=tn,fp=fp,
            positive_support=tp+fn,negative_support=tn+fp,
            tpr=(tp+alpha)/(tp+fn+2*alpha),
            tnr=(tn+alpha)/(tn+fp+2*alpha),
            empirical_tpr=tp/(tp+fn) if tp+fn else None,
            empirical_tnr=tn/(tn+fp) if tn+fp else None,
            alpha=alpha))
    return rates


def binary_posterior(prior, theta, evidence, rates, missing_confidence=.4):
    prior = probabilities(np.asarray(prior)[None,:])[0]
    if not evidence or not 0 <= missing_confidence <= 1:
        raise ValueError('Evidence and valid masking threshold required')
    n = len(next(iter(evidence.values())))
    score = np.broadcast_to(np.log(np.maximum(prior,EPS)),(n,len(prior))).copy()
    used = {}
    for f, raw in evidence.items():
        q, th = probabilities(raw), probabilities(theta[f])
        if len(q)!=n or th.shape!=(len(prior),q.shape[1]) or len(rates[f])!=q.shape[1]:
            raise ValueError('Binary evidence shape mismatch')
        tpr = np.array([r['tpr'] for r in rates[f]])
        tnr = np.array([r['tnr'] for r in rates[f]])
        if not np.isfinite([tpr,tnr]).all() or ((tpr<0)|(tpr>1)|(tnr<0)|(tnr>1)).any():
            raise ValueError('Invalid binary rates')
        j = q.argmax(1)
        probability_true_value = th[:,j].T
        likelihood = probability_true_value*tpr[j,None] + (1-probability_true_value)*(1-tnr[j,None])
        active = q.max(1)>=missing_confidence
        score += np.where(active[:,None],np.log(np.maximum(likelihood,EPS)),0.)
        used[f]=active
    return softmax(score), used
