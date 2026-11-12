import math
import numpy as np
from bn_resolver.core import probabilities, labels, ece, nll


def classification(p,y,accepted=None):
    p=probabilities(p)
    y=labels(y,len(p),p.shape[1])
    pred=p.argmax(1)
    correct=pred==y
    order=np.argsort(-p,axis=1,kind='stable')
    f1=[]
    for k in range(p.shape[1]):
        tp=int(((pred==k)&(y==k)).sum())
        denom=int((pred==k).sum()+(y==k).sum())
        f1.append(2*tp/denom if denom else 0.)
    a=np.ones(len(y),dtype=bool) if accepted is None else np.asarray(accepted,dtype=bool)
    if a.shape != y.shape:
        raise ValueError('Acceptance mask length mismatch')
    return dict(n=len(y),accuracy=float(correct.mean()),
                top3=float((order[:,:min(3,p.shape[1])]==y[:,None]).any(1).mean()),
                top5=float((order[:,:min(5,p.shape[1])]==y[:,None]).any(1).mean()),
                macro_f1=float(np.mean(f1)),macro_f1_zero_division=0,
                nll=nll(p,y),ece=ece(p,y),accepted=int(a.sum()),
                coverage=float(a.mean()),deferral_rate=float(1-a.mean()),
                accepted_accuracy=float(correct[a].mean()) if a.any() else None)


def bootstrap_accuracy(y,pred,accepted=None,resamples=1000,seed=42):
    y,pred=np.asarray(y),np.asarray(pred)
    if not len(y) or y.shape!=pred.shape:
        raise ValueError('Invalid bootstrap inputs')
    a=np.ones(len(y),dtype=bool) if accepted is None else np.asarray(accepted,dtype=bool)
    rng=np.random.default_rng(seed)
    values=[]
    for _ in range(resamples):
        ix=rng.integers(0,len(y),len(y))
        keep=a[ix]
        if keep.any():
            values.append(float((y[ix][keep]==pred[ix][keep]).mean()))
    return dict(low=float(np.quantile(values,.025)) if values else None,
                high=float(np.quantile(values,.975)) if values else None,
                valid_resamples=len(values),resamples=resamples,seed=seed,
                method='IID image percentile bootstrap; requires independent image units')


def mcnemar_exact(y,baseline,final):
    a=np.asarray(baseline)==y
    b=np.asarray(final)==y
    lost=int((a&~b).sum())
    gained=int((~a&b).sum())
    n=lost+gained
    # Exact two-sided binomial test under p=0.5, evaluated in log space.
    logs=[math.lgamma(n+1)-math.lgamma(k+1)-math.lgamma(n-k+1)-n*math.log(2)
          for k in range(min(lost,gained)+1)]
    peak=max(logs)
    p=min(1.,2*math.exp(peak)*sum(math.exp(v-peak) for v in logs)) if n else 1.
    return dict(baseline_only_correct=lost,final_only_correct=gained,p_value=p,
                comparison='Full-coverage baseline versus full-coverage fusion')
