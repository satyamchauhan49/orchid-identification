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


def grouped_bootstrap(y, baseline, final, groups, accepted=None,
                      resamples=1000, seed=42):
    """Paired cluster percentile CIs, sampling whole clusters with replacement.

    Statistic is image-weighted accuracy (ratio of sums), not mean cluster accuracy.
    Policy and predictions remain frozen; intervals condition on model fitting.
    """
    y, baseline, final = map(np.asarray,(y,baseline,final))
    g = np.asarray(groups)
    if (y.ndim!=1 or not len(y) or baseline.shape!=y.shape or final.shape!=y.shape
            or g.shape!=y.shape or g.dtype.kind not in 'US'
            or any(not str(v).strip() for v in g)):
        raise ValueError('Require aligned nonempty vectors and nonempty string groups')
    if resamples<1 or int(resamples)!=resamples:
        raise ValueError('Positive integer resamples required')
    a = np.ones(len(y),dtype=bool) if accepted is None else np.asarray(accepted,dtype=bool)
    if a.shape!=y.shape:
        raise ValueError('Acceptance mask length mismatch')
    unique, inverse = np.unique(g,return_inverse=True)
    count = len(unique)
    # Aggregate once: all repeated images in a sampled cluster move together.
    sizes = np.bincount(inverse,minlength=count)
    correct_base = np.bincount(inverse,weights=baseline==y,minlength=count)
    correct_final = np.bincount(inverse,weights=final==y,minlength=count)
    accepted_n = np.bincount(inverse,weights=a,minlength=count)
    accepted_correct = np.bincount(inverse,weights=a&(final==y),minlength=count)
    samples = {k:[] for k in ('baseline_accuracy','final_accuracy','accuracy_delta',
                              'coverage','accepted_accuracy')}
    rng = np.random.default_rng(seed)
    # One cluster cannot estimate between-cluster variation.
    if count>=2:
        for _ in range(resamples):
            ix = rng.integers(0,count,count)
            n = sizes[ix].sum()
            b, f = correct_base[ix].sum()/n, correct_final[ix].sum()/n
            an = accepted_n[ix].sum()
            for k,v in [('baseline_accuracy',b),('final_accuracy',f),
                        ('accuracy_delta',f-b),('coverage',an/n)]:
                samples[k].append(float(v))
            if an: samples['accepted_accuracy'].append(float(accepted_correct[ix].sum()/an))
    def interval(values):
        return dict(low=float(np.quantile(values,.025)) if values else None,
                    high=float(np.quantile(values,.975)) if values else None,
                    valid_resamples=len(values))
    return dict(method='paired whole-cluster percentile bootstrap; image-weighted ratios',
        group_count=count,image_count=len(y),resamples=resamples,seed=seed,
        estimable=count>=2,
        intervals={k:interval(v) for k,v in samples.items()},
        warning='Assumes independent clusters; fewer clusters give unstable intervals. Predictions and policy are frozen. No selection or training uncertainty included.')
