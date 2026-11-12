"""Numerical routines; no GPU, scipy, sklearn or test labels required."""
import math
import numpy as np

EPS = 1e-12


def softmax(logits):
    x = np.asarray(logits, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Expected a finite N x K matrix")
    z = np.exp(x - x.max(axis=1, keepdims=True))
    return z / z.sum(axis=1, keepdims=True)


def probabilities(p):
    p = np.asarray(p, dtype=float)
    if p.ndim != 2 or not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("Invalid probability matrix")
    if not np.allclose(p.sum(axis=1), 1):
        raise ValueError("Probability rows must sum to one")
    return p


def labels(y, n, k, missing=False):
    a = np.asarray(y)
    if a.shape != (n,) or a.dtype.kind not in 'iu':
        raise ValueError("Labels must be a length-N integer vector")
    if ((a < (-1 if missing else 0)) | (a >= k)).any():
        raise ValueError("Label outside class map")
    return a


def nll(p, y):
    p = probabilities(p)
    y = labels(y, len(p), p.shape[1])
    if not len(y):
        raise ValueError("Cannot score an empty sample")
    return float(-np.log(np.maximum(p[np.arange(len(y)), y], EPS)).mean())


def reliability(p, y, bins=15):
    p = probabilities(p)
    y = labels(y, len(p), p.shape[1])
    conf, correct = p.max(1), p.argmax(1) == y
    out = []
    for i in range(bins):
        mask = (conf >= i / bins) & ((conf < (i+1)/bins) if i < bins-1 else (conf <= 1))
        count = int(mask.sum())
        out.append(dict(lower=i/bins, upper=(i+1)/bins, count=count,
                        confidence=float(conf[mask].mean()) if count else None,
                        accuracy=float(correct[mask].mean()) if count else None))
    return out


def ece(p, y):
    rows = reliability(p, y)
    return float(sum(r['count'] * abs(r['accuracy']-r['confidence'])
                     for r in rows if r['count']) / len(y))


def fit_temperature(logits, y, minimum=20):
    x = np.asarray(logits, dtype=float)
    softmax(x)  # validate even if support is inadequate
    y = labels(y, len(x), x.shape[1], missing=True)
    keep = y >= 0
    x, y = x[keep], y[keep]
    if len(y) < minimum or len(np.unique(y)) < 2:
        return dict(temperature=1., support=len(y), fitted=False,
                    reason='insufficient labelled calibration support')
    # Bounded golden-section search in log T; retain identity if it is better.
    lo, hi = -4., 4.
    ratio = (math.sqrt(5)-1)/2
    def objective(t):
        return nll(softmax(x / math.exp(t)), y)
    a, b = hi-ratio*(hi-lo), lo+ratio*(hi-lo)
    fa, fb = objective(a), objective(b)
    for _ in range(80):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi-ratio*(hi-lo)
            fa = objective(a)
        else:
            lo, a, fa = a, b, fb
            b = lo+ratio*(hi-lo)
            fb = objective(b)
    best = min([0., lo, hi, (lo+hi)/2], key=objective)
    t = math.exp(best)
    return dict(temperature=t, support=len(y), fitted=True,
                nll_before=nll(softmax(x), y), nll_after=nll(softmax(x/t), y),
                ece_before=ece(softmax(x), y), ece_after=ece(softmax(x/t), y))


def empirical_theta(species, feature, classes, states, alpha=1.):
    """TRAIN labels only. Missing species-feature rows become uniform."""
    s = labels(species, len(species), classes, missing=True)
    f = labels(feature, len(s), states, missing=True)
    count = np.zeros((classes, states))
    keep = (s >= 0) & (f >= 0)
    np.add.at(count, (s[keep], f[keep]), 1)
    theta = count + alpha
    return theta / theta.sum(1, keepdims=True), count.sum(1)


def confusion_cpt(q, truth, alpha=1.):
    """M[predicted, true], normalized down columns; cal_model only."""
    q = probabilities(q)
    truth = labels(truth, len(q), q.shape[1], missing=True)
    keep = truth >= 0
    count = np.zeros((q.shape[1], q.shape[1]))
    np.add.at(count, (q.argmax(1)[keep], truth[keep]), 1)
    matrix = count + alpha
    return matrix / matrix.sum(0, keepdims=True), count.sum(0)


def bn_posterior(prior, theta, evidence, matrices=None, feature_priors=None,
                 variant='direct_soft', missing_confidence=0.4):
    """V1 hard_direct, V2 hard_confusion, V3 direct_soft, V4 virtual_soft.

    V3 and V4 use train-prior correction and are likelihood approximations.
    Masking a feature makes its contribution one for every species.
    """
    if variant not in {'hard_direct','hard_confusion','direct_soft','virtual_soft'}:
        raise ValueError("Unknown evidence variant")
    prior = probabilities(np.asarray(prior)[None, :])[0]
    if not evidence:
        raise ValueError("Supply at least one feature matrix to establish row count")
    n = len(next(iter(evidence.values())))
    log_score = np.broadcast_to(np.log(np.maximum(prior, EPS)), (n, len(prior))).copy()
    used = {}
    for name, raw in evidence.items():
        q = probabilities(raw)
        th = probabilities(theta[name])
        if len(q) != n or th.shape != (len(prior), q.shape[1]):
            raise ValueError("Feature/species shape mismatch")
        active = q.max(1) >= missing_confidence
        if variant == 'hard_direct':
            likelihood = th[:, q.argmax(1)].T
        elif variant == 'direct_soft':
            pi = probabilities(np.asarray(feature_priors[name])[None, :])[0]
            likelihood = (q / np.maximum(pi, EPS)) @ th.T
        else:
            m = np.asarray(matrices[name], dtype=float)
            if m.shape != (q.shape[1], q.shape[1]) or (m < 0).any() or not np.isfinite(m).all() or not np.allclose(m.sum(0), 1):
                raise ValueError("Confusion CPT must be column normalized")
            if variant == 'hard_confusion':
                likelihood = m[q.argmax(1)] @ th.T
            else:
                pi = probabilities(np.asarray(feature_priors[name])[None, :])[0]
                r = m @ pi
                likelihood = (q / np.maximum(r, EPS)) @ m @ th.T
        log_score += np.where(active[:, None], np.log(np.maximum(likelihood, EPS)), 0.)
        used[name] = active
    return softmax(log_score), used


def fuse(dnn, bn, weight):
    dnn, bn = probabilities(dnn), probabilities(bn)
    if dnn.shape != bn.shape or not 0 <= weight <= 1:
        raise ValueError("Invalid fusion shape or weight")
    return softmax(weight*np.log(np.maximum(dnn, EPS)) +
                   (1-weight)*np.log(np.maximum(bn, EPS)))


def risk_coverage(p, y):
    p = probabilities(p)
    y = labels(y, len(p), p.shape[1])
    conf, correct = p.max(1), p.argmax(1) == y
    # Group ties: a threshold must never accept only part of a confidence tie.
    order = np.argsort(-conf, kind='stable')
    c, ok = conf[order], correct[order]
    ends = np.r_[np.flatnonzero(c[:-1] != c[1:]), len(c)-1]
    errors = np.cumsum(~ok)
    return [dict(threshold=float(c[i]), accepted=int(i+1),
                 errors=int(errors[i]), coverage=float((i+1)/len(y)),
                 risk=float(errors[i]/(i+1))) for i in ends]


def fit_policy(dnn, bn, y, target=0.015, minimum=50):
    candidates = [(nll(fuse(dnn,bn,float(w)),y), -float(w), float(w))
                  for w in np.linspace(0,1,11)]
    w = min(candidates)[2]
    curve = risk_coverage(fuse(dnn,bn,w), y)
    feasible = [r for r in curve if r['accepted'] >= minimum and r['risk'] <= target]
    choice = max(feasible, key=lambda r:r['accepted']) if feasible else None
    return dict(weight=w, threshold=choice['threshold'] if choice else None,
                feasible=choice is not None, target=target, minimum_accepted=minimum,
                calibration_selection=choice,
                limitation='Empirical risk; weight and threshold reuse cal_policy; no risk guarantee'), curve


def resolve(dnn, bn, weight, threshold):
    p = fuse(dnn,bn,weight)
    proposal, confidence = p.argmax(1), p.max(1)
    accepted = np.zeros(len(p),dtype=bool) if threshold is None else confidence >= threshold
    top5 = np.argsort(-bn, axis=1, kind='stable')[:, :min(5,bn.shape[1])]
    d = dnn.argmax(1)
    trusted = (top5 == d[:,None]).any(1) & (proposal == d)
    stages = np.where(~accepted,'low_confidence',np.where(trusted,'trusted_agreement','auto_resolved'))
    return p, proposal, accepted, stages
