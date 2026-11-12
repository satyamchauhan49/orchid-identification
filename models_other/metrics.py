"""Per-head validation metrics. Do not call on the locked test split before Sync 4."""
import numpy as np


def classification_metrics(logits, targets, class_names):
    logits, targets = np.asarray(logits), np.asarray(targets)
    valid = targets != -100
    logits, targets = logits[valid], targets[valid]
    if not len(targets):
        return {"labelled_count": 0}
    classes = len(class_names)
    confusion = np.zeros((classes, classes), dtype=int)
    prediction = logits.argmax(axis=1)
    np.add.at(confusion, (targets, prediction), 1)
    tp = np.diag(confusion)
    fn = confusion.sum(axis=1) - tp
    fp = confusion.sum(axis=0) - tp
    tn = len(targets) - tp - fn - fp
    ratio = lambda n, d: np.divide(n, d, out=np.zeros_like(n, dtype=float), where=d != 0)
    f1 = ratio(2 * tp, 2 * tp + fp + fn)
    present = (confusion.sum(axis=0) + confusion.sum(axis=1)) > 0
    ranking = np.argsort(-logits, axis=1, kind="stable")
    return {"labelled_count": len(targets), "macro_f1": float(f1[present].mean()),
            "top_k": {str(k): float((ranking[:, :min(k, classes)] == targets[:, None]).any(axis=1).mean()) for k in (1, 3, 5)},
            "per_value": {name: {"support": int(confusion[i].sum()),
                                  "tpr": float(ratio(tp, tp + fn)[i]) if tp[i] + fn[i] else None,
                                  "tnr": float(ratio(tn, tn + fp)[i]) if tn[i] + fp[i] else None}
                          for i, name in enumerate(class_names)},
            "confusion_matrix": confusion.tolist()}
