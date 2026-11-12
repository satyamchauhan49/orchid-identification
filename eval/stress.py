"""Synthetic-only stress scenarios; frozen policy, no test data or retraining."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from bn_resolver.contracts import FEATURES, load_maps, read_manifest, load_bundle, truth, digest
from bn_resolver.core import fit_temperature, empirical_theta, resolve
from bn_resolver.pipeline import predict, write_json
from bn_resolver.audit import audit
from .experiments import is_synthetic, save_table
from .metrics import classification


def scenarios(arrays, seed=42):
    rng=np.random.default_rng(seed)
    result={'baseline':copy.deepcopy(arrays)}
    a=copy.deepcopy(arrays)
    for f in FEATURES: a[f][:]=0
    result['uninformative_features']=a
    a=copy.deepcopy(arrays)
    for f in ('CF','CL'): a[f]=rng.normal(0,5,a[f].shape)
    result['weak_colour_heads']=a
    a=copy.deepcopy(arrays)
    a['species']=np.roll(a['species'],1,axis=1)*4
    result['confident_conflicting_dnn']=a
    a=copy.deepcopy(arrays)
    for f in FEATURES: a[f]=np.roll(a[f],1,axis=1)*3
    result['confident_wrong_features']=a
    a={f:np.zeros_like(x) for f,x in arrays.items()}
    result['all_uniform']=a
    return result


def run(args):
    out=Path(args.output)
    if out.exists(): raise ValueError('Use a new output directory')
    maps=load_maps(args.maps)
    if not is_synthetic(maps): raise ValueError('Stress suite requires synthetic species names')
    rows=read_manifest(args.manifest,maps)
    ids,arrays=load_bundle(args.val,rows,maps,'val',digest(args.maps))
    artifact=json.loads(Path(args.artifact).read_text())
    if artifact['label_maps_sha256']!=digest(args.maps) or artifact['manifest_sha256']!=digest(args.manifest):
        raise ValueError('Frozen map/manifest hash mismatch')
    y=truth(rows,ids,maps,'species')
    if (y<0).any(): raise ValueError('Complete synthetic validation truth required')
    policy=artifact['policy']
    metrics=[]
    for name,values in scenarios(arrays).items():
        d,b,used=predict(values,artifact)
        fused,pred,accepted,stages=resolve(d,b,policy['weight'],policy['threshold'])
        metrics.append(dict(scenario=name,data_kind='synthetic_stress',
            **classification(fused,y,accepted),
            mean_features_used=float(np.column_stack(list(used.values())).sum(1).mean()),
            dnn_bn_disagreement=float((d.argmax(1)!=b.argmax(1)).mean())))
    # Deliberately remove TRAIN supervision for one species; smoothing must give
    # uniform rows, not invented botanical facts or NaNs.
    train=[r for r in rows if r['split']=='train']
    ti=[r['image_id'] for r in train]
    sy=truth(rows,ti,maps,'species')
    theta_checks={}
    for f in FEATURES:
        fy=truth(rows,ti,maps,f)
        fy[sy==0]=-1
        th,support=empirical_theta(sy,fy,len(maps['species']),len(maps['features'][f]))
        theta_checks[f]=dict(support=float(support[0]),
            uniform=bool(np.allclose(th[0],1/th.shape[1])))
    low_support=fit_temperature(arrays['species'][:5],y[:5])
    out.mkdir(parents=True)
    save_table(out/'stress_metrics.csv',metrics)
    write_json(out/'fallback_checks.json',dict(data_kind='synthetic_stress',
        rare_species_without_annotations=theta_checks,low_support_temperature=low_support,
        passed=all(r['uniform'] and r['support']==0 for r in theta_checks.values()) and
               low_support['temperature']==1 and not low_support['fitted']))
    write_json(out/'input_audit.json',audit(args.maps,args.manifest,args.calib,args.val))
    write_json(out/'run_manifest.json',dict(data_kind='synthetic_stress',seed=42,
        artifact_sha256=digest(args.artifact),val_sha256=digest(args.val),
        frozen_policy=policy,test_accessed=False,
        caveat='Synthetic perturbations test behavior, not real-world robustness or OOD detection. Uninformative is not always masked: max 1/K can exceed 0.4.'))
    (out/'REPORT.md').write_text('''# Synthetic stress checks - not research results

Six frozen-policy scenarios completed: baseline, uninformative features, noisy colour
heads, confidently conflicting DNN, confidently wrong features, and all-uniform logits.
These perturbations deliberately preserve shapes and class maps. They are not
measured robustness results or unknown-species detection.

See stress_metrics.csv for full/accepted accuracy, coverage, disagreement and feature
usage. A confidently wrong model may still be accepted: confidence is not a safety
guarantee. No fitting or thresholds were changed to improve scenario outcomes.

fallback_checks.json checks uniform unsupported TRAIN feature distributions and
T=1 for insufficient calibration support. input_audit.json identifies sparse species,
missing feature labels and constant logits. Real checkpoint/preprocessing identity
still requires team verification. No test bundle was opened.

Important: uniform 2-state probabilities have maximum 0.5 and therefore remain used
under a 0.4 confidence mask. This suite reports this behavior rather than claiming
all uniform evidence is missing. A different missingness rule requires a separate
development ablation and a newly frozen policy.
''')
    print(out)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('maps','manifest','calib','val','artifact','output'): p.add_argument('--'+key,required=True)
    run(p.parse_args())


if __name__=='__main__': main()
