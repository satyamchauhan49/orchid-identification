"""Compare supplied backbone validation runs with common metrics and paired CIs."""
import argparse,json
from pathlib import Path
import numpy as np
from bn_resolver.contracts import load_maps,read_manifest,load_bundle,truth,digest
from bn_resolver.pipeline import predict,write_json
from .metrics import grouped_bootstrap,mcnemar_exact
from .experiments import save_table


def summarize(repo,output):
    repo=Path(repo);out=Path(output)
    if out.exists():raise ValueError('Choose a new comparison directory')
    specs=[('ResNet E1','work/e1-real-experiments-01','work/e1-audited-input','work/e1-real-metadata/label_maps.json'),
           ('ResNet smoothing','work/e2-real-v1/resnet18_e2_label_smoothing_0p1/experiments','work/e2-real-v1/resnet18_e2_label_smoothing_0p1/audited','work/e1-real-metadata/label_maps.json')]
    for n in ('convnext_tiny_lr0.0001','efficientnet_b0_lr0.0003'):
        p='work/sneha-analysis/'+n;specs.append((n,p+'/experiments',p+'/audited',p+'/audited/label_maps.json'))
    result=[];preds={};base_ids=None
    for name,exp,source,maps_path in specs:
        exp=repo/exp;source=repo/source;mp=repo/maps_path
        artifact=json.loads((exp/'frozen_direct_soft.json').read_text());maps=load_maps(mp)
        rows=read_manifest(repo/'outputs/split_manifest_grouped.csv',maps)
        ids,arrays=load_bundle(source/'val.npz',rows,maps,'val',digest(mp))
        y=truth(rows,ids,maps,'species');d,b,_=predict(arrays,artifact)
        if base_ids is None:base_ids=ids;lookup={r['image_id']:r for r in rows};groups=np.array([lookup[str(i)]['phash_cluster'] for i in ids]);base_y=y
        positions={str(i):j for j,i in enumerate(ids)};order=np.array([positions[str(i)] for i in base_ids])
        if not np.array_equal(y[order],base_y):raise ValueError('Misaligned ground truth')
        preds[name]=d.argmax(1)[order]
        import csv
        with (exp/'E5_R4_resolver.csv').open() as stream:m={r['name']:r for r in csv.DictReader(stream)}
        result.append(dict(model=name,n=len(ids),dnn_top1=float(m['dnn']['accuracy']),dnn_top3=float(m['dnn']['top3']),dnn_top5=float(m['dnn']['top5']),macro_f1_all_156=float(m['dnn']['macro_f1']),
            dnn_ece=float(m['dnn']['ece']),bn_top1=float(m['bn']['accuracy']),weight=artifact['policy']['weight'],threshold=artifact['policy']['threshold'],
            coverage=float(m['fusion_selective']['coverage']),accepted_accuracy=m['fusion_selective']['accepted_accuracy'] or None))
    conv='convnext_tiny_lr0.0001';comparisons={}
    for name in preds:
        if name==conv:continue
        comparisons[conv+' minus '+name]=dict(grouped_ci=grouped_bootstrap(base_y,preds[name],preds[conv],groups),mcnemar=mcnemar_exact(base_y,preds[name],preds[conv]),
             caveat='Exploratory development comparison after validation selection; no multiplicity adjustment. McNemar assumes image independence; grouped intervals assume independent pHash groups.')
    for name,comparison in comparisons.items():
        comparison['mcnemar']['comparison']=name+' (paired DNN predictions)'
    out.mkdir(parents=True);save_table(out/'TEAM_VALIDATION_COMPARISON.csv',result);write_json(out/'PAIRED_BACKBONE_COMPARISONS.json',comparisons)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--repo',default='.');p.add_argument('--output',required=True);a=p.parse_args();summarize(a.repo,a.output)
