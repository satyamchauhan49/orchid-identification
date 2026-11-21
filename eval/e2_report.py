"""Summarize existing E2 runs with paired baseline comparisons; no test inputs."""
import argparse,csv,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from bn_resolver.contracts import load_maps,read_manifest,load_bundle,truth,digest
from bn_resolver.pipeline import predict,evaluate,write_json
from bn_resolver.core import resolve
from .metrics import grouped_bootstrap,mcnemar_exact
from .experiments import save_table


def summarize(root,maps_path,manifest,baseline_bundle,baseline_artifact):
    root=Path(root);maps=load_maps(maps_path);rows=read_manifest(manifest,maps)
    bid,ba=load_bundle(baseline_bundle,rows,maps,'val',digest(maps_path))
    base=json.loads(Path(baseline_artifact).read_text());bd,_,_=predict(ba,base)
    baseline={str(i):int(p) for i,p in zip(bid,bd.argmax(1))}
    lookup={r['image_id']:r for r in rows};summary=[];comparisons={}
    for folder in sorted(root.glob('resnet18_e2_*')):
        exp=folder/'experiments';ap=exp/'frozen_direct_soft.json';vp=folder/'audited/val.npz'
        ids,arrays=load_bundle(vp,rows,maps,'val',digest(maps_path))
        y=truth(rows,ids,maps,'species');a=json.loads(ap.read_text());d,b,_=predict(arrays,a)
        fused,pred,accepted,stages=resolve(d,b,a['policy']['weight'],a['policy']['threshold'])
        bp=np.array([baseline[str(i)] for i in ids]);groups=np.array([lookup[str(i)]['phash_cluster'] for i in ids])
        comparison=grouped_bootstrap(y,bp,d.argmax(1),groups)
        comparisons[folder.name]=dict(dnn_vs_E1=comparison,mcnemar=mcnemar_exact(y,bp,d.argmax(1)),
            caveat='Exploratory paired validation comparison after model selection; no multiplicity correction or final test claim.')
        output=exp/'validation_evaluation.json'
        evaluate(SimpleNamespace(maps=maps_path,manifest=manifest,bundle=str(vp),artifact=str(ap),split='val',unlock_test=False,output=str(output)))
        with (exp/'E5_R4_resolver.csv').open() as stream:metrics={r['name']:r for r in csv.DictReader(stream)}
        with (exp/'E4_calibration.csv').open() as stream:cal={r['name']:r for r in csv.DictReader(stream)}
        summary.append(dict(model=folder.name,dnn_accuracy=float(metrics['dnn']['accuracy']),bn_accuracy=float(metrics['bn']['accuracy']),
            fusion_accuracy=float(metrics['fusion_only']['accuracy']),weight=a['policy']['weight'],threshold=a['policy']['threshold'],
            coverage=float(metrics['fusion_selective']['coverage']),accepted_accuracy=metrics['fusion_selective']['accepted_accuracy'],
            raw_dnn_ece=float(cal['dnn_raw']['ece']),calibrated_dnn_ece=float(cal['dnn_temperature']['ece']),
            validation_error_target_met=bool(accepted.any() and (pred[accepted]!=y[accepted]).mean()<=.015),
            delta_ci_low=comparison['intervals']['accuracy_delta']['low'],delta_ci_high=comparison['intervals']['accuracy_delta']['high']))
    save_table(root/'E2_RESOLVER_COMPARISON.csv',summary)
    write_json(root/'PAIRED_E1_COMPARISONS.json',comparisons)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','maps','manifest','baseline-bundle','baseline-artifact'):p.add_argument('--'+key,required=True)
    a=p.parse_args();summarize(a.root,a.maps,a.manifest,a.baseline_bundle,a.baseline_artifact)

if __name__=='__main__':main()
