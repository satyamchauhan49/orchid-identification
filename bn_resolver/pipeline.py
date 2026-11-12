"""Local CLI: fit on development data; evaluate locked test only explicitly."""
import argparse
import json
from pathlib import Path
import numpy as np
from .contracts import FEATURES, digest, load_maps, read_manifest, truth, load_bundle, calibration_partition
from .core import (softmax, fit_temperature, empirical_theta, confusion_cpt,
                   bn_posterior, fit_policy, resolve, reliability, risk_coverage)
from eval.metrics import classification, bootstrap_accuracy, mcnemar_exact


def write_json(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def predict(arrays,artifact):
    temps=artifact['temperatures']
    dnn=softmax(arrays['species']/temps['species']['temperature'])
    q={f:softmax(arrays[f]/temps[f]['temperature']) for f in FEATURES}
    bn,used=bn_posterior(artifact['prior'],artifact['theta'],q,
                        artifact['matrices'],artifact['feature_priors'],
                        artifact['variant'],artifact['missing_confidence'])
    return dnn,bn,used


def fit(args):
    maps=load_maps(args.maps)
    rows=read_manifest(args.manifest,maps)
    mh=digest(args.maps)
    ids,arrays=load_bundle(args.bundle,rows,maps,'calib',mh)
    cm=calibration_partition(rows,ids)
    if cm.sum()<20 or (~cm).sum()<args.minimum:
        raise ValueError('Insufficient cal_model/cal_policy rows; inspect grouped partition')
    sy=truth(rows,ids,maps,'species')
    if (sy<0).any():
        raise ValueError('Calibration species labels must be complete')
    train=[r for r in rows if r['split']=='train']
    train_ids=[r['image_id'] for r in train]
    train_species=truth(rows,train_ids,maps,'species')
    if not len(train):
        raise ValueError('Training labels required for empirical theta')
    artifact=dict(schema_version=1,species_count=len(maps['species']),
                  label_maps_sha256=mh,manifest_sha256=digest(args.manifest),
                  variant=args.variant,missing_confidence=.4,
                  prior=[1/len(maps['species'])]*len(maps['species']),
                  theta={},theta_support={},feature_priors={},matrices={},
                  cpt_support={},temperatures={},
                  provenance='Empirical TRAIN image labels with add-one smoothing; not expert taxonomy',
                  cal_model_ids=ids[cm].tolist(),cal_policy_ids=ids[~cm].tolist())
    artifact['temperatures']['species']=fit_temperature(arrays['species'][cm],sy[cm])
    for f in FEATURES:
        k=len(maps['features'][f])
        ty=truth(rows,train_ids,maps,f)
        cy=truth(rows,ids,maps,f)
        theta,support=empirical_theta(train_species,ty,len(maps['species']),k)
        artifact['theta'][f]=theta.tolist()
        artifact['theta_support'][f]=support.tolist()
        count=np.bincount(ty[ty>=0],minlength=k)+1
        artifact['feature_priors'][f]=(count/count.sum()).tolist()
        temperature=fit_temperature(arrays[f][cm],cy[cm])
        artifact['temperatures'][f]=temperature
        matrix,support=confusion_cpt(softmax(arrays[f][cm]/temperature['temperature']),cy[cm])
        artifact['matrices'][f]=matrix.tolist()
        artifact['cpt_support'][f]=support.tolist()
    dnn,bn,_=predict(arrays,artifact)
    policy,curve=fit_policy(dnn[~cm],bn[~cm],sy[~cm],args.risk,args.minimum)
    artifact['policy']=policy
    write_json(args.output,artifact)
    write_json(str(args.output)+'.policy_curve.json',curve)
    print(f"Fitted {len(ids)} development rows; policy feasible: {policy['feasible']}")


def evaluate(args):
    if args.split=='test' and not args.unlock_test:
        raise ValueError('Locked test: pass --unlock-test only at agreed Sync 4')
    if Path(args.output).exists():
        raise ValueError('Output exists; refusing to overwrite an evaluation record')
    maps=load_maps(args.maps)
    rows=read_manifest(args.manifest,maps)
    artifact=json.loads(Path(args.artifact).read_text())
    if artifact['label_maps_sha256']!=digest(args.maps) or artifact['manifest_sha256']!=digest(args.manifest):
        raise ValueError('Frozen maps or manifest changed since fitting')
    ids,arrays=load_bundle(args.bundle,rows,maps,args.split,digest(args.maps))
    y=truth(rows,ids,maps,'species')
    if (y<0).any():
        raise ValueError('Evaluation species labels must be complete')
    dnn,bn,used=predict(arrays,artifact)
    policy=artifact['policy']
    fused,proposed,accepted,stages=resolve(dnn,bn,policy['weight'],policy['threshold'])
    rank=(np.argsort(-bn,axis=1)[:,:5]==dnn.argmax(1)[:,None]).any(1)
    result=dict(split=args.split,kind=('synthetic_demo' if all(
                    s.startswith('SYNTHETIC_') for s in maps['species']) else 'measured_predictions'),n=len(ids),
                artifact_sha256=digest(args.artifact),bundle_sha256=digest(args.bundle),
                dnn=classification(dnn,y),bn=classification(bn,y),
                rank_rule=classification(dnn,y,rank),fusion=classification(fused,y),
                selective=classification(fused,y,accepted),policy=policy,
                accuracy_ci=bootstrap_accuracy(y,proposed),
                accepted_accuracy_ci=bootstrap_accuracy(y,proposed,accepted),
                mcnemar=mcnemar_exact(y,dnn.argmax(1),proposed),
                limitations=['IID bootstrap assumes independent images; use grouped bootstrap if source clusters remain correlated.',
                             'Empirical theta needs adequate TRAIN feature annotations.',
                             'Calibration selection is not a guarantee of test risk.'])
    write_json(args.output,result)
    write_json(str(args.output)+'.reliability.json',reliability(fused,y))
    write_json(str(args.output)+'.risk_coverage.json',risk_coverage(fused,y))
    records=[dict(image_id=str(i),proposed_species=maps['species'][p],
                  accepted_species=maps['species'][p] if a else None,
                  status='accepted' if a else 'deferred',stage=str(s),
                  confidence=float(fused[j].max()),
                  features_used={f:bool(used[f][j]) for f in FEATURES})
             for j,(i,p,a,s) in enumerate(zip(ids,proposed,accepted,stages))]
    write_json(str(args.output)+'.decisions.json',records)
    print(f"Evaluated {len(ids)} {args.split} rows -> {args.output}")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    for name in ('fit','evaluate'):
        s=sub.add_parser(name)
        for arg in ('maps','manifest','bundle','output'):
            s.add_argument('--'+arg,required=True)
        if name=='fit':
            s.add_argument('--variant',choices=['hard_direct','hard_confusion','direct_soft','virtual_soft'],default='direct_soft')
            s.add_argument('--risk',type=float,default=.015)
            s.add_argument('--minimum',type=int,default=50)
            s.set_defaults(func=fit)
        else:
            s.add_argument('--artifact',required=True)
            s.add_argument('--split',choices=['val','test'],default='val')
            s.add_argument('--unlock-test',action='store_true')
            s.set_defaults(func=evaluate)
    args=p.parse_args()
    args.func(args)


if __name__=='__main__':
    main()
