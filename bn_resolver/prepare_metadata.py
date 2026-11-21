"""Adapt Agrim's metadata without images, model logits or test evaluation.

Species IDs remain strings in numeric model order; names are a display sidecar.
All-split columns are checked solely for contract/group isolation. Counts, feature
priors and theta are fitted from TRAIN; no held-out performance is computed.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .contracts import FEATURES,digest,load_maps,read_manifest,truth,calibration_partition
from .core import empirical_theta
from .pipeline import write_json


def canonicalize(source):
    def ordered(value):
        if isinstance(value,dict):
            indices=list(value.values())
            if sorted(indices)!=list(range(len(indices))): raise ValueError('Class indices must be contiguous')
            return [str(k) for k,v in sorted(value.items(),key=lambda kv:kv[1])]
        if not isinstance(value,list): raise ValueError('Expected ordered list or indexed class map')
        return [str(v) for v in value]
    species=ordered(source['species'])
    features={f:ordered(source.get('features',source)[f]) for f in FEATURES}
    # Agrim's trainer builds sorted numeric IDs and alphabetical feature orders.
    if species!=sorted(species,key=int): raise ValueError('Species order differs from Agrim trainer numeric order')
    if any(v!=sorted(v) for v in features.values()): raise ValueError('Feature order differs from Agrim trainer alphabetical order')
    return dict(species=species,features=features)


def prepare(source_maps,manifest,output,revision):
    out=Path(output)
    if out.exists(): raise ValueError('Choose a new metadata output directory')
    source=json.loads(Path(source_maps).read_text())
    maps=canonicalize(source)
    out.mkdir(parents=True)
    write_json(out/'label_maps.json',maps)
    load_maps(out/'label_maps.json')
    rows=read_manifest(manifest,maps)
    # Preserve IDs, paths, splits, labels and exact CSV bytes. No reassignment.
    (out/'split_manifest.csv').write_bytes(Path(manifest).read_bytes())
    names=source.get('species_id_to_name',{})
    write_json(out/'species_names.json',{s:names.get(s) for s in maps['species']})
    train=[r for r in rows if r['split']=='train']; ids=[r['image_id'] for r in train]
    sy=truth(rows,ids,maps,'species')
    if (sy<0).any(): raise ValueError('TRAIN species labels required')
    artifact=dict(kind='real_train_metadata_only',species_order=maps['species'],
        prior=[1/156]*156,theta={},support={},feature_priors={},
        provenance='Empirical TRAIN labels, Dirichlet(1) smoothing; not expert botanical taxonomy',
        source_revision=revision,label_maps_sha256=digest(out/'label_maps.json'),manifest_sha256=digest(manifest))
    for f in FEATURES:
        y=truth(rows,ids,maps,f); k=len(maps['features'][f])
        theta,support=empirical_theta(sy,y,156,k)
        artifact['theta'][f]=theta.tolist();artifact['support'][f]=support.tolist()
        count=np.bincount(y[y>=0],minlength=k)+1
        artifact['feature_priors'][f]=(count/count.sum()).tolist()
    write_json(out/'train_bn_metadata.json',artifact)
    calib=[r for r in rows if r['split']=='calib']; cid=np.array([r['image_id'] for r in calib])
    cm=calibration_partition(rows,cid)
    from eval.experiments import divide_policy
    fw,ft=divide_policy(rows,cid,~cm)
    partitions={'cal_model':cm,'cal_fusion':fw,'cal_threshold':ft}
    report=dict(kind='real_metadata_audit',source_revision=revision,
        source_maps_sha256=digest(source_maps),canonical_maps_sha256=digest(out/'label_maps.json'),
        train_n=len(train),calibration_n=len(calib),feature_classes={f:len(maps['features'][f]) for f in FEATURES},
        partitions={},test_predictions_opened=False,test_metrics_computed=False,
        group_isolation='read_manifest passed for all declared splits; no image/NPZ opened',
        colour_mismatch='CL GreenRed absent; do not invent alias to PinkRed or claim matched literature replication')
    for name,mask in partitions.items():
        selected=cid[mask]; y=truth(rows,selected,maps,'species')
        report['partitions'][name]=dict(n=len(selected),ids=selected.tolist(),species_represented=len(np.unique(y[y>=0])),
            feature_support={f:np.bincount(truth(rows,selected,maps,f)[truth(rows,selected,maps,f)>=0],minlength=len(maps['features'][f])).tolist() for f in FEATURES})
    report['export_blocker']='Agrim export_split currently omits label_maps_sha256. Verify checkpoint class_maps.json against canonical map and re-export with hashes. Never attach assumed provenance to existing logits.'
    write_json(out/'audit.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('maps','manifest','output','source-revision'):p.add_argument('--'+key,required=True)
    a=p.parse_args();r=prepare(a.maps,a.manifest,a.output,a.source_revision)
    print(json.dumps({k:r[k] for k in ('train_n','calibration_n','feature_classes')},indent=2))


if __name__=='__main__':main()
