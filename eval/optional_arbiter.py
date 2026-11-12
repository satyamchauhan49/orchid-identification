"""Fit/evaluate the plan's optional arbiter on development bundles only."""
import argparse
import json
from pathlib import Path
import numpy as np
from bn_resolver import arbiter
from bn_resolver.contracts import load_maps,read_manifest,load_bundle,digest,truth,FEATURES
from bn_resolver.core import softmax
from bn_resolver.pipeline import predict,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('maps','manifest','calib','val','artifact','output'): p.add_argument('--'+key,required=True)
    a=p.parse_args()
    if Path(a.output).exists(): raise ValueError('Output exists')
    maps=load_maps(a.maps); rows=read_manifest(a.manifest,maps)
    artifact=json.loads(Path(a.artifact).read_text())
    if artifact['label_maps_sha256']!=digest(a.maps) or artifact['manifest_sha256']!=digest(a.manifest):
        raise ValueError('Frozen map/manifest mismatch')
    inputs=[]
    for split,path in [('calib',a.calib),('val',a.val)]:
        ids,arrays=load_bundle(path,rows,maps,split,digest(a.maps))
        y=truth(rows,ids,maps,'species')
        if (y<0).any(): raise ValueError('Complete development species truth required')
        d,b,_=predict(arrays,artifact)
        q={f:softmax(arrays[f]/artifact['temperatures'][f]['temperature']) for f in FEATURES}
        x=arbiter.features(d,b,q,artifact['prior'])
        inputs.append((ids,x,y,d,b))
    ids,x,y,d,b=inputs[0]
    allowed=set(artifact['calibration_partitions']['cal_fusion'])
    mask=np.array([str(i) in allowed for i in ids])
    model=arbiter.fit(x[mask],y[mask],d[mask],b[mask])
    vid,vx,vy,vd,vb=inputs[1]
    pred=arbiter.choose(vx,vd,vb,model)
    write_json(a.output,dict(data_kind='synthetic_demo' if all(s.startswith('SYNTHETIC_') for s in maps['species']) else 'development_validation',
        optional=True,test_accessed=False,fit_ids=ids[mask].tolist(),model=model,
        validation_n=len(vid),validation_accuracy=float((pred==vy).mean()),
        dnn_accuracy=float((vd.argmax(1)==vy).mean()),bn_accuracy=float((vb.argmax(1)==vy).mean()),
        caveat='Exploratory candidate choice only. No calibrated confidence/deferral or test performance claim. Ignores both-wrong disagreements in supervised fitting; do not replace frozen E5 policy without independent selection.'))
    print(a.output)


if __name__=='__main__': main()
