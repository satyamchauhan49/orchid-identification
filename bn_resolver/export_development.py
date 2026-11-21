"""Fresh calibration/validation exports from trusted A/B checkpoints.

Never opens test images or test NPZ. Never assigns hashes to old logits. Requires
canonical nested maps and validates the checkpoint's ordered head labels.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from bn_resolver.contracts import load_maps,read_manifest,digest,FEATURES
from bn_resolver.pipeline import write_json
from explain.checkpoints import load_checkpoint
from models_other.preprocessing import image_transform


def export(maps_path,manifest,checkpoint,image_root,output,batch_size=16):
    if batch_size<1:raise ValueError('Positive batch size required')
    out=Path(output)
    if out.exists():raise ValueError('Choose a new output directory')
    maps=load_maps(maps_path);rows=read_manifest(manifest,maps)
    model,labels,cfg,backbone,*_=load_checkpoint(checkpoint)
    if labels!={'species':maps['species'],**maps['features']}:raise ValueError('Checkpoint class order mismatch')
    if not cfg.get('mean') or not cfg.get('std'):raise ValueError('Stored training normalization required')
    transform=image_transform(cfg['mean'],cfg['std'],cfg['preprocessing'],False)
    root=Path(image_root).resolve()
    splits={s:[r for r in rows if r['split']==s] for s in ('calib','val')}
    if any(not r for r in splits.values()):raise ValueError('Both calib and val rows required')
    # Check only development paths. Reject accidental traversal outside dataset root.
    for group in splits.values():
        for row in group:
            path=(root/row['path']).resolve()
            if not path.is_relative_to(root) or not path.is_file():raise ValueError('Missing/unsafe development image: '+row['image_id'])
    out.mkdir(parents=True);model.eval()
    provenance=dict(label_maps_sha256=digest(maps_path),manifest_sha256=digest(manifest),checkpoint_sha256=digest(checkpoint))
    ledger=dict(kind='development_reexport',backbone=backbone,preprocessing=cfg['preprocessing'],
        normalization={k:cfg[k] for k in ('mean','std')},test_opened=False,
        synthetic_checkpoint=bool(cfg.get('dummy') or cfg.get('demo_data')),**provenance,splits={})
    for split,group in splits.items():
        outputs={k:[] for k in labels}
        for start in range(0,len(group),batch_size):
            batch=[]
            for row in group[start:start+batch_size]:
                # Match training dataset behavior: RGB conversion, no implicit EXIF reorientation.
                with Image.open(root/row['path']) as image:batch.append(transform(image.convert('RGB')))
            with torch.inference_mode():pred,_=model(torch.stack(batch))
            for key in outputs:
                values=pred[key].cpu().numpy()
                if values.shape!=(len(batch),len(labels[key])) or not np.isfinite(values).all():raise ValueError('Invalid model logits')
                outputs[key].append(values)
        payload=dict(ids=np.array([r['image_id'] for r in group]),**{k:np.array(v) for k,v in provenance.items()})
        payload['species_logits']=np.concatenate(outputs['species'])
        payload.update({'logits_'+f:np.concatenate(outputs[f]) for f in FEATURES})
        np.savez_compressed(out/(split+'.npz'),**payload)
        ledger['splits'][split]=dict(n=len(group),sha256=digest(out/(split+'.npz')))
    write_json(out/'export_provenance.json',ledger)
    return ledger


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('maps','manifest','checkpoint','image-root','output'):p.add_argument('--'+key,required=True)
    p.add_argument('--batch-size',type=int,default=16)
    a=p.parse_args();torch.set_num_threads(2)
    print(json.dumps(export(a.maps,a.manifest,a.checkpoint,a.image_root,a.output,a.batch_size),indent=2))

if __name__=='__main__':main()
