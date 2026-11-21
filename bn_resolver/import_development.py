"""Audited legacy development import, without claiming exporter provenance.

Verifies IDs/targets/class order, retains original files, and records post-hoc map
binding separately. Does NOT assert checkpoint identity or original manifest hash.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from .contracts import load_maps,read_manifest,truth,digest,FEATURES
from .prepare_metadata import canonicalize
from .pipeline import write_json


def ingest(source,maps_path,manifest,output):
    source,out=Path(source),Path(output)
    if out.exists():raise ValueError('Choose a new import directory')
    maps=load_maps(maps_path);rows=read_manifest(manifest,maps)
    original=json.loads((source/'class_maps.json').read_text())
    if canonicalize(original)!=maps:raise ValueError('Supplied class map mismatch')
    ck=torch.load(source/'best.pt',map_location='cpu',weights_only=True)
    if canonicalize(ck['label_maps'])!=maps:raise ValueError('Checkpoint order mismatch')
    ledger=dict(kind='audited_legacy_development_import',checkpoint_linkage='UNVERIFIED: arrays lack exporter hashes; checkpoint and maps supplied together',
        canonical_map_binding='Post-hoc verified class order and manifest target agreement; not an original exporter attestation',
        supplied_checkpoint_sha256=digest(source/'best.pt'),class_maps_sha256=digest(source/'class_maps.json'),
        manifest_sha256=digest(manifest),test_opened=False,splits={})
    payloads={}
    for split in ('calib','val'):
        with np.load(source/(split+'.npz'),allow_pickle=False) as z:
            ids=z['ids'].astype(str)
            expected={r['image_id'] for r in rows if r['split']==split}
            if len(set(ids))!=len(ids) or set(ids)!=expected:raise ValueError(split+' ID mismatch')
            payload={'ids':ids,'label_maps_sha256':np.array(digest(maps_path)),
                'provenance_status':np.array('posthoc_class_order_verified_checkpoint_unverified')}
            for k in ('species',*FEATURES):
                target=z['targets_'+k];y=truth(rows,ids,maps,k)
                if not np.array_equal(target,y):raise ValueError(split+' '+k+' target mismatch')
                key='species_logits' if k=='species' else 'logits_'+k
                a=z[key];width=len(maps['species']) if k=='species' else len(maps['features'][k])
                if a.shape!=(len(ids),width) or not np.isfinite(a).all():raise ValueError('Invalid '+key)
                payload[key]=a
            payloads[split]=payload
            ledger['splits'][split]=dict(rows=len(ids),original_sha256=digest(source/(split+'.npz')))
    out.mkdir(parents=True)
    for split,payload in payloads.items():
        np.savez_compressed(out/(split+'.npz'),**payload)
        ledger['splits'][split]['derived_sha256']=digest(out/(split+'.npz'))
    write_json(out/'IMPORT_PROVENANCE.json',ledger)
    return ledger


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','maps','manifest','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();print(json.dumps(ingest(a.source,a.maps,a.manifest,a.output),indent=2))

if __name__=='__main__':main()
