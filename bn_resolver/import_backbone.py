"""Verify Role B checkpoint exports and adapt map serialization transparently.

Strict path: requires original map bytes and exporter hashes. No test bundle input.
"""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from models_other.label_contract import read_maps
from .contracts import load_maps,read_manifest,load_bundle,digest,FEATURES
from .pipeline import write_json


def ingest(source,original_maps,manifest,output):
    source,out=Path(source),Path(output)
    if out.exists():raise ValueError('Choose a new output directory')
    labels=read_maps(original_maps)
    checkpoint=source/'best.pt';ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if ck.get('labels')!=labels:raise ValueError('Checkpoint label order mismatch')
    if ck.get('args',{}).get('dummy') or ck.get('args',{}).get('demo_data'):raise ValueError('Real backbone handoff cannot use synthetic checkpoint')
    canonical={'species':labels['species'],'features':{f:labels[f] for f in FEATURES}}
    # Hash canonical bytes in memory before creating output.
    import hashlib
    map_bytes=(json.dumps(canonical,indent=2,allow_nan=False)+'\n').encode()
    canonical_hash=hashlib.sha256(map_bytes).hexdigest()
    rows=read_manifest(manifest,canonical)
    hashes={'label_maps_sha256':digest(original_maps),'manifest_sha256':digest(manifest),'checkpoint_sha256':digest(checkpoint)}
    payloads={};ledger=dict(kind='verified_backbone_development_import',test_opened=False,
        original_hashes=hashes,canonical_label_maps_sha256=canonical_hash,
        adaptation='Same labels/order; serialization change recorded explicitly',splits={})
    for split in ('calib','val'):
        with np.load(source/(split+'.npz'),allow_pickle=False) as z:
            for k,v in hashes.items():
                if k not in z or str(z[k].item())!=v:raise ValueError(split+' provenance mismatch: '+k+'; obtain original bytes or fresh exports, do not guess hashes')
            ids=z['ids'].astype(str)
            if set(ids)!={r['image_id'] for r in rows if r['split']==split} or len(ids)!=len(set(ids)):raise ValueError(split+' ID mismatch')
            payload=dict(ids=ids,label_maps_sha256=np.array(canonical_hash),
                original_label_maps_sha256=np.array(hashes['label_maps_sha256']),
                manifest_sha256=np.array(hashes['manifest_sha256']),checkpoint_sha256=np.array(hashes['checkpoint_sha256']))
            for head in ('species',*FEATURES):
                key='species_logits' if head=='species' else 'logits_'+head
                a=z[key]
                if a.shape!=(len(ids),len(labels[head])) or not np.isfinite(a).all():raise ValueError('Invalid '+key)
                payload[key]=a
            payloads[split]=payload
            ledger['splits'][split]=dict(rows=len(ids),original_sha256=digest(source/(split+'.npz')))
    out.mkdir(parents=True);(out/'label_maps.json').write_bytes(map_bytes)
    for split,payload in payloads.items():
        np.savez_compressed(out/(split+'.npz'),**payload)
        ledger['splits'][split]['derived_sha256']=digest(out/(split+'.npz'))
        load_bundle(out/(split+'.npz'),rows,canonical,split,canonical_hash)
    write_json(out/'IMPORT_PROVENANCE.json',ledger)
    return ledger


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('source','original-maps','manifest','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();print(json.dumps(ingest(a.source,a.original_maps,a.manifest,a.output),indent=2))

if __name__=='__main__':main()
