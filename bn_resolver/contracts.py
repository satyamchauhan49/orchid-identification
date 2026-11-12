"""Strict interfaces shared with Agrim and Sneha."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from .core import labels

FEATURES = ('T', 'NF', 'In', 'LC', 'CF', 'CL')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_maps(path, expected=156):
    m = json.loads(Path(path).read_text())
    if len(m['species']) != expected:
        raise ValueError(f"Expected {expected} verified species; received {len(m['species'])}")
    for key in ('species', *FEATURES):
        values = m['species'] if key == 'species' else m['features'][key]
        if not isinstance(values, list) or not values or any(not isinstance(v,str) or not v for v in values) or len(set(values)) != len(values):
            raise ValueError(f"{key}: expected unique nonempty names in fixed class order")
    return m


def read_manifest(path, maps):
    rows = list(csv.DictReader(Path(path).open()))
    required = {'image_id','path','species','phash_cluster','split',*FEATURES}
    if not rows or not required <= rows[0].keys():
        raise ValueError(f"Manifest requires {sorted(required)}")
    ids, groups = set(), {}
    for r in rows:
        if not r['image_id'] or r['image_id'] in ids:
            raise ValueError("Empty or duplicate image_id")
        ids.add(r['image_id'])
        if r['split'] not in {'train','calib','val','test'}:
            raise ValueError("Unknown split")
        if not r['phash_cluster']:
            raise ValueError("Missing duplicate cluster")
        prior = groups.setdefault(r['phash_cluster'],r['split'])
        if prior != r['split']:
            raise ValueError("Duplicate cluster crosses splits")
        for key in ('species',*FEATURES):
            domain = maps['species'] if key == 'species' else maps['features'][key]
            if r[key] and r[key] not in domain:
                raise ValueError(f"Unknown {key} label: {r[key]}")
    return rows


def truth(rows, ids, maps, key):
    lookup = {r['image_id']:r for r in rows}
    domain = maps['species'] if key == 'species' else maps['features'][key]
    index = {v:i for i,v in enumerate(domain)}
    return np.array([index.get(lookup[str(i)][key],-1) for i in ids], dtype=int)


def load_bundle(path, rows, maps, split, map_hash):
    """Never loads pickle; exact IDs, dimensions and label-map hash are mandatory."""
    with np.load(path,allow_pickle=False) as z:
        ids = z['ids'].astype(str)
        if ids.ndim != 1 or len(set(ids)) != len(ids):
            raise ValueError("Bundle IDs must be unique")
        expected = {r['image_id'] for r in rows if r['split']==split}
        if set(ids) != expected:
            raise ValueError(f"{split}: bundle IDs do not exactly match manifest")
        if str(z['label_maps_sha256'].item()) != map_hash:
            raise ValueError("Model class order not verified: label-map hash mismatch")
        arrays = {}
        for key in ('species',*FEATURES):
            name = 'species_logits' if key == 'species' else 'logits_'+key
            k = len(maps['species']) if key == 'species' else len(maps['features'][key])
            a = np.asarray(z[name],dtype=float)
            if a.shape != (len(ids),k) or not np.isfinite(a).all():
                raise ValueError(f"{name}: expected finite {(len(ids),k)} array")
            arrays[key]=a
    return ids, arrays


def calibration_partition(rows, ids):
    """Assign complete duplicate clusters, reproducibly, without examining labels.

    Hash parity avoids a sample's partition changing when another ID is added.
    Agrim should review class support before freezing the manifest.
    """
    lookup = {r['image_id']:r for r in rows}
    return np.array([int(hashlib.sha256(('42:'+lookup[str(i)]['phash_cluster']).encode()).hexdigest(),16)%2==0 for i in ids])
