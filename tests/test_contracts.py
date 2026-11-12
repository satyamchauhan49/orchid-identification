import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from bn_resolver.contracts import (FEATURES,load_maps,read_manifest,load_bundle,
                                   calibration_partition,digest)
from bn_resolver.pipeline import evaluate
from eval.experiments import divide_policy,explanations,is_synthetic
from types import SimpleNamespace


class ContractChecks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.maps={'species':[f'SYNTHETIC_{i}' for i in range(156)],
                   'features':{f:['a','b'] for f in FEATURES}}
        self.mp=self.root/'maps.json'
        self.mp.write_text(json.dumps(self.maps))
        self.rows=[dict(image_id='v1',path='v1.png',species='SYNTHETIC_0',
                       phash_cluster='g1',split='val',**{f:'a' for f in FEATURES})]
        self.bundle=dict(ids=np.array(['v1']),species_logits=np.zeros((1,156)),
                         label_maps_sha256=np.array(digest(self.mp)),
                         **{'logits_'+f:np.zeros((1,2)) for f in FEATURES})

    def manifest(self,rows):
        path=self.root/'manifest.csv'
        with path.open('w') as stream:
            w=csv.DictWriter(stream,fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        return path

    def load(self,bundle=None,split='val'):
        p=self.root/'bundle.npz'
        np.savez(p,**(self.bundle if bundle is None else bundle))
        return load_bundle(p,self.rows,self.maps,split,digest(self.mp))

    def test_verified_species_count(self):
        self.assertEqual(len(load_maps(self.mp)['species']),156)
        self.maps['species'].pop()
        self.mp.write_text(json.dumps(self.maps))
        with self.assertRaisesRegex(ValueError,'156'):
            load_maps(self.mp)

    def test_duplicate_names(self):
        self.maps['features']['T']=['a','a']
        self.mp.write_text(json.dumps(self.maps))
        with self.assertRaises(ValueError): load_maps(self.mp)

    def test_duplicate_group_cross_split(self):
        rows=self.rows+[dict(self.rows[0],image_id='v2',split='train')]
        with self.assertRaisesRegex(ValueError,'crosses'):
            read_manifest(self.manifest(rows),self.maps)

    def test_unknown_annotation(self):
        rows=[dict(self.rows[0],CF='unknown')]
        with self.assertRaisesRegex(ValueError,'Unknown CF'):
            read_manifest(self.manifest(rows),self.maps)

    def test_missing_feature_label_is_allowed(self):
        rows=[dict(self.rows[0],CF='')]
        self.assertEqual(read_manifest(self.manifest(rows),self.maps)[0]['CF'],'')

    def test_wrong_split_bundle_rejected(self):
        with self.assertRaisesRegex(ValueError,'IDs'):
            self.load(split='calib')

    def test_wrong_order_hash_rejected(self):
        b=dict(self.bundle,label_maps_sha256=np.array('different-order'))
        with self.assertRaisesRegex(ValueError,'hash mismatch'): self.load(b)

    def test_wrong_class_dimension(self):
        b=dict(self.bundle,species_logits=np.zeros((1,63)))
        with self.assertRaisesRegex(ValueError,'156'): self.load(b)

    def test_nonfinite_logits(self):
        b=copy.deepcopy(self.bundle); b['logits_T'][0,0]=np.nan
        with self.assertRaisesRegex(ValueError,'finite'): self.load(b)

    def test_duplicate_ids(self):
        b=dict(self.bundle,ids=np.array(['v1','v1']))
        with self.assertRaisesRegex(ValueError,'unique'): self.load(b)

    def test_grouped_partitions_disjoint(self):
        rows=[dict(image_id=str(i),phash_cluster=str(i//2)) for i in range(100)]
        ids=np.array([r['image_id'] for r in rows])
        cm=calibration_partition(rows,ids)
        fw,ft=divide_policy(rows,ids,~cm)
        np.testing.assert_equal(cm.astype(int)+fw.astype(int)+ft.astype(int),1)
        for mask in (cm,fw,ft):
            np.testing.assert_equal(mask[::2],mask[1::2])

    def test_test_lock_precedes_file_access(self):
        with self.assertRaisesRegex(ValueError,'Locked test'):
            evaluate(SimpleNamespace(split='test',unlock_test=False))

    def test_synthetic_detection(self):
        self.assertTrue(is_synthetic(self.maps))
        self.maps['species'][0]='Real name'
        self.assertFalse(is_synthetic(self.maps))


if __name__=='__main__': unittest.main()
