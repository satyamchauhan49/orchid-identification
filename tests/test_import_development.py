import csv,json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from bn_resolver.import_development import ingest
from bn_resolver.contracts import FEATURES

class ImportChecks(unittest.TestCase):
    def fixture(self,root):
        maps={'species':[str(i) for i in range(1,157)],'features':{f:['a','b'] for f in FEATURES}}
        (root/'maps.json').write_text(json.dumps(maps))
        indexed={k:{v:i for i,v in enumerate(values)} for k,values in {'species':maps['species'],**maps['features']}.items()}
        source=root/'source';source.mkdir();(source/'class_maps.json').write_text(json.dumps(indexed));torch.save({'label_maps':indexed},source/'best.pt')
        rows=[]
        for split in ('calib','val'):
            rows.append(dict(image_id=split,path=split+'.png',species='1',split=split,phash_cluster=split,**{f:'a' for f in FEATURES}))
            payload=dict(ids=np.array([split]),species_logits=np.zeros((1,156)),targets_species=np.array([0]))
            for f in FEATURES:payload['logits_'+f]=np.zeros((1,2));payload['targets_'+f]=np.array([0])
            np.savez(source/(split+'.npz'),**payload)
        with (root/'manifest.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        return source
    def test_explicit_unverified_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);s=self.fixture(r);ledger=ingest(s,r/'maps.json',r/'manifest.csv',r/'out')
            self.assertIn('UNVERIFIED',ledger['checkpoint_linkage'])
            with np.load(r/'out/calib.npz') as z:self.assertNotIn('checkpoint_sha256',z.files)
            self.assertFalse(ledger['test_opened'])
    def test_target_mismatch_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);s=self.fixture(r)
            with np.load(s/'val.npz') as z:p={k:z[k] for k in z.files}
            p['targets_species']=np.array([1]);np.savez(s/'val.npz',**p)
            with self.assertRaisesRegex(ValueError,'target mismatch'):ingest(s,r/'maps.json',r/'manifest.csv',r/'out')
            self.assertFalse((r/'out').exists())

if __name__=='__main__':unittest.main()
