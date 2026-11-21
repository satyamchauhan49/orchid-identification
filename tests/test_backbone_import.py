import csv,json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from bn_resolver.import_backbone import ingest
from bn_resolver.contracts import FEATURES,digest

class BackboneImportChecks(unittest.TestCase):
    def fixture(self,r):
        labels={'species':[str(i) for i in range(1,157)],**{f:['a','b'] for f in FEATURES}}
        (r/'maps.json').write_text(json.dumps(labels))
        torch.save({'labels':labels,'args':{'dummy':False}},r/'best.pt')
        rows=[dict(image_id=s,path=s+'.png',species='1',split=s,phash_cluster=s,**{f:'a' for f in FEATURES}) for s in ('calib','val')]
        with (r/'manifest.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        for s in ('calib','val'):
            p=dict(ids=np.array([s]),species_logits=np.zeros((1,156)),**{'logits_'+f:np.zeros((1,2)) for f in FEATURES})
            p.update({k:np.array(digest(r/v)) for k,v in [('label_maps_sha256','maps.json'),('manifest_sha256','manifest.csv'),('checkpoint_sha256','best.pt')]})
            np.savez(r/(s+'.npz'),**p)
    def test_verified_serialization_adaptation(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);self.fixture(r);d=ingest(r,r/'maps.json',r/'manifest.csv',r/'out')
            self.assertNotEqual(d['canonical_label_maps_sha256'],d['original_hashes']['label_maps_sha256'])
            self.assertFalse(d['test_opened'])
    def test_reject_wrong_original_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);self.fixture(r);(r/'maps.json').write_text((r/'maps.json').read_text()+'\n')
            with self.assertRaisesRegex(ValueError,'provenance mismatch'):ingest(r,r/'maps.json',r/'manifest.csv',r/'out')
            self.assertFalse((r/'out').exists())
    def test_reject_checkpoint_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);self.fixture(r);ck=torch.load(r/'best.pt',weights_only=True);ck['epoch']=9;torch.save(ck,r/'best.pt')
            with self.assertRaisesRegex(ValueError,'checkpoint_sha256'):ingest(r,r/'maps.json',r/'manifest.csv',r/'out')

if __name__=='__main__':unittest.main()
