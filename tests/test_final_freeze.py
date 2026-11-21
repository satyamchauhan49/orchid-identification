import json,tempfile,unittest
from pathlib import Path
from bn_resolver.contracts import digest
from eval.final_convnext import run,verify

class FinalFreezeChecks(unittest.TestCase):
    def test_lock_precedes_all_file_access(self):
        with self.assertRaisesRegex(ValueError,'Locked test'):
            run('nonexistent','nonexistent','nonexistent','nonexistent',False)
    def test_file_change_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);(r/'model').write_bytes(b'abc')
            f=r/'freeze.json';f.write_text(json.dumps({'files':{'checkpoint':{'path':'model','sha256':digest(r/'model')}}}))
            self.assertIn('files',verify(f,r))
            (r/'model').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'changed'):verify(f,r)

if __name__=='__main__':unittest.main()
