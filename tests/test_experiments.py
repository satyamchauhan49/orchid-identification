"""Integration checks with generated arrays only; never real test labels."""
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from scripts.make_synthetic import main as make_synthetic
from eval.experiments import run


class ExperimentChecks(unittest.TestCase):
    def test_full_suite_and_validation_label_isolation(self):
        try:
            import matplotlib
        except ImportError:
            self.skipTest('Install requirements-c-resolver.txt for figure integration check')
        with tempfile.TemporaryDirectory() as directory:
            previous=Path.cwd()
            try:
                os.chdir(directory)
                with contextlib.redirect_stdout(io.StringIO()):
                    make_synthetic()
                root=Path('work/synthetic')
                args=SimpleNamespace(maps=str(root/'label_maps.json'),
                    manifest=str(root/'split_manifest.csv'),calib=str(root/'calib.npz'),
                    val=str(root/'val.npz'),output='run-one',synthetic_only=True,
                    risk=.015,minimum=50)
                with contextlib.redirect_stdout(io.StringIO()): run(args)
                first=json.loads(Path('run-one/frozen_direct_soft.json').read_text())
                ledger=json.loads(Path('run-one/run_manifest.json').read_text())
                self.assertEqual(ledger['data_kind'],'synthetic_demo')
                self.assertFalse(ledger['test_accessed'])
                for file,count in [('E3_R3_bn_comparison.csv',10),
                                   ('E4_calibration.csv',7),('E5_R4_resolver.csv',6)]:
                    with Path('run-one',file).open() as stream:
                        table=list(csv.DictReader(stream))
                    self.assertEqual(len(table),count)
                    self.assertTrue(all(r['data_kind']=='synthetic_demo' for r in table))
                intervals=json.loads(Path('run-one/grouped_confidence_intervals.json').read_text())
                self.assertTrue(intervals['estimable'])
                self.assertEqual(intervals['data_kind'],'synthetic_demo')
                self.assertEqual(intervals['intervals']['accuracy_delta']['valid_resamples'],1000)
                for file in ('calibration_and_risk.png','bn_comparison.png','calibration_ablations.png'):
                    self.assertGreater(Path('run-one',file).stat().st_size,1000)
                examples=json.loads(Path('run-one/example_records.json').read_text())
                self.assertEqual(len(examples),10)
                self.assertEqual(len({r['image_id'] for r in examples}),10)
                self.assertEqual({r['decision']['stage'] for r in examples},
                    {'trusted_agreement','auto_resolved','low_confidence'})
                for r in examples:
                    self.assertFalse(r['visual_paths'])
                    if r['decision']['status']=='deferred':
                        self.assertIsNone(r['decision']['accepted'])
                with self.assertRaisesRegex(ValueError,'already exists'):
                    run(args)
                # Validation truth must affect scores, never frozen temperatures/policy.
                manifest=Path(args.manifest)
                with manifest.open() as stream:
                    rows=list(csv.DictReader(stream))
                for row in rows:
                    if row['split']=='val': row['species']='SYNTHETIC_species_000'
                with manifest.open('w',newline='') as stream:
                    writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
                    writer.writeheader(); writer.writerows(rows)
                args.output='run-two'
                with contextlib.redirect_stdout(io.StringIO()): run(args)
                second=json.loads(Path('run-two/frozen_direct_soft.json').read_text())
                for key in ('temperatures','theta','matrices','policy','calibration_partitions'):
                    self.assertEqual(first[key],second[key],key)
            finally:
                os.chdir(previous)


if __name__=='__main__': unittest.main()
