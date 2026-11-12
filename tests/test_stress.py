import copy
import unittest
import numpy as np
from eval.stress import scenarios
import test_contracts


class StressChecks(unittest.TestCase):
    def test_scenarios_preserve_inputs_shapes_and_determinism(self):
        arrays={f:np.arange(24,dtype=float).reshape(4,6) for f in ('species','T','NF','In','LC','CF','CL')}
        before=copy.deepcopy(arrays)
        a,b=scenarios(arrays),scenarios(arrays)
        self.assertEqual(len(a),6)
        for key,case in a.items():
            for f,x in case.items():
                self.assertEqual(x.shape,arrays[f].shape)
                self.assertTrue(np.isfinite(x).all())
                np.testing.assert_equal(x,b[key][f])
        for f in arrays: np.testing.assert_equal(arrays[f],before[f])
        self.assertFalse(a['all_uniform']['species'].any())


class StrongerContracts(unittest.TestCase):
    def setUp(self):
        fixture=test_contracts.ContractChecks()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.bundle=fixture.bundle
        self.load=fixture.load

    def test_missing_keys_actionable(self):
        b=dict(self.bundle); del b['logits_CF']
        with self.assertRaisesRegex(ValueError,'logits_CF'): self.load(b)

    def test_numeric_ids_rejected(self):
        b=dict(self.bundle,ids=np.array([1]))
        with self.assertRaisesRegex(ValueError,'strings'): self.load(b)


if __name__=='__main__': unittest.main()
