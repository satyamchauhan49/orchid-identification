import unittest
import numpy as np
from bn_resolver.arbiter import features,fit,choose


class ArbiterChecks(unittest.TestCase):
    def test_evidence_gap_removes_prior(self):
        d=np.array([[.8,.2]]); b=np.array([[.2,.8]])
        x=features(d,b,{'T':np.array([[.7,.3]])},[.2,.8])
        self.assertAlmostEqual(x[0,2],0)
        self.assertAlmostEqual(x[0,3],.7)

    def test_insufficient_support_falls_back(self):
        d=np.array([[.8,.2]]); b=np.array([[.2,.8]])
        m=fit(np.zeros((1,4)),np.array([1]),d,b)
        self.assertFalse(m['fitted'])
        np.testing.assert_equal(choose(np.zeros((1,4)),d,b,m),[0])

    def test_both_wrong_and_agreements_excluded(self):
        d=np.tile([.8,.1,.1],(50,1));b=np.tile([.1,.8,.1],(50,1))
        y=np.r_[np.zeros(20,dtype=int),np.ones(20,dtype=int),np.full(10,2)]
        x=np.random.default_rng(42).normal(size=(50,4))
        m=fit(x,y,d,b)
        self.assertEqual(m['support'],40)
        self.assertTrue(m['fitted'])

    def test_fit_decisive_disagreements_only(self):
        rng=np.random.default_rng(42)
        x=rng.normal(size=(100,4));y=(x[:,0]>0).astype(int)
        d=np.tile([.9,.1],(100,1));b=np.tile([.1,.9],(100,1))
        m=fit(x,y,d,b)
        self.assertTrue(m['fitted'])
        self.assertGreater((choose(x,d,b,m)==y).mean(),.9)


if __name__=='__main__': unittest.main()
