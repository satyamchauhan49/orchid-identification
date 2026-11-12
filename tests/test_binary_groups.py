import unittest
import numpy as np
from bn_resolver.binary import fit_binary_rates,binary_posterior
from eval.metrics import grouped_bootstrap


class BinaryChecks(unittest.TestCase):
    def test_rates_hand_counts_and_missing(self):
        q=np.array([[.9,.1],[.9,.1],[.1,.9],[.1,.9],[.9,.1]])
        r=fit_binary_rates(q,np.array([0,1,1,1,-1]))
        self.assertEqual((r[0]['tp'],r[0]['fn'],r[0]['tn'],r[0]['fp']),(1,0,2,1))
        self.assertAlmostEqual(r[0]['tpr'],2/3)
        self.assertAlmostEqual(r[0]['tnr'],3/5)

    def test_selected_value_likelihood(self):
        # With deterministic species features, posterior is proportional to .8 and .1.
        theta={'T':np.eye(2)}
        q={'T':np.array([[.9,.1]])}
        rates={'T':[dict(tpr=.8,tnr=.9),dict(tpr=.7,tnr=.6)]}
        p,_=binary_posterior([.5,.5],theta,q,rates,0)
        np.testing.assert_allclose(p,[[8/9,1/9]])

    def test_other_detectors_not_multiplied(self):
        theta={'T':np.eye(2)}; q={'T':np.array([[.9,.1]])}
        rates={'T':[dict(tpr=.8,tnr=.9),dict(tpr=.001,tnr=.001)]}
        p,_=binary_posterior([.5,.5],theta,q,rates,0)
        np.testing.assert_allclose(p,[[8/9,1/9]])

    def test_mask_returns_prior(self):
        p,used=binary_posterior([.7,.3],{'T':np.eye(2)},
            {'T':np.array([[.5,.5]])},{'T':[dict(tpr=.8,tnr=.9)]*2},.6)
        np.testing.assert_allclose(p,[[.7,.3]])
        self.assertFalse(used['T'][0])

    def test_unsupported_rates_flagged_and_smoothed(self):
        r=fit_binary_rates(np.array([[.8,.2]]),np.array([-1]))
        self.assertEqual(r[0]['positive_support'],0)
        self.assertIsNone(r[0]['empirical_tpr'])
        self.assertEqual(r[0]['tpr'],.5)

    def test_invalid_rates(self):
        with self.assertRaises(ValueError):
            binary_posterior([.5,.5],{'T':np.eye(2)},{'T':np.array([[.8,.2]])},
                {'T':[dict(tpr=2,tnr=.9)]*2})


class GroupChecks(unittest.TestCase):
    def test_whole_clusters_not_individual_images(self):
        y=np.zeros(20,dtype=int); pred=np.r_[np.zeros(10,dtype=int),np.ones(10,dtype=int)]
        r=grouped_bootstrap(y,pred,pred,np.array(['a']*10+['b']*10),resamples=1000)
        self.assertEqual(r['intervals']['final_accuracy']['low'],0)
        self.assertEqual(r['intervals']['final_accuracy']['high'],1)
        self.assertEqual(r['intervals']['accuracy_delta']['low'],0)
        self.assertEqual(r['intervals']['accuracy_delta']['high'],0)

    def test_single_cluster_not_estimable(self):
        r=grouped_bootstrap([0],[0],[0],['a'])
        self.assertFalse(r['estimable'])
        self.assertIsNone(r['intervals']['final_accuracy']['low'])

    def test_zero_accepted_interval_null(self):
        r=grouped_bootstrap([0,1],[0,1],[0,1],['a','b'],[False,False])
        self.assertIsNone(r['intervals']['accepted_accuracy']['low'])
        self.assertEqual(r['intervals']['coverage']['high'],0)

    def test_deterministic_seed_and_unequal_cluster_sizes(self):
        args=([0,0,0],[1,0,0],[0,0,1],['a','a','b'])
        a=grouped_bootstrap(*args,resamples=40)
        self.assertEqual(a,grouped_bootstrap(*args,resamples=40))
        self.assertEqual(a['group_count'],2)

    def test_missing_group_rejected(self):
        with self.assertRaises(ValueError): grouped_bootstrap([0],[0],[0],[''])
        with self.assertRaises(ValueError): grouped_bootstrap([0],[0],[0],['a'],resamples=0)


if __name__=='__main__': unittest.main()
