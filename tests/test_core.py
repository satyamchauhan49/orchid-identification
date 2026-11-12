import unittest
import numpy as np
from bn_resolver.core import (softmax,nll,fit_temperature,confusion_cpt,
    empirical_theta,bn_posterior,fuse,fit_policy,resolve,risk_coverage)
from eval.metrics import mcnemar_exact,classification


class ScientificChecks(unittest.TestCase):
    def test_temperature_reduces_overconfidence(self):
        x=np.tile([8.,0.],(100,1))
        y=np.r_[np.zeros(70,dtype=int),np.ones(30,dtype=int)]
        result=fit_temperature(x,y)
        self.assertGreater(result['temperature'],1)
        self.assertLess(result['nll_after'],result['nll_before'])

    def test_cpt_orientation_and_missing(self):
        q=np.array([[.9,.1],[.8,.2],[.1,.9]])
        m,support=confusion_cpt(q,np.array([0,1,-1]))
        np.testing.assert_allclose(m,[[2/3,2/3],[1/3,1/3]])
        np.testing.assert_equal(support,[1,1])

    def test_all_missing_returns_nonuniform_prior(self):
        prior=[.8,.2]
        th={'T':np.array([[.9,.1],[.1,.9]])}
        p,used=bn_posterior(prior,th,{'T':np.array([[.5,.5]])},
                            feature_priors={'T':[.5,.5]},missing_confidence=.6)
        np.testing.assert_allclose(p,[prior])
        self.assertFalse(used['T'][0])

    def test_bayes_hand_calculation(self):
        th={'T':np.array([[.9,.1],[.2,.8]])}
        q={'T':np.array([[.8,.2]])}
        p,_=bn_posterior([.5,.5],th,q,feature_priors={'T':[.5,.5]},missing_confidence=0)
        expected=np.array([.74,.32])
        np.testing.assert_allclose(p[0],expected/expected.sum())

    def test_all_four_variants_identity_confusion(self):
        th={'T':np.array([[.9,.1],[.2,.8]])}
        q={'T':np.array([[1.,0.]])}
        for v in ('hard_direct','hard_confusion','direct_soft','virtual_soft'):
            p,_=bn_posterior([.5,.5],th,q,{'T':np.eye(2)}, {'T':[.5,.5]},v,0)
            np.testing.assert_allclose(p[0],[.9/1.1,.2/1.1])

    def test_fusion_endpoints_156_classes(self):
        rng=np.random.default_rng(42)
        a,b=softmax(rng.normal(size=(4,156))),softmax(rng.normal(size=(4,156)))
        np.testing.assert_allclose(fuse(a,b,1),a)
        np.testing.assert_allclose(fuse(a,b,0),b)

    def test_no_false_trusted_tag(self):
        a=np.array([[.8,.2]])
        b=np.array([[.1,.9]])
        _,pred,accepted,stages=resolve(a,b,0,.5)
        self.assertEqual(stages[0],'auto_resolved')
        self.assertEqual(pred[0],1)

    def test_infeasible_policy_defers_all(self):
        a=np.tile([.8,.2],(60,1))
        policy,_=fit_policy(a,a,np.ones(60,dtype=int))
        self.assertFalse(policy['feasible'])
        _,_,accepted,_=resolve(a,a,policy['weight'],policy['threshold'])
        self.assertFalse(accepted.any())

    def test_ties_cannot_be_partially_accepted(self):
        a=np.array([[.9,.1],[.9,.1],[.6,.4]])
        curve=risk_coverage(a,np.array([0,1,0]))
        self.assertEqual([r['accepted'] for r in curve],[2,3])

    def test_missing_theta_rows_uniform(self):
        th,support=empirical_theta(np.array([0,0]),np.array([0,-1]),2,2)
        np.testing.assert_allclose(th,[[2/3,1/3],[.5,.5]])
        np.testing.assert_equal(support,[1,0])

    def test_invalid_inputs_rejected(self):
        with self.assertRaises(ValueError):
            softmax([[np.nan,1]])
        with self.assertRaises(ValueError):
            nll([[.5,.5]],np.array([2]))

    def test_exact_mcnemar(self):
        result=mcnemar_exact(np.zeros(4,dtype=int),np.zeros(4,dtype=int),np.ones(4,dtype=int))
        self.assertAlmostEqual(result['p_value'],.125)

    def test_zero_coverage_accuracy_is_null(self):
        r=classification(np.array([[.7,.3]]),np.array([0]),[False])
        self.assertIsNone(r['accepted_accuracy'])


if __name__=='__main__':
    unittest.main()
