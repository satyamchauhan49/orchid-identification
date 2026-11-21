import unittest
from bn_resolver.prepare_metadata import canonicalize
from bn_resolver.contracts import FEATURES

class MetadataChecks(unittest.TestCase):
    def test_lists_preserve_numeric_order_as_strings(self):
        m={'species':list(range(1,157)),**{f:['a','b'] for f in FEATURES}}
        c=canonicalize(m)
        self.assertEqual(c['species'][9],'10')
        self.assertEqual(c['features']['CF'],['a','b'])
    def test_indexed_checkpoint_order(self):
        m={'species':{'2':1,'1':0},**{f:{'b':1,'a':0} for f in FEATURES}}
        self.assertEqual(canonicalize(m)['species'],['1','2'])
    def test_reject_reordered_features(self):
        m={'species':[1,2],**{f:['b','a'] for f in FEATURES}}
        with self.assertRaises(ValueError):canonicalize(m)
    def test_reject_ambiguous_indices(self):
        m={'species':{'1':0,'2':0},**{f:['a','b'] for f in FEATURES}}
        with self.assertRaises(ValueError):canonicalize(m)
