#------------------------------------------------------------------------------#
# Header
#------------------------------------------------------------------------------#
import unittest as ut

import sys, os
if __name__ == '__main__':
    path_hack = "%s/../.." % (os.path.dirname(__file__),)
    if sys.path[0] != path_hack:
        sys.path.insert(0, path_hack)

#------------------------------------------------------------------------------#
# Body
#------------------------------------------------------------------------------#
import numpy as np

from genlearn.features import features

class TestFeatures(ut.TestCase):
    def setUp(self):
        self.identity = lambda x: x
        self.square = lambda x: x**2
        self.inverse = lambda x: 1.0 / x
        self.zero = lambda x: 0.0

        self.feats = features([self.identity,
                               self.square,
                               self.inverse,
                               self.zero])

    def testFeatureLength(self):
        self.assertEqual(len(self.feats), 4)

    def testFeatureValues(self):
        self.assertTrue( (np.array([2.0, 4.0, 0.5, 0.0]) == self.feats.get(2.0)).all() )
        self.assertTrue( (np.array([1.0, 1.0, 1.0, 0.0]) == self.feats.get(1.0)).all() )
