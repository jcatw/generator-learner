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
        
from genlearn.basis import basis

class TestBasis(ut.TestCase):
    def setUp(self):
        self.fullPoly = [lambda x: 1,
                         lambda x: x,
                         lambda x: x**2,
                         lambda x: x**3,
                         lambda x: x**4]
        
    def testBasisLength(self):
        fullPolyBasis = basis(self.fullPoly)
        self.assertEqual( len(fullPolyBasis), 5)

    def testBasisFloatCoerce(self):
        fullPolyBasis = basis(self.fullPoly)
        expandedInt = fullPolyBasis.expand(2)
        
        self.assertTrue( isinstance(expandedInt[0], float))
        self.assertTrue( isinstance(expandedInt[1], float))
        self.assertTrue( isinstance(expandedInt[2], float))
        self.assertTrue( isinstance(expandedInt[3], float))
        self.assertTrue( isinstance(expandedInt[4], float))

        expandedIntArray = fullPolyBasis.array_expand([2,2])
        
        self.assertTrue( isinstance(expandedIntArray[0], float))
        self.assertTrue( isinstance(expandedIntArray[1], float))
        self.assertTrue( isinstance(expandedIntArray[2], float))
        self.assertTrue( isinstance(expandedIntArray[3], float))
        self.assertTrue( isinstance(expandedIntArray[4], float))
        self.assertTrue( isinstance(expandedIntArray[5], float))
        self.assertTrue( isinstance(expandedIntArray[6], float))
        self.assertTrue( isinstance(expandedIntArray[7], float))
        self.assertTrue( isinstance(expandedIntArray[8], float))
        self.assertTrue( isinstance(expandedIntArray[9], float))

        
    def testTrivialPolynomialBasis(self):
        trivialBasis = basis(self.fullPoly[:1])

        self.assertTrue( ( np.array([1.0]) == trivialBasis.expand(1.0) ).all())
        self.assertTrue( ( np.array([1.0]) == trivialBasis.expand(2.1) ).all())

        self.assertTrue( ( np.array([1.0, 1.0]) == trivialBasis.expand([2.0, 3.0]) ).all())

    def testLinearPolynomialBasis(self):
        linearBasis = basis(self.fullPoly[:2])
        
        self.assertTrue( (np.array([1.0, 1.0]) == linearBasis.expand(1.0)).all() )
        self.assertTrue( (np.array([1.0, 5.0]) == linearBasis.expand(5.0)).all() )

        self.assertTrue( (np.array([1.0, 5.0, 1.0, 3.0]) == linearBasis.array_expand([5.0, 3.0]) ).all() )

if __name__ == '__main__':
    ut.main()
