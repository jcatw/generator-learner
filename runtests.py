import sys, os
import unittest as ut

# set up path
sys.path.insert(0, os.path.dirname(__file__))

# import tests to perform
from genlearn.test.TestBasis import TestBasis
from genlearn.test.TestFeatures import TestFeatures

ut.main()
