import numpy as np

class features:
    """
    A features instance represents the features which 
    comprise the value function.
    """
    def __init__(self, feature_functions):
        # seriously weird shit: np.vectorize treats class instances as
        # underlying dict, which breaks method calls.
        # Just using list comprehensions.

        #self.feature_functions = np.array(feature_functions)
        #self.vfapply = np.vectorize(lambda f,x: f(x))
        self.feature_functions = feature_functions

    def get(self, G):
        #self.vfapply(self.feature_functions, G)
        return np.array([f(G) for f in self.feature_functions])

