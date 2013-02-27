class Q_fn:
    """
    This class maps feature values to Q values.  
    A Q_fn instance is a parameterized Q function 
    which takes care of feature expansion under the hood.  
    """
    def __init__(self, basis, weights):
        self.basis = basis
        self.weights = weights

    def eval(self, x):
        return (self.basis.array_expand(x) * self.weights).sum()

