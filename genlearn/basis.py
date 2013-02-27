import numpy as np

class basis:
    """
    A basis instance expands data according to a set of basis functions.

    example: degree 2 polynomial
    b = basis([lambda x: 1, lambda x: x, lambda x: x**2])
    
    b.expand(3)
    >>> np.array([1 3 9])

    b.expand_array(np.array([4 5]))
    >>> np.array([1 4 16 1 5 25])
    """
    def __init__(self, functions):
        self.functions = np.array(functions)
        self.vfapply = np.vectorize(lambda f,x: f(x))

    def __len__(self):
        return len(self.functions)

    def expand(self, term):
        return self.vfapply(self.functions, term)

    def array_expand(self, term_array):
        expanded_array = np.zeros([len(term_array), len(self)])
        for i,term in enumerate(term_array):
            expanded_array[i] = self.expand(term)
        return expanded_array.flatten()
