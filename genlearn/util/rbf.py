import numpy as np
import numpy.random as rand

def rbf_gen(center,sigma):
    return lambda a, c = np.array(center), s = sigma: np.exp(-np.square(a-c).sum()/(2.0 * sigma**2))

default_rbf_range = [0.0, 5.0]
default_radial_basis = [rbf_gen([0.,0.,0.],3.0),
                        rbf_gen(rand.randint(default_rbf_range[0],default_rbf_range[1],3.0),3.0),
                        rbf_gen(rand.randint(default_rbf_range[0],default_rbf_range[1],3.0),3.0),
                        rbf_gen(rand.randint(default_rbf_range[0],default_rbf_range[1],3.0),3.0),
                        rbf_gen(rand.randint(default_rbf_range[0],default_rbf_range[1],3.0),3.0),
                        rbf_gen(rand.randint(default_rbf_range[0],default_rbf_range[1],1.0),3.0),
                        rbf_gen(rand.randint(default_rbf_range[0],default_rbf_range[1],1.0),3.0),
                        rbf_gen(rand.randint(default_rbf_range[0],default_rbf_range[1],1.0),3.0),
                        rbf_gen(rand.randint(default_rbf_range[0],default_rbf_range[1],1.0),3.0),
                        rbf_gen(rand.randint(default_rbf_range[0],default_rbf_range[1],5.0),3.0),
                        rbf_gen(rand.randint(default_rbf_range[0],default_rbf_range[1],5.0),3.0),
                        rbf_gen(rand.randint(default_rbf_range[0],default_rbf_range[1],5.0),3.0),
                        rbf_gen(rand.randint(default_rbf_range[0],default_rbf_range[1],5.0),3.0)]
