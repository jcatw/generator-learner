import numpy as np
import networkx as nx
import scikits.statsmodels.api as sm
import matplotlib.pyplot as plt
import ols
import os
from copy import deepcopy

import images2gif
from PIL import Image

#def Q_fn_gen(basis, weights):
#    return lambda x, basis=basis, weights=weights: (basis.expand(x) * weights).sum()

class animator:
    """
    animator instances extract frames from networkx graphs 
    to create animated gifs of network evolution.

    Requires PIL (Python Imaging Library)
    """
    def __init__(self, dirname, interval=10):
        

        self.dir = dirname
        os.mkdir(self.dir)

        self.interval = interval
        
        self.frame_count = 0 

        

    def add_frame(self, G):
        self.frame_count += 1
        nx.draw(G)
        plt.savefig("%s/frame%06d.gif" % (self.dir, self.frame_count))
        
    def animate(self):
        images = [Image.open("%s/frame%06d.gif" % (self.dir, i)) for i in range(1, self.frame_count)]
        images2gif.writeGif("%s/animation%06d.gif" % (self.dir, self.frame_count), images)
            
            
    
        
        

    
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

class episode:
    """
    class episode provides the machinery for parameterizing and
    running a single episode of the Q-learner.
    """
    def __init__(self, learner, alpha, gamma, epsilon):
        self.learner = learner
        self.G = None
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def run(self, draw_steps = False, animate = None): 
        """ 
        Run the Q-learner.  If draw_steps is true, draw and 
        show the learned graph at each iteration.

        Parameters
        ----------
        draw_steps : bool
          view a graph at each iteration?

        animate : None, string
          If string, create an animated gif of network evolution, 
          storing animation and frames in directory specified by animate.
        
        """ 
        
        if not animate is None:
            episode_animator = animator(animate)
        
        print self.learner.actions.action_dict 
        Q = 0.0
        action_taken = np.random.randint(0, len(self.learner.actions))
        reward = 0

        for i in xrange(self.learner.max_rows):

            if not i%100: print i
            #print i

            self.learner.actions.get(action_taken).execute(self.learner.G)


            if not animate is None and i % episode_animator.interval == 0:
                episode_animator.add_frame(self.learner.G)
                
            if draw_steps:
                #print "Action taken: %s" % (self.learner.actions.action_dict[action_taken],)
                nx.draw(self.learner.G)
                plt.show()
                #raw_input("Press Enter to continue")

            

            feature_values = self.learner.features.get(self.learner.G)
            self.learner.actions.get(action_taken).state.add_sample(feature_values, Q)

            #print self.learner.actions.get(action_taken).state.design_matrix[self.learner.actions.get(action_taken).state.n-1], Q

            if reward == 3: break
            
            reward = self.learner.reward_function(self.learner.G)

            #if self.learner.termination_function(reward): break
            #if self.learner.termination_function(G): break
            
            Q_values = self.learner.actions.Qs(feature_values)

            if np.random.rand() <= self.epsilon:
                action_taken = np.random.randint(0, len(self.learner.actions))
            else:
                action_taken = Q_values.argmax()

            Q = (1 - self.alpha) * Q + self.alpha * (reward + self.gamma * Q_values[action_taken])

        for i in xrange(len(self.learner.actions)):
            self.learner.actions.get(i).compute_Q_fn()

        #if not animate is None:
        #    episode_animator.animate()

        self.G = self.learner.G


class gglearner:
    """
    A gglearner instance represents a single Q-learner, including 
    the reward function, the features which make up the value function,
    the basis expansion of those features, and the actions the learner can take.

    """
    def __init__(self,
                 G0,
                 reward_function,
                 action_functions,
                 action_names,
                 basis_functions,
                 feature_functions,
                 # termination_function,
                 max_rows):

        self.G0 = G0
        self.G = deepcopy(G0)
        #self.Gs = [G0]
        self.episodes = []
        self.reward_function = reward_function
        self.basis = basis(basis_functions)
        self.actions = actions(action_functions,
                               action_names,
                               self.basis,
                               len(feature_functions),
                               max_rows)
        self.features = features(feature_functions)
        #self.termination_function = termination_function
        self.max_rows = max_rows

    def run_episode(self ,alpha, gamma, epsilon, draw_steps=False, animate=None):
        new_episode = episode(self, alpha, gamma, epsilon)
        new_episode.run(draw_steps, animate)
        self.episodes.append(new_episode)
        self.G = deepcopy(self.G0)


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

class state:
    """
    state instances track the relationship between the basis-expanded 
    features and the Q function.  An instance also provides the machinery 
    for learning (aka regressing) a value function from this data.
    """
    # note: each state builds up a design matrix and Q vector with max_rows
    # rows.  These are shrunk down to the number of instances actually populated
    # before regression occurs.  Rationale: numpy arrays like pre-allocation.
    def __init__(self, max_rows, n_features, basis):
        self.n = 0
        self.basis = basis
        self.design_matrix = np.zeros([max_rows, n_features * len(basis)])
        self.Q = np.zeros(max_rows)

    def add_sample(self, feature_values, Q):
        expanded_features = np.zeros([len(feature_values),len(self.basis)])
        for i,feature in enumerate(feature_values):
            expanded_features[i] = self.basis.expand(feature)

        self.design_matrix[self.n] = expanded_features.flatten()

        self.Q[self.n] = Q

        self.n += 1


    def regress(self):
        self.design_matrix = self.design_matrix[:self.n]
        self.Q = self.Q[:self.n]

        res = sm.OLS(self.Q,self.design_matrix)
        fit = res.fit()
        return fit.params
        #self.weights = res.weights
        #res = ols.ols(self.Q,self.design_matrix)
        #return res.b

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

class action:
    """
    action instances provide the means to both execute an action 
    and to compute that action's value function.
    """
    def __init__(self, f, name, basis, num_features, max_size):
        self.f = f
        self.basis = basis
        self.name = name
        self.num_features = num_features
        self.max_size = max_size
        self.state = state(max_size, num_features, basis)
        self.Q_function = Q_fn(self.basis, np.zeros(len(self.basis)*num_features))

    def execute(self, x):
        return self.f(x)

    def compute_Q_fn(self):
        weights = self.state.regress()
        self.Q_function = Q_fn(self.basis, weights)
        self.state = state(self.max_size, self.num_features, self.basis)

class actions:
    """
    An actions instance is a container for action instances.  
    It provides convenience functions and length semantics.
    """
    def __init__(self, function_iterable, names_iterable, basis, num_features, max_size=1000):
        self.actions_list = [action(f, name, basis, num_features, max_size) for f, name in zip(function_iterable, names_iterable)]
        self.action_dict = {}
        for i,an_action in enumerate(self.actions_list):
            self.action_dict[i] = an_action.name

        # self.vfapply = np.vectorize(lambda a,x: a.Q_function.eval(x))

    def __len__(self):
        return len(self.actions_list)

    def get(self, action_id):
        if isinstance(action_id,str):
            action_id = self.action_dict[action_id]
        return self.actions_list[action_id]

    def Qs(self, feature_values):
        #return self.vfapply(self.actions_list, feature_values)
        return np.array([a.Q_function.eval(feature_values) for a in self.actions_list])
