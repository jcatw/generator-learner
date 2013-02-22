import numpy as np
import numpy.random as rand
import networkx as nx
import scikits.statsmodels.api as sm
import matplotlib.pyplot as plt
import ols
import os
import logging
from copy import deepcopy

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

        self.iterations = 0
        

    #TODO: clean up method comments
    def run(self, n_iter, draw_steps = False, animate = None): 
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
        self.n_iter = n_iter
        self.actions_taken = -1 * np.ones(n_iter)
        
        if not animate is None:
            episode_animator = animator(animate)
        
        #print self.learner.actions.action_dict 
        logging.info("action dictionary: %s" % (self.learner.actions.action_dict,))
        #Q = 0.0
        action_A = np.random.randint(0, len(self.learner.actions))

        for i in xrange(n_iter):
            prev_q = self.learner.actions.get(action_A).q(self.learner.features.get(self.learner.G))
            #if not i%100: print i
            logging.debug("iteration %s" % (i,))
            #print i

            #if np.random.rand() <= self.epsilon:
            #    action_A = np.random.randint(0, len(self.learner.actions))
            #else:
            #    action_A = Q_values.argmax()

            self.learner.actions.get(action_A).execute(self.learner.G)

            reward = self.learner.reward_function(self.learner.G)

            if not animate is None and i % episode_animator.interval == 0:
                episode_animator.add_frame(self.learner.G)
                
            if draw_steps:
                #print "Action taken: %s" % (self.learner.actions.action_dict[action_A],)
                nx.draw(self.learner.G)
                plt.show()
                #raw_input("Press Enter to continue")

            if self.learner.termination_function(self.learner.G):
                
                break

            #new_feature_values = self.learner.features.get(self.learner.G)
            #self.learner.actions.get(action_A).state.add_sample(feature_values, Q)

            #print self.learner.actions.get(action_A).state.design_matrix[self.learner.actions.get(action_A).state.n-1], Q

            #if reward == 3: break
            
            

            #if self.learner.termination_function(reward): break
            #if self.learner.termination_function(G): break

            feature_vals = self.learner.features.get(self.learner.G)
            Q_values = self.learner.actions.Qs(feature_vals)

            if np.random.rand() <= self.epsilon:
                action_A = np.random.randint(0, len(self.learner.actions))
                #print "random action: %s" % (self.learner.actions.action_dict[action_A],)
                logging.debug("random action: %s" % (self.learner.actions.action_dict[action_A],))
            else:
                #action_A = Q_values.argmax()
                action_A = self.learner.actions.rand_max_Q_index(feature_vals)
                #print "optimal action: %s" % (self.learner.actions.action_dict[action_A],)
                logging.debug("optimal action: %s" % (self.learner.actions.action_dict[action_A],))
                
            self.actions_taken[i] = action_A
            #Q = (1 - self.alpha) * Q + self.alpha * (reward + self.gamma * Q_values[action_A])
            logging.debug("Q: %s" % (Q_values[action_A],))
            self.learner.actions.get(action_A).w += self.alpha * (reward + self.gamma * Q_values[action_A] - prev_q) * self.learner.basis.array_expand(feature_vals)
            #print self.learner.actions.get(action_A).w
            logging.debug("w: %s" % (self.learner.actions.get(action_A).w,))            

        #for i in xrange(len(self.learner.actions)):
        #    self.learner.actions.get(i).compute_Q_fn()

        #if not animate is None:
        #    episode_animator.animate()
        self.iterations = i
        self.actions_taken = self.actions_taken[:self.iterations]
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
                 termination_function):

                 #max_rows):
        logging.info("gglearner instance initialization")
        self.G0 = G0
        self.G = deepcopy(G0)
        #self.Gs = [G0]
        self.episodes = []
        self.reward_function = reward_function
        self.basis = basis(basis_functions)
        self.actions = actions(action_functions,
                               action_names,
                               self.basis,
                               len(feature_functions))
        self.features = features(feature_functions)
        self.termination_function = termination_function


    def run_episode(self, n_iter, alpha, gamma, epsilon, draw_steps=False, animate=None):
        logging.info("episode: %s" % (len(self.episodes) + 1,))
        
        logging.info("max iterations: %s" % (n_iter,))
        
        logging.info("alpha: %s" % (alpha,))
        logging.info("gamma: %s" % (gamma,))
        logging.info("epsilon: %s" % (epsilon,))
        
        new_episode = episode(self, alpha, gamma, epsilon)
        new_episode.run(n_iter, draw_steps, animate)

        logging.info("learned network: number of nodes = %s" % (new_episode.G.number_of_nodes(),) )
        logging.info("learned network: number of edges = %s" % (new_episode.G.size(),) )
        
        self.episodes.append(new_episode)
        self.G = deepcopy(self.G0)

    def dashboard(self,filename=None):
        fig = plt.figure()

        ax = fig.add_subplot(211)
        ax.plot(range(len(self.episodes)),[e.iterations for e in self.episodes])
        ax.set_title("Number of Iterations")

        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)
        
        #ax = fig.add_subplot(212)
        #ax.plot(range(len(self.episodes)),[e.iterations for e in self.episodes])
        #ax.plot(range(len(self.episodes)),[e.G.number_of_nodes() for e in self.episodes])
        #ax.set_title("Learned Nodes")
        
        


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

#TODO: remove state class
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
    def __init__(self, f, name, basis, num_features):
        self.f = f
        self.basis = basis
        self.name = name
        self.num_features = num_features
        self.w = np.zeros(len(self.basis)*num_features)
        #self.max_size = max_size
        #self.state = state(max_size, num_features, basis)
        #self.Q_function = Q_fn(self.basis, np.zeros(len(self.basis)*num_features))

    def execute(self, x):
        return self.f(x)

    def q(self, x):
        return (self.basis.array_expand(x) * self.w).sum()
    #def compute_Q_fn(self):
    #    weights = self.state.regress()
    #    self.Q_function = Q_fn(self.basis, weights)
    #    self.state = state(self.max_size, self.num_features, self.basis)

class actions:
    """
    An actions instance is a container for action instances.  
    It provides convenience functions and length semantics.
    """
    def __init__(self, function_iterable, names_iterable, basis, num_features):
        self.actions_list = [action(f, name, basis, num_features) for f, name in zip(function_iterable, names_iterable)]
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
        #return np.array([a.Q_function.eval(feature_values) for a in self.actions_list])
        return np.array([a.q(feature_values) for a in self.actions_list])

    def max_Q_indices(self, feature_values):
        these_Qs = self.Qs(feature_values)
        max_indices = [0]
        max_value = these_Qs[0]

        for i in range(1,len(these_Qs)):
            if these_Qs[i] > max_value:
                max_indices = [i]
                max_value = these_Qs[i]
            elif these_Qs[i] == max_value:
                max_indices.append(i)

        return max_indices

    def rand_max_Q_index(self, feature_values):
        max_indices = self.max_Q_indices(feature_values)

        return max_indices[rand.randint(len(max_indices))]
        
