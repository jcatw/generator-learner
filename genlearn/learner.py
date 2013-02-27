import numpy as np
import matplotlib.pyplot as plt
import logging
from copy import deepcopy

from actions  import actions
from features import features
from episode  import episode
from basis    import basis


class learner:
    """
    A learner instance represents a generic network learner, including 
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
        logging.info("learner instance initialization")
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
        
