import numpy as np
import networkx as nx
import logging
import matplotlib.pyplot as plt
from copy import deepcopy
from time import time

import learner
import util.rbf as rbf
from util.netfn import *

class scalefree_episode(learner.episode):
    def run(self, n_iter, node_process):
        self.n_iter = n_iter
        self.actions_taken = -1 * np.ones(n_iter)
        self.average_in_degree = -1 * np.ones(n_iter)
        self.number_of_nodes = -1 * np.ones(n_iter)
        self.number_of_edges = -1 * np.ones(n_iter)
        
        logging.info("action dictionary: %s" % (self.learner.actions.action_dict,))
        
        action_A = np.random.randint(0, len(self.learner.actions))

        for i in xrange(n_iter):
            logging.debug("iteration %s" % (i,))
            
            prev_q = self.learner.actions.get(action_A).q(self.learner.features.get(self.learner.G))

            self.learner.actions.get(action_A).execute(self.learner.G)

            reward = self.learner.reward_function(self.learner.G)

            if self.learner.termination_function(self.learner.G):
                break

            node_process(i, self.learner.G)

            feature_vals = self.learner.features.get(self.learner.G)
            Q_values = self.learner.actions.Qs(feature_vals)

            if np.random.rand() <= self.epsilon:
                action_A = np.random.randint(0, len(self.learner.actions))
                
                logging.debug("random action: %s" % (self.learner.actions.action_dict[action_A],))
            else:
                action_A = self.learner.actions.rand_max_Q_index(feature_vals)
                
                logging.debug("optimal action: %s" % (self.learner.actions.action_dict[action_A],))
                
            self.actions_taken[i] = action_A
            self.average_in_degree[i] = np.mean(self.learner.G.in_degree().values())
            self.number_of_nodes[i] = self.learner.G.number_of_nodes()
            self.number_of_edges[i] = self.learner.G.size()
            
            logging.debug("Q: %s" % (Q_values[action_A],))
            self.learner.actions.get(action_A).w += self.alpha * (reward + self.gamma * Q_values[action_A] - prev_q) * self.learner.basis.array_expand(feature_vals)
            
            logging.debug("w: %s" % (self.learner.actions.get(action_A).w,))            

        

        self.iterations = i
        
        self.actions_taken = self.actions_taken[:self.iterations]
        self.average_in_degree = self.average_in_degree[:self.iterations]
        self.number_of_nodes = self.number_of_nodes[:self.iterations]
        self.number_of_edges = self.number_of_edges[:self.iterations]
        
        self.G = self.learner.G

    def dashboard(self,target_indegree_exponent, target_R2, filename=None):
        fig = plt.figure()
        
        # number of nodes
        ax = fig.add_subplot(231)
        ax.bar([0],[self.G.number_of_nodes()],color=['b'])
        ax.set_title("Learned Nodes")
    
        # iterations
        ax = fig.add_subplot(232)
        ax.bar([0,1],[self.iterations, self.n_iter],color=['b','r'])
        ax.set_title("Number of Iterations")
        
        # regression
        """
        exponent, R2 = fit_powerlaw_regress(self.G)
        
        ax = fig.add_subplot(233)
        ax.bar([0,1],[exponent, target_indegree_exponent],color=['b','r'])
        ax.set_title("Exponent")
    
        ax = fig.add_subplot(234)
        ax.bar([0,1],[R2,target_R2],color=['b','r'])
        ax.set_title("R2")

        m, b = fit_powerlaw(self.G)

        x = np.arange(0.,10.,0.1)
        ax = fig.add_subplot(235)
        ax.plot(x, m * x + b, 'b', x, target_indegree_exponent * x + b, 'r')
        ax.set_title("Log-Log In-Degree Frequency")
        """

        reg_res, ccdf, in_degree= fit_powerlaw_cumulative(self.G)
        slope = reg_res.params[0]
        intercept = reg_res.params[0]

        exp = slope - 1.0

        R2 = reg_res.rsquared

        ax = fig.add_subplot(233)
        ax.bar([0,1], [exp, target_indegree_exponent], color=['b','r'])
        ax.set_title("alpha")

        ax = fig.add_subplot(234)
        ax.bar([0,1], [R2, target_R2], color=['b','r'])
        ax.set_title("R2")

        ax = fig.add_subplot(235)
        x = np.arange(in_degree[0], in_degree[-1], 1.0 / float(len(in_degree)))
        ax.plot(np.log10(in_degree+1), np.log10(ccdf+1), 'g',
                x, slope * x + intercept, 'b',
                x, (target_indegree_exponent + 1) * x + intercept, 'r')
        ax.set_title("In-Degree CCDF")

        plt.suptitle("Actual: Blue, Target: Red")
    
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)

class scalefree_learner(learner.learner):
    def run_episode(self, n_iter, node_process, alpha, gamma, epsilon):
        logging.info("episode: %s" % (len(self.episodes) + 1,))
        
        logging.info("max iterations: %s" % (n_iter,))
        
        logging.info("alpha: %s" % (alpha,))
        logging.info("gamma: %s" % (gamma,))
        logging.info("epsilon: %s" % (epsilon,))
        
        new_episode = scalefree_episode(self, alpha, gamma, epsilon)
        new_episode.run(n_iter, node_process)

        logging.info("learned network: number of nodes = %s" % (new_episode.G.number_of_nodes(),) )
        logging.info("learned network: number of edges = %s" % (new_episode.G.size(),) )
        
        self.episodes.append(new_episode)
        self.G = deepcopy(self.G0)


    def dashboard(self,filename=None):
        fig = plt.figure()

        ax = fig.add_subplot(221)
        ax.plot(range(len(self.episodes)),[e.iterations for e in self.episodes])
        ax.set_title("Number of Iterations")

        ax = fig.add_subplot(222)
        ax.plot(range(len(self.episodes)),[e.average_in_degree[-1] for e in self.episodes])
        ax.set_title("Average In Degree")
        
        ax = fig.add_subplot(223)
        ax.plot(range(len(self.episodes)),[e.G.number_of_nodes() for e in self.episodes])
        ax.set_title("Learned Nodes")

        ax = fig.add_subplot(224)
        ax.plot(range(len(self.episodes)),[e.G.size() for e in self.episodes])
        ax.set_title("Learned Edges")

        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)

    def write_csv(self,filename):
        csv = open(filename,'w')

        csv.writeline(["episode, iterations, nodes, edges, avg_in_degree"])

        for i,e in enumerate(self.episodes):
            iterations = str(e.iterations)
            n_nodes = str(e.G.number_of_nodes())
            n_edges = str(e.G.number_of_edges())
            avg_in_degree = str(e.avg_in_degree[-1])

            csv.write(','.join([str(i),iterations,n_nodes,n_edges,avg_in_degree]))
            csv.write('\n')

        csv.close()
