import numpy as np
import networkx as nx
import logging
import matplotlib.pyplot as plt
from copy import deepcopy

import learner as gg
import util.rbf as rbf
from util.netfn import *

logging.basicConfig(filename="../plawLearner.log", level=logging.DEBUG)

target_indegree_exponent = -2.0
target_R2 = 0.9
target_num_nodes = 4100

initial_num_nodes = 3
initial_num_edges = 2

basis_functions = rbf.default_radial_basis
                   
class plaw_episode(gg.episode):
    def run(self, n_iter, draw_steps = False): 
        """ 
        Run the Q-learner.  If draw_steps is true, draw and 
        show the learned graph at each iteration.

        Parameters
        ----------
        n_iter: maximum number of iterations
        """ 
        self.n_iter = n_iter
        self.actions_taken = -1 * np.ones(n_iter)
        self.average_in_degree = -1 * np.ones(n_iter)
        self.number_of_nodes = -1 * np.ones(n_iter)
        self.number_of_edges = -1 * np.ones(n_iter)
        
        logging.info("action dictionary: %s" % (self.learner.actions.action_dict,))
        
        action_A = np.random.randint(0, len(self.learner.actions))

        for i in xrange(n_iter):
            prev_q = self.learner.actions.get(action_A).q(self.learner.features.get(self.learner.G))
            
            logging.debug("iteration %s" % (i,))
            
            self.learner.actions.get(action_A).execute(self.learner.G)

            reward = self.learner.reward_function(self.learner.G)

            if draw_steps:
                nx.draw(self.learner.G)
                plt.show()
                
            if self.learner.termination_function(self.learner.G):
                break
            
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
        
    def dashboard(self,filename=None):
        fig = plt.figure()
        
        # number of nodes
        ax = fig.add_subplot(231)
        ax.bar([0,1],[self.G.number_of_nodes(), target_num_nodes],color=['b','r'])
        ax.set_title("Learned Nodes")
    
        # iterations
        ax = fig.add_subplot(232)
        ax.bar([0,1],[self.iterations, self.n_iter],color=['b','r'])
        ax.set_title("Number of Iterations")
        
        # regression
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
    
        plt.suptitle("Actual: Blue, Target: Red")
    
        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)
        
            
class plaw_gglearner(gg.learner):
    def run_episode(self, n_iter, alpha, gamma, epsilon, draw_steps=False):
        logging.info("episode: %s" % (len(self.episodes) + 1,))
        
        logging.info("max iterations: %s" % (n_iter,))
        
        logging.info("alpha: %s" % (alpha,))
        logging.info("gamma: %s" % (gamma,))
        logging.info("epsilon: %s" % (epsilon,))
        
        new_episode = plaw_episode(self, alpha, gamma, epsilon)
        new_episode.run(n_iter, draw_steps)

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


glearn = plaw_gglearner(initial_graph(3,2),
                      lambda G: -1,
                      [add_node_random_edge, 
                       add_edge_random, 
                       add_edge_in_degree],
                      ["add node",
                       "add random edge", 
                       "add edge by in-degree"],
                      basis_functions,
                      [num_nodes, num_edges, average_in_degree],
                      termination_fn)

def scale_free_reward(G):
    exp_tol = 0.1
    exp, R2 = fit_powerlaw_regress(G)
    

    R2_condition = R2 >= target_R2
    exp_condition = exp >= target_indegree_exponent - exp_tol and exp <= target_indegree_exponent + exp_tol

    if exp_condition and R2_condition:
        return 1
    else:
        return 0



sfreelearn = plaw_gglearner(initial_graph(3,2),
                      scale_free_reward,
                      [add_node_random_edge,
                       add_edge_random, 
                       add_edge_in_degree],
                      ["add node",
                       "add random edge", 
                       "add edge by in-degree"],
                      basis_functions,
                      [num_nodes, num_edges, average_in_degree],
                      lambda G: False)



#glearn.run_episode(12000,0.00001,0.9,0.05)


