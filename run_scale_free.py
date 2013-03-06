import sys
import logging
import numpy as np
from time import time

import genlearn.scalefree as scalefree

import genlearn.util.rbf as rbf
from genlearn.util.netfn import *

timestamp = time()

logging.basicConfig(filename="/nfs/pantanal/scratch1/jatwood/genlog/scalefree_%s_%s_%s_%s_%s.log" % (timestamp, 
                                                                                                     sys.argv[1],
                                                                                                     sys.argv[2],
                                                                                                     sys.argv[3],
                                                                                                     sys.argv[4]), level=logging.DEBUG)

def node_process_gen(frequency, n):
    def node_process(i, G):
        if i % frequency == 0:
            for k in range(n):
                index = G.number_of_nodes()
                target = np.random.randint(G.number_of_nodes())
                G.add_node(index)
                G.add_edge(index, target)

    return node_process

def reward_gen(target_indegree_exponent, exp_tol, target_R2, false_reward, true_reward):
    def reward_fn(G):
        exp, R2 = fit_powerlaw_regress(G)
        
        R2_condition = R2 >= target_R2
        exp_condition = np.abs(exp) >= np.abs(target_indegree_exponent) - exp_tol and np.abs(exp) <= np.abs(target_indegree_exponent) + exp_tol
        
        if exp_condition and R2_condition:
            return true_reward
        else:
            return false_reward

    return reward_fn
    

G_init = initial_graph(3,2)
reward_fn = reward_gen(-2.0, 0.1, 0.95, 0.0, 1.0) 
action_fns = [add_edge_random,
              add_edge_in_degree]
action_names = ["add random edge",
                "add edge by in-degree"]
basis_fns = rbf.default_radial_basis
feature_fns = [num_nodes, num_edges, average_in_degree]
termination_fn = lambda G: False




def learn_process(frequency, n):
    agent = scalefree.scalefree_learner(G_init,
                                        reward_fn,
                                        action_fns,
                                        action_names,
                                        basis_fns,
                                        feature_fns,
                                        termination_fn)
    
    node_process = node_process_gen(frequency, n)
    
    # to run an episode: agent.run_episode([number of iterations], [node process], [alpha], [gamma], [epsilon])
    for k in xrange(int(sys.argv[2])):
        print "Running episode %s." % (k+1,)
        agent.run_episode(int(sys.argv[1]), node_process, 0.0001, 0.9, 0.5)
        
    agent.dashboard("/nfs/pantanal/scratch1/jatwood/genlog/scalefree_dashboard_%s_%s_%s_%s_%s.png" % (timestamp,
                                                                                                      sys.argv[1],
                                                                                                      sys.argv[2],
                                                                                                      sys.argv[3],
                                                                                                      sys.argv[4]))
    agent.episodes[0].dashboard(-2.0, 0.95, "/nfs/pantanal/scratch1/jatwood/genlog/scalefree_first_episode_%s_%s_%s_%s_%s.png" % (timestamp,
                                                                                                                                  sys.argv[1],
                                                                                                                                  sys.argv[2],
                                                                                                                                  sys.argv[3],
                                                                                                                                  sys.argv[4]))
    agent.episodes[-1].dashboard(-2.0, 0.95, "/nfs/pantanal/scratch1/jatwood/genlog/scalefree_last_episode_%s_%s_%s_%s_%s.png" % (timestamp,
                                                                                                                                  sys.argv[1],
                                                                                                                                  sys.argv[2],
                                                                                                                                  sys.argv[3],
                                                                                                                                  sys.argv[4]))

                                    
learn_process(int(sys.argv[3]), int(sys.argv[4]))
