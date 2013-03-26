"""
Run a scale-free learner.

arg1: number of iterations per episode
arg2: number of episodes
arg3: node addition period
arg4: node addition amount
arg5: target alpha (positive-valued)
arg6: log tag
"""

import sys
import logging
import numpy as np
from time import time

import genlearn.scalefree as scalefree

import genlearn.util.rbf as rbf
from genlearn.util.netfn import *

timestamp = time()

logging.basicConfig(filename="/nfs/pantanal/scratch1/jatwood/genlog/%s_%s_%s_%s_%s_%s.log" % (timestamp, 
                                                                                              sys.argv[6],
                                                                                              sys.argv[1],
                                                                                              sys.argv[2],
                                                                                              sys.argv[3],
                                                                                              sys.argv[4]), 
                                                                                              level=logging.DEBUG)

#logging.basicConfig(filename="testlog",level = logging.DEBUG)

def node_process_gen(frequency, n):
    def node_process(i, G):
        if i % frequency == 0:
            for k in range(n):
                index = G.number_of_nodes()
                target = np.random.randint(G.number_of_nodes())
                G.add_node(index)
                G.add_edge(index, target)

    return node_process

#def reward_gen(target_indegree_exponent, exp_tol, target_R2, false_reward, true_reward):
#    def reward_fn(G):
#        reg_res, ccdf, in_degree = fit_powerlaw_cumulative(G)
#
#        # target indegree exponent is for the pdf
#        # ccdf has exponent of (pdf exponent) + 1
#        # slope(log ccdf) - 1 = pdf exponent
#        slope = reg_res.params[0]
#        exp = slope - 1.0
#        
#        R2 = reg_res.rsquared
#        
#        R2_condition = R2 >= target_R2
#        exp_condition = np.abs(exp) >= np.abs(target_indegree_exponent) - exp_tol and np.abs(exp) <= np.abs(target_indegree_exponent) + exp_tol
#        
#        if exp_condition and R2_condition:
#            return true_reward
#        else:
#            return false_reward
#
#    return reward_fn
    
def reward_gen(target_alpha, alpha_tol, acceptable_D, false_reward, true_reward):
    def reward_fn(G):
        data = get_in_degree(G)
        results = powerlaw.Fit(data)

        if ((results.power_law.alpha > target_alpha - alpha_tol and 
            results.power_law.alpha < target_alpha + alpha_tol) and
            results.power_law.D <= acceptable_D):
            return true_reward
        else:
            return false_reward

    return reward_fn
    
G_init = initial_graph(100,200)
#reward_fn = reward_gen(-2.0, 0.1, 0.95, 0.0, 1.0) 
reward_fn = reward_gen(float(sys.argv[5]), 0.1, 0.2, 0.0, 1.0)
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
        
    agent.dashboard("/nfs/pantanal/scratch1/jatwood/genlog/%s_dashboard_%s_%s_%s_%s_%s.png" % (sys.argv[6],
                                                                                               timestamp,
                                                                                               sys.argv[1],
                                                                                               sys.argv[2],
                                                                                               sys.argv[3],
                                                                                               sys.argv[4]))
    agent.episodes[0].dashboard(-2.0, 0.95, "/nfs/pantanal/scratch1/jatwood/genlog/%s_first_episode_%s_%s_%s_%s_%s.png" % (sys.argv[6],
                                                                                                                           timestamp,
                                                                                                                           sys.argv[1],
                                                                                                                           sys.argv[2],
                                                                                                                           sys.argv[3],
                                                                                                                           sys.argv[4]))
    agent.episodes[-1].dashboard(-2.0, 0.95, "/nfs/pantanal/scratch1/jatwood/genlog/%s_last_episode_%s_%s_%s_%s_%s.png" % (sys.argv[6],
                                                                                                                                  timestamp,
                                                                                                                                  sys.argv[1],
                                                                                                                                  sys.argv[2],
                                                                                                                                  sys.argv[3],
                                                                                                                                  sys.argv[4]))

                                    
learn_process(int(sys.argv[3]), int(sys.argv[4]))
