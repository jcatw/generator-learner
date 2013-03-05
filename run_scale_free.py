import logging
from time import time

import genlearn.scalefree as scalefree

import genlearn.util.rbf as rbf
from genlearn.util.netfn import *

logging.basicConfig(filename="scalefree_%s.log" % (time()), level=logging.DEBUG)

def node_process(G):
    index = G.number_of_nodes()
    G.add_node(index)
    

G_init = initial_graph(3,2)
reward_fn = lambda G: -1
action_fns = [add_edge_random,
              add_edge_in_degree]
action_names = ["add random edge",
                "add edge by in-degree"]
basis_fns = rbf.default_radial_basis
feature_fns = [num_nodes, num_edges, average_in_degree]
termination_fn = lambda G: False

agent = scalefree.scalefree_learner(G_init,
                                    reward_fn,
                                    action_fns,
                                    action_names,
                                    basis_fns,
                                    feature_fns,
                                    termination_fn)

# to run an episode: agent.run_episode([number of iterations], [node process], [alpha], [gamma], [epsilon])
# agent.run_episode(30000, node_process, 0.0001, 0.9, 0.5)


                                    
