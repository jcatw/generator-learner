import sys
import logging
from time import time

import genlearn.scalefree as scalefree

import genlearn.util.rbf as rbf
from genlearn.util.netfn import *

timestamp = time()

logging.basicConfig(filename="/nfs/pantanal/scratch1/jatwood/genlog/scalefree_%s.log" % (timestamp), level=logging.DEBUG)

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
for k in xrange(int(sys.argv[2])):
    print "Running episode %s." % (k+1,)
    agent.run_episode(int(sys.argv[1]), node_process, 0.0001, 0.9, 0.5)
    
agent.dashboard("/nfs/pantanal/scratch1/jatwood/genlog/scalefree_dashboard_%s.png" % (timestamp,))
agent.episodes[0].dashboard("/nfs/pantanal/scratch1/jatwood/genlog/scalefree_first_episode_%s.png" % (timestamp,))
agent.episodes[-1].dashboard("/nfs/pantanal/scratch1/jatwood/genlog/scalefree_last_episode_%s.png" % (timestamp,))

                                    
