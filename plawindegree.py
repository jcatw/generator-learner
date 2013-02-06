import gglearn as gg
import numpy as np
import numpy.random as rand
from scipy import stats
import networkx as nx
import scikits.statsmodels.api as sm
#import ols
import logging
import matplotlib.pyplot as plt

logging.basicConfig(filename="gglearn.log", level=logging.DEBUG)

target_indegree_exponent = -2.7
target_R2 = 0.8
target_num_nodes = 4000

initial_num_nodes = 3
initial_num_edges = 2

#basis_functions = [lambda x: 1,
#                   lambda x: x,
#                   lambda x: x**2,
#                   lambda x: x**3,
#                   lambda x: x**4,
#                   lambda x: x**5]

#def reward_function_gen(target_exponent, target_num_nodes):
#    # no multiline lambdas => running regress once for
#    # each parameter.  Ewwwww.
#    return lambda G, target_exponent=target_exponent, target_num_nodes=target_num_nodes: fit_powerlaw_regress(G)[1] / (((np.abs(target_exponent - fit_powerlaw_regress(G)[0]) + 1) * np.log10(np.abs(G.number_of_nodes() - target_num_nodes) + 1))+1)

#def reward_fn(G,termination_fn):
#    if termination_fn(G):
#        return 0
#    else:
#        return -1
#

#def termination_function(reward):
#    return reward >= 0.9 and reward <= 1.1

def termination_fn(G, exp_tol=0.25, R2_tol=0.1, n_node_tol=10):
    exp, R2 = fit_powerlaw_regress(G)
    R2_condition =  (R2 >= target_R2 - R2_tol)

    exp_condition = (np.abs(exp) >= np.abs(target_indegree_exponent) - exp_tol) and (np.abs(exp) <= np.abs(target_indegree_exponent) + exp_tol)

    num_nodes = G.number_of_nodes()
    node_condition = (num_nodes >= target_num_nodes - n_node_tol) and (num_nodes <= target_num_nodes + n_node_tol)
    #print R2, exp, num_nodes
    #print R2_condition, exp_condition, node_condition

    #return int(R2_condition) + int(exp_condition) + int(node_condition)
    #if (R2_condition and exp_condition and node_condition):
    #    return 0
    #else:
    #    return -1
    
    return (R2_condition and exp_condition and node_condition)

#------------------------------------------------------------------------------#
# Utility Functions
#------------------------------------------------------------------------------#
def sample_pmf(pmf):
    uniform_draw = rand.rand()
    cmf = np.array(pmf).cumsum()
    return np.array(range(len(cmf)))[uniform_draw < cmf][0]

def get_in_degree(G):
    return np.array(G.in_degree().values())

def fit_powerlaw_MLE(G):
    return stats.powerlaw.fit(get_in_degree(G))

def fit_powerlaw_regress(G):
    in_degree = get_in_degree(G)
    n, bins = np.histogram(in_degree)

    bins_midpoint = np.zeros(len(n))
    for i in xrange(len(n)):
        bins_midpoint[i] = (bins[i] + bins[i+1]) / 2.0

    #reg_res = ols.ols(np.log(n+1), np.log(bins_midpoint))
    reg_res = sm.OLS(np.log(n+1), np.array(sm.add_constant(np.log(bins_midpoint)))).fit()

    #return reg_res.b[1],reg_res.R2
    return reg_res.params[0], reg_res.rsquared

#------------------------------------------------------------------------------#
# Initialization Functions
#------------------------------------------------------------------------------#
def initial_graph(n_nodes, n_edges):
    G = nx.DiGraph()
    G.node_holes = [] #stack that tracks holes in sequential node numbering

    for i in xrange(n_nodes):
        G.add_node(G.number_of_nodes())

    for i in xrange(n_edges):
        edge_from = rand.randint(0, G.number_of_nodes())
        edge_to = rand.randint(0, G.number_of_nodes())
        G.add_edge(edge_from, edge_to)

    return G

#------------------------------------------------------------------------------#
# Action Functions
#------------------------------------------------------------------------------#
def sample_node(G):
    nodes = G.nodes()
    return nodes[rand.randint(0, len(nodes))]

def add_node(G):
    if G.node_holes:
        G.add_node(G.node_holes.pop())
    else:
        G.add_node(G.number_of_nodes())

def add_node_random_edge(G):

    if G.node_holes:
        node_label = G.node_holes.pop()
    else:
        node_label = G.number_of_nodes()

    random_existing_node = sample_node(G)
    G.add_node(node_label)
    G.add_edge(node_label, random_existing_node)

def delete_node_random(G):
    if G.number_of_nodes() > target_num_nodes:
        #node_index = rand.randint(G.number_of_nodes())
        #node_label = G.nodes()[node_index]
        node_label = sample_node(G)
        G.node_holes.append(node_label)
        G.remove_node(node_label)


def delete_node_in_degree(G):
    if G.number_of_nodes() > 1:
        in_degree = np.array(G.in_degree().values())
        if in_degree.max() > 0:
            pmf = in_degree.astype(float) / in_degree.sum()

            node_index = sample_pmf(pmf)
            node_label = G.nodes()[node_index]

            G.node_holes.append(node_label)
            G.remove_node(node_label)

def delete_node_in_degree_inverse(G):
    if G.number_of_nodes() >= target_num_nodes:
        inv_in_degree = 1 / (np.array(G.in_degree().values()) + 1)
        pmf = inv_in_degree.astype(float) / inv_in_degree.sum()

        #node_index = sample_pmf(pmf)
        #node_label = G.nodes()[node_index]
        node_label = sample_node(G)

        G.node_holes.append(node_label)
        G.remove_node(node_label)

def add_edge_random(G):
    edge_from_index = rand.randint(0,G.number_of_nodes())
    edge_from_label = G.nodes()[edge_from_index]

    edge_to_index   = rand.randint(0,G.number_of_nodes())
    edge_to_label   = G.nodes()[edge_to_index]
    G.add_edge(edge_from_label, edge_to_label)

def add_edge_in_degree(G):
    in_degree = np.array(G.in_degree().values())

    if in_degree.max() > 0 :
        edge_from_index = rand.randint(0,G.number_of_nodes())
        edge_from_label = G.nodes()[edge_from_index]

        pmf = in_degree.astype(float) / in_degree.sum()

        edge_to_index = sample_pmf(pmf)
        edge_to_label = G.nodes()[edge_to_index]

        G.add_edge(edge_to_label, edge_from_label)
    else:
        add_random_edge(G)

def delete_edge_random(G):
    if G.number_of_edges() > 0:
        edge_index = rand.randint(0, G.number_of_edges())
        G.remove_edge(*G.edges()[edge_index])

#------------------------------------------------------------------------------#
# Feature Functions
#------------------------------------------------------------------------------#
def num_nodes(G):
    return float(G.number_of_nodes()) / 1000.0

def num_edges(G):
    return float(G.size()) / 1000.0

def diameter(G):
    return nx.diameter(G)

def average_clustering(G):
    return nx.average_clustering(G)

def average_in_degree(G):
    return np.array(G.in_degree().values()).mean()

def powerlaw_mle(G):
    return fit_powerlaw_regress(G)[0]

#def average_out_degree(G):
#    return np.array(G.out_degree().values()).mean()

def rbf_gen(center,sigma):
    return lambda a, c = np.array(center), s = sigma: np.exp(-np.square(a-c).sum()/(2.0 * sigma**2))



rbf_range = [0.0,5.0]
    
basis_functions = [rbf_gen([0.,0.,0.],3.0),
                   rbf_gen(rand.randint(rbf_range[0],rbf_range[1],3.0),3.0),
                   rbf_gen(rand.randint(rbf_range[0],rbf_range[1],3.0),3.0),
                   rbf_gen(rand.randint(rbf_range[0],rbf_range[1],3.0),3.0),
                   rbf_gen(rand.randint(rbf_range[0],rbf_range[1],3.0),3.0),
                   rbf_gen(rand.randint(rbf_range[0],rbf_range[1],1.0),3.0),
                   rbf_gen(rand.randint(rbf_range[0],rbf_range[1],1.0),3.0),
                   rbf_gen(rand.randint(rbf_range[0],rbf_range[1],1.0),3.0),
                   rbf_gen(rand.randint(rbf_range[0],rbf_range[1],1.0),3.0),
                   rbf_gen(rand.randint(rbf_range[0],rbf_range[1],5.0),3.0),
                   rbf_gen(rand.randint(rbf_range[0],rbf_range[1],5.0),3.0),
                   rbf_gen(rand.randint(rbf_range[0],rbf_range[1],5.0),3.0),
                   rbf_gen(rand.randint(rbf_range[0],rbf_range[1],5.0),3.0)]
                   

glearn = gg.gglearner(initial_graph(3,2),
                      # reward_function_gen(target_indegree_exponent, target_num_nodes),
                      lambda G: -1,
                      [add_node_random_edge, 
                       delete_node_in_degree_inverse,
                       add_edge_random, 
                       add_edge_in_degree],
                      ["add node",
                       "delete node by inverse in-degree",
                       "add random edge", 
                       "add edge by in-degree"],
                      basis_functions,
                      [num_nodes, num_edges, average_in_degree],#, powerlaw_mle],
                      termination_fn)


#glearn.run_episode(12000,0.00001,0.9,0.05)

