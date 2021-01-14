import networkx as nx
import numpy as np

def starmon_graph(): #Starmon 5 shape
    n     = 5
    V     = np.arange(0,n,1)
    E     =[(0,1,1.0),(0,2,1.0),(0,3,1.0),(4,0,1.0)]
    G     = nx.Graph()
    G.add_nodes_from(V)
    G.add_weighted_edges_from(E)
    return G

def triangle():
    n     = 3
    V     = np.arange(0,n,1)
    E     =[(0,1,1.0),(1,2,1.0),(2,0,1.0)]
    G     = nx.Graph()
    G.add_nodes_from(V)
    G.add_weighted_edges_from(E)
    return G

def two_nodes_graph():
    n     = 2
    V     = np.arange(0,n,1)
    E     =[(0,1,1.0)]
    G     = nx.Graph()
    G.add_nodes_from(V)
    G.add_weighted_edges_from(E)
    return G

def fournodes_3reg_graph(): #4-node 3-regular yutsis graph
    n     = 4
    V     = np.arange(0,n,1)
    E     =[(0,1,1.0), (1,2,1.0), (2,3,1.0), (3,0,1.0), (0,2,1.0), (1,3,1.0)]
    G     = nx.Graph()
    G.add_nodes_from(V)
    G.add_weighted_edges_from(E)
    return G

def weighted_fournodes_graph(): #4-node 3-regular yutsis graph
    n     = 4
    V     = np.arange(0,n,1)
    E     =[(0,1,2.0), (1,2,1.0), (2,3,1.0), (3,0,1.0), (0,2,1.0), (1,3,1.0)]
    G     = nx.Graph()
    G.add_nodes_from(V)
    G.add_weighted_edges_from(E)
    return G

def sixn_prism():
    n     = 6
    V     = np.arange(0,n,1)
    E     = [(0,1,1.0),(0,5,1.0),(0,4,1.0),(1,2,1.0),(1,5,1.0),(2,3,1.0),(2,4,1.0),(3,4,1.0),(3,5,1.0)]
    G     = nx.Graph()
    G.add_nodes_from(V)
    G.add_weighted_edges_from(E)
    return G

def erdos_renyi(): #Returns a Gn,p random graph, also known as an Erdős-Rényi graph or a binomial graph.

    n = 9 # number of nodes
    p = 0.5 # probability of creating an edge
    G = nx.fast_gnp_random_graph(n, p, seed=1)
    for e in G.edges():
        G[e[0]][e[1]] ['weight'] = 1.0
    return G

if __name__ == "__main__":
    #Generate plot of the Graph
    import matplotlib.pyplot as plt
    G = triangle()
    colors = ['g' for node in G.nodes()]
    nx.draw_networkx(G, node_color=colors)
    #plt.savefig('4 node regular graph')
    plt.show()
