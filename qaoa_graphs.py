import networkx as nx
import numpy as np

def diamond_graph(): #Starmon 5 shape
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

