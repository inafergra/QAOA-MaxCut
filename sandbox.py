import qaoa_graphs as graphs
import networkx as nx
import matplotlib.pyplot as plt

G = graphs.fournodes_3reg_graph()
n = len(G.nodes())
E = G.edges()

colors = ['g' for node in G.nodes()]
nx.draw_networkx(G, node_color=colors)
plt.show()
