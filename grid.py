
import networkx as nx

import matplotlib.pyplot as plt 
import numpy as np

from qiskit import Aer
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute
from qiskit.tools.monitor       import job_monitor
from qiskit.visualization import plot_histogram

from scipy.optimize import minimize     

# Generating the graph with 2 nodes 
n = 2
V = np.arange(0,n,1)
E =[(0,1,1.0)] 

G = nx.Graph()
G.add_nodes_from(V)
G.add_weighted_edges_from(E)

nx.draw_networkx(G, node_size=600)

def cost_function_C(x,G):

    E = G.edges()
    C = 0
    for index in E:
        e1 = index[0]
        e2 = index[1]
        w      = G[e1][e2]['weight']
        C = C + w*x[e1]*(1-x[e2]) + w*x[e2]*(1-x[e1])

    return C

def quantum_circuit(G, gamma, beta, shots):

    n = len(G.nodes())

    QAOA = QuantumCircuit(n, n)
    QAOA.h(range(n))
    for edge in E:
        k = edge[0]
        l = edge[1]
        QAOA.cu1(-2*gamma, k, l)
        QAOA.u1(gamma, k)
        QAOA.u1(gamma, l)
        
    QAOA.rx(2*beta, range(n))
    QAOA.measure(range(n),range(n))
    
    job = execute(QAOA, backend=backend, shots=shots)
    results = job.result()
    counts = results.get_counts() #dictionary with keys 'bit string x' and items 'counts of x'

    return counts

def get_expect_value(counts, shots):
    total_cost  = 0
    for state in list(counts.keys()): 
        x = [int(bit_num) for bit_num in list(state)] #the bit string is saved as a list
        cost_x = cost_function_C(x,G)
        total_cost += counts[state]*cost_x
    avr_cost = total_cost/shots
    return avr_cost

backend = Aer.get_backend("qasm_simulator")
shots = 100
avr_cost_grid = np.zeros((50,50))

#Making a solution grid-------------------------------------------------------------------

for i in np.arange(0, 10, 0.2): #gamma and beta grid
    for j in np.arange(0, 10, 0.2):

        gamma = i
        beta = j

        counts = quantum_circuit(G,gamma,beta,shots)
        avr_cost = get_expect_value(counts, shots)

        avr_cost_grid[int(i*5),int(j*5)] = avr_cost

plt.imshow(avr_cost_grid) #alpha->x axis, beta->y axis #gamma is periodic in (0,pi) and beta in (0,2pi)
plt.title('Parameter grid', fontsize=8)
plt.colorbar()
plt.show()
