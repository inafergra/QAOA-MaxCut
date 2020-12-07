
import networkx as nx

import matplotlib.pyplot as plt 
import numpy as np

from qiskit import Aer
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute
from qiskit.tools.monitor       import job_monitor
from qiskit.visualization import plot_histogram

from scipy.optimize import minimize, LinearConstraint     

# Generating the graph
n = 4
V = np.arange(0,n,1)
E =[(0,1,1.0), (1,2,1.0), (2,3,1.0), (3,0,1.0)] 

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

def quantum_circuit(G, gamma, beta, shots): #returns an instance of the Results class

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

    backend = Aer.get_backend("qasm_simulator")
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

#we need a float objective function to minimize that takes gamma and beta and return the expectation value:
def expect_value_function(parameters,G):
    gamma = parameters[0]
    beta = parameters[1]
    shots = 1000
    counts = quantum_circuit(G,gamma,beta,shots)
    avr_cost = get_expect_value(counts, shots)
    return avr_cost

#Optimizing the solution
bounds = ((0, np.pi), (0, 2*np.pi))

max_expect_value = minimize(expect_value_function, x0=np.random.randn(2),args=(G), bounds=bounds, options={'disp': False}, method = 'SLSQP')
optimal_gamma, optimal_beta = max_expect_value['x']

print(optimal_gamma,optimal_beta)
print(max_expect_value)

counts = quantum_circuit(G,optimal_gamma,optimal_beta,100)
fig = plot_histogram(counts,figsize = (8,6),bar_labels = False)
plt.show()