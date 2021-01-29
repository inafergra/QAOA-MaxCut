# Private libraries
import qaoa_graphs as graphs
from statevector_functions import *
from classical_algos import *

# General
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Qiskit
from qiskit.providers.aer import StatevectorSimulator
from qiskit import QuantumCircuit, execute

# Optimizers
from scipy.optimize import minimize, differential_evolution

# Time
import time as t

#---------------------------------------end libraries------------------------------------------

#tic = t.process_time()

# Choose your fighter
G = graphs.sixn_prism()
n = len(G.nodes())
E = G.edges()

# Choose number of rounds
p_max = 4

# Generate plot of the Graph
colors = ['g' for node in G.nodes()]
nx.draw_networkx(G, node_color=colors)

#toc = t.process_time()
#print("It took " + str(toc - tic) + " seconds to generate graph")
#tic = t.process_time()

#circuit_ansatz(G, 'gamma', 'beta').draw(output = 'mpl') #draw the circuit

#toc = t.process_time()
#print("It took " + str(toc - tic) + " seconds to generate the grid")
#tic = t.process_time()

# setting the bounds for gamma and beta (only for SLQP)
#bounds = ((0, np.pi), (0, 2*np.pi))

costs=[]
for p in range(1,p_max+1):

    bound = (0, 2*np.pi)
    bounds = []
    for i in range(2*p):
        bounds.append(bound)

    # Nelder-Mead optimizer:
    # max_expect_value = minimize(cost_function, x0=np.random.randn(2,p), args=(G,p), options={'disp': True}, method = 'Nelder-Mead')

    # Differential evolution optimizer:
    max_expect_value = differential_evolution(cost_function,args=(G,p), bounds=bounds)

    #toc = t.process_time()
    #print("It took " + str(toc - tic) + " seconds to optimize")
    #tic = t.process_time()

    optimal_gamma, optimal_beta = max_expect_value['x'][:p], max_expect_value['x'][p:]
    state_dictionary = execute_circuit(G,optimal_gamma,optimal_beta,p=p)
    solution, solution_cost = get_solution(state_dictionary, G)
    cost = -max_expect_value.get('fun')
    costs.append(cost)

    print(f'Number of layers p = {p}')
    print('Expectation value of the cost function = ', -max_expect_value.get('fun'))
    print('Approximation ratio = ', -max_expect_value.get('fun')/solution_cost )
    print("Optimal gamma and beta are :(", optimal_gamma, ", ", optimal_beta, ")")
    print(f'Cost list : {costs}')
    show_amplitudes(G, optimal_gamma, optimal_beta, p = p)

greedy_sol = greedy_solution(G)
greedy_sol_cost = cost_function_C(greedy_sol, G)
print(f'The greedy solution gives cost: {greedy_sol_cost}')