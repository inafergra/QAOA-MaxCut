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
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute

#Optimizers
from scipy.optimize import minimize, differential_evolution

# Time
import time as t

#---------------------------------------end libraries---------------------------------------

tic = t.process_time()

# Choose your fighter
G = graphs.fournodes_3reg_graph()
n = len(G.nodes())
E = G.edges()

# Choose the arena
backend = StatevectorSimulator()

# Choose number of rounds
p = 2 

# Generate plot of the Graph
colors = ['g' for node in G.nodes()]
nx.draw_networkx(G, node_color=colors)

toc = t.process_time()
print("It took " + str(toc - tic) + " seconds to generate graph")
tic = t.process_time()

#circuit_ansatz(G, 'gamma', 'beta').draw(output = 'mpl') #draw the circuit

toc = t.process_time()
print("It took " + str(toc - tic) + " seconds to generate the grid")
tic = t.process_time()

# setting the bounds for gamma and beta (only for SLQP)
#bounds = ((0, np.pi), (0, 2*np.pi)) 

# Nelder-Mead optimizer:
max_expect_value = minimize(cost_function, x0=np.random.randn(2,p), args=(G,p), options={'disp': True}, method = 'Nelder-Mead')

# SLQP optimizer:
# max_expect_value = minimize(expect_value_function, x0=np.random.randn(2),args=(backend,G,shots), bounds = bounds, options={'disp': True}, method = 'SLQP')

# Differential evolution optimizer:
# max_expect_value = differential_evolution(expect_value_function,args=(backend,G,shots,noise_model), bounds=bounds)

toc = t.process_time()
print("It took " + str(toc - tic) + " seconds to optimize")
tic = t.process_time()

optimal_gamma, optimal_beta = max_expect_value['x'][:p], max_expect_value['x'][p:]
state_dictionary = execute_circuit(G,optimal_gamma,optimal_beta,p=p)
solution, solution_cost = get_solution(state_dictionary, G)

print('Expectation value of the cost function = ', -max_expect_value.get('fun'))
print('Approximation ratio = ', -max_expect_value.get('fun')/solution_cost )
print("Optimal gamma and beta are :(", optimal_gamma, ", ", optimal_beta, ")")


greedy_sol = greedy_solution(G)
greedy_sol_cost = cost_function_C(greedy_sol, G)
print(f'The greedy solution gives cost: {greedy_sol_cost}')

show_amplitudes(G, optimal_gamma, optimal_beta, p = p)
