# Private libraries
import qaoa_graphs as graphs
from qasm_functions import *

# General
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

# Qiskit
from qiskit.providers.aer import QasmSimulator
from qiskit import QuantumCircuit, execute
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_histogram
import warnings
warnings.filterwarnings('ignore')

# Optimizers
from scipy.optimize import minimize, differential_evolution

# Noises
from qiskit import Aer
from qiskit.providers.aer.noise import NoiseModel, QuantumError, thermal_relaxation_error, depolarizing_error
 
# Choose your fighter
G = graphs.fournodes_3reg_graph()
n = len(G.nodes())
E = G.edges()

# Choose the arena
backend = QasmSimulator()
shots = 10000

# Choose number of rounds
p = 2

# Noise model
error1 = depolarizing_error(0.2, 1) #single qubit gates
error2 = depolarizing_error(0.2, 2) #two qubit gates

noise_model = NoiseModel()
noise_model.add_all_qubit_quantum_error(error1, ['u1', 'u2', 'u3'])
noise_model.add_all_qubit_quantum_error(error2, ['cx'])

basis_gates = noise_model.basis_gates

cost_list = []
n = 5

for p in range(1,n):

    # Initial values
    x0 = np.random.randn(2,p)

    # Parameters bounds
    bound = (0, 2*np.pi)
    bounds = []
    for i in range(2*p):
        bounds.append(bound)

    # Nelder-Mead optimizer:
    # max_expect_value = minimize(expect_value_function, x0=x0,args=(backend,G,shots,p,None), options={'disp': True}, method = 'Nelder-Mead')

    # SLQP optimizer:
    # max_expect_value = minimize(expect_value_function, x0=np.random.randn(2),args=(backend,G,shots,p,None), bounds = bounds, options={'disp': True}, method = 'SLQP')

    # Differential evolution optimizer:
    max_expect_value = differential_evolution(expect_value_function, bounds=bounds, args=(backend,G,shots,p,NoiseModel))

    optimal_gamma, optimal_beta = max_expect_value['x'][:p], max_expect_value['x'][p:]
    counts = execute_circuit(G, optimal_gamma, optimal_beta, backend, shots, p=p)
    solution, solution_cost = get_solution(counts, G)
    avr_cost = -max_expect_value.get('fun')

    cost_list.append(avr_cost)

    print(f"Number of layers: {p}")
    print('Optimal gamma, optimal beta = ', optimal_gamma, optimal_beta)
    print('Expectation value of the cost function = ', avr_cost)
    print('Approximation ratio = ', avr_cost/solution_cost ) # This approximation ratio is computed assuming the algorithm is able to find the solution

    #plot_histogram(counts,figsize = (8,6),bar_labels = False)
    #plt.show()

np.save('saved_cost_list', cost_list)

plt.plot(range(1,n), cost_list)
plt.show()
plt.save('cost list')

''' This is my attempt of building a cheating noise model 

# T1 and T2 values (from analysis)
T1 = 45e-3
T2 = 20e-3

# Gate execution times
T_1qubit_gates = 20e-6
T_2qubit_gates = 60e-6

# Thermal relaxation
single_qubit_error = thermal_relaxation_error(T1, T2, T_1qubit_gates) #single qubit gates
two_qubits_error = thermal_relaxation_error(T1, T2, T_2qubit_gates) #two qubit gates

#missing gate (in)fidelity error. 1qubit = 1.5 × 10 −3 , 2qubit = 4 × 10 −2

# Add errors to noise model
noise_model = NoiseModel()
noise_model.add_all_qubit_quantum_error(single_qubit_error, ['u1', 'u2', 'u3'])
noise_model.add_all_qubit_quantum_error(two_qubits_error, ['Cx'])
'''