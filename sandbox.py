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

# Noises
from qiskit import Aer
from qiskit.providers.aer.noise import NoiseModel, QuantumError, depolarizing_error, thermal_relaxation_error
 
G = graphs.fournodes_3reg_graph()
n = len(G.nodes())
E = G.edges()



# T1 and T2 values (from analysis)
T1 = 45e-3
T2 = 20e-3

#Gate execution times
T_1qubit_gates = 20e-3
T_2qubit_gates = 60e-3

noise_model = NoiseModel()

# 
error1 = depolarizing_error(1, 1) #single qubit gates
error2 = depolarizing_error(0.02, 2) #two qubit gates
#single_qubit_error = thermal_relaxation_error(T1, T2, T_1qubit_gates) #single qubit gates
#two_qubits_error = thermal_relaxation_error(T1, T2, T_2qubit_gates) #two qubit gates

# Add errors to noise model
#noise_model.add_all_qubit_quantum_error(single_qubit_error, ['u1', 'u2', 'u3'])
#noise_model.add_all_qubit_quantum_error(two_qubits_error, ['cx'])
noise_model.add_all_qubit_quantum_error(error1, ['u1', 'u2', 'u3'])
noise_model.add_all_qubit_quantum_error(error2, ['cx'])

basis_gates = noise_model.basis_gates

circ = QuantumCircuit(1,1)
circ.x(0)
circ.measure_all()
job = execute(circ, backend=QasmSimulator(), shots=10000, basis_gates=basis_gates, noise_model=noise_model)
results = job.result()
counts = results.get_counts()
#plot_histogram(counts)
print(counts)
