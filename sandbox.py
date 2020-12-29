import qaoa_graphs as graphs
from qasm_functions import *
from qiskit.test.mock import FakeVigo

# General
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

# Qiskit
from qiskit.providers.aer import QasmSimulator
from qiskit import QuantumCircuit, execute, IBMQ
from qiskit.tools.monitor import job_monitor
from qiskit.visualization import plot_histogram

# Noises
from qiskit import Aer
from qiskit.providers.aer.noise import NoiseModel, QuantumError, depolarizing_error, thermal_relaxation_error
 
G = graphs.fournodes_3reg_graph()
n = len(G.nodes())
E = G.edges()


'''
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
'''
#backend = FakeVigo()
#vigo_simulator = QasmSimulator.from_backend(backend)
provider = IBMQ.load_account()
backend = provider.get_backend('ibmq_vigo')
IBMQ.providers()    # List all available providers

circ = QuantumCircuit(3, 3)
circ.h(0)
circ.cx(0, 1)
circ.cx(1, 2)
circ.measure([0, 1, 2], [0, 1, 2])

job = execute(circ, backend=backend, shots = 1000)
job.status()
results = job.result()
counts = results.get_counts()
plot_histogram(counts)
print(counts)