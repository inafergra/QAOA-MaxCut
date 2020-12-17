import numpy as np
import matplotlib.pyplot as plt

from qiskit import Aer
from qiskit.providers.aer import QasmSimulator, StatevectorSimulator
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute

from qiskit.providers.aer.noise import NoiseModel, QuantumError, depolarizing_error
from qiskit.visualization import *

error1 = depolarizing_error(0.1, 1) #single qubit gates
error2 = depolarizing_error(0.1, 2) #two qubit gates

noise_model = NoiseModel()
noise_model.add_all_qubit_quantum_error(error1, ['u1', 'u2', 'u3'])
noise_model.add_all_qubit_quantum_error(error2, ['Cx'])

backend = Aer.get_backend("qasm_simulator")
shots = 10000

circ = QuantumCircuit(2,2)
circ.rx(np.pi, [0,1])
circ.measure_all()

job = execute(circ, backend = backend, shots = shots, noise_model = noise_model)
counts = job.result().get_counts()
plot_histogram(counts)
plt.show()