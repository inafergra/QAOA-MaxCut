#math and plotting
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

#qiskit stuff
from qiskit import Aer
from qiskit.providers.aer import QasmSimulator, StatevectorSimulator
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute
from qiskit.tools.monitor       import job_monitor
from qiskit.visualization import plot_histogram
import warnings
warnings.filterwarnings('ignore')

#Optimizers
from scipy.optimize import minimize, LinearConstraint, differential_evolution

#Noises
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise import QuantumError
from qiskit.providers.aer.noise import thermal_relaxation_error

# Generating the graph G(V,N)
def diamond_graph(): #Starmon 5 shape
    n     = 5
    V     = np.arange(0,n,1)
    E     =[(0,1,1.0),(0,2,1.0),(0,3,1.0),(4,0,1.0)] 
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

#Choose your fighter
G = fournodes_3reg_graph()
n = len(G.nodes())
E = G.edges()

# Generate plot of the Graph
#colors = ['g' for node in G.nodes()]
#nx.draw_networkx(G, node_color=colors)

def circuit_ansatz(G, gamma, beta , p=1): #gamma and beta are p-arrays or lists
    n = len(G.nodes())
    E = G.edges()

    QAOA = QuantumCircuit(n, n)
    for i in range(p):
        QAOA.h(range(n))

        QAOA.barrier()
        for edge in E:
            k = edge[0]
            l = edge[1]
            QAOA.cu1(-2*gamma[i], k, l) #Controlled-Z gate with a -2*gamma phase
            QAOA.u1(gamma[i], k) #Rotation of gamma around the z axis
            QAOA.u1(gamma[i], l) #

        QAOA.barrier()
        QAOA.rx(2*beta[i], range(n)) #X rotation

    QAOA.barrier()
    QAOA.measure(range(n),range(n)) #measures the circuit

    return QAOA

p=2
#gamma = [str(f'gamma{layer+1}') for layer in range(p)]
#beta = [str(f'beta{layer+1}') for layer in range(p)]
#circuit_ansatz(G, 'gamma', 'beta', p = p).draw(output = 'mpl') #draw the circuit

def execute_circuit(G, gamma, beta, backend, shots, p, noise_model = None ): #returns an instance of the Results class

    QAOA = circuit_ansatz(G, gamma, beta, p=p) #creates the circuit
    job = execute(QAOA, backend=backend, shots=shots, noise_model=noise_model)
    #job_monitor(job)
    results = job.result()
    counts = results.get_counts() #dictionary with keys 'bit string x' and items 'counts of x'
    print(counts)
    return counts

def cost_function_C(x,G): #input x is a list
    E = G.edges()
    C = 0
    for vertice in E:
        e1 = vertice[0]
        e2 = vertice[1]
        w = G[e1][e2]['weight']
        C = C + w*x[e1]*(1-x[e2]) + w*x[e2]*(1-x[e1])
    return C


def get_expectval(counts, shots):
    total_cost = 0
    for sample in list(counts.keys()):
        x = [int(bit_num) for bit_num in list(sample)] #the bit string is saved as a list
        cost_x = cost_function_C(x,G)
        total_cost += counts[sample]*cost_x
    avr_cost = total_cost/shots
    return avr_cost

def get_solution(counts): #takes as the solution the state with the highest cost within all the measured states
    solution_cost = 0
    for sample in list(counts.keys()):
        x = [int(bit_num) for bit_num in list(sample)] #the bit string is saved as a list
        cost_x = cost_function_C(x,G)
        if cost_x > solution_cost:
            solution = x
            solution_cost = cost_x
    print(f'The solution is the state {solution} with a cost value of {solution_cost}')
    return solution, solution_cost


'''
# Import from Qiskit Aer noise module
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise import QuantumError
from qiskit.providers.aer.noise import thermal_relaxation_error

# T1 and T2 values (from analysis)
T1 = 45e-3
T2 = 20e-3

#Gate execution times
T_1qubit_gates = 20e-6
T_2qubit_gates = 60e-6

# Thermal relaxation
single_qubit_error = thermal_relaxation_error(T1, T2, T_1qubit_gates) #single qubit gates
two_qubits_error = thermal_relaxation_error(T1, T2, T_2qubit_gates) #two qubit gates

#missing gate (in)fidelity error. 1qubit = 1.5 × 10 −3 , 2qubit = 4 × 10 −2

# Add errors to noise model
noise_model = NoiseModel()
noise_model.add_all_qubit_quantum_error(single_qubit_error, ['u1', 'u2', 'u3'])
noise_model.add_all_qubit_quantum_error(two_qubits_error, ['CU1'])
'''

# Importing noise model from a IBM backend
#from qiskit import IBMQ
#provider = IBMQ.load_account()

# Making the grid with noise
from qiskit.providers.aer.noise import NoiseModel, QuantumError, depolarizing_error

error1 = depolarizing_error(0.2, 1) #single qubit gates
error2 = depolarizing_error(0.2, 2) #two qubit gates

noise_model = NoiseModel()
noise_model.add_all_qubit_quantum_error(error1, ['u1', 'u2', 'u3'])
noise_model.add_all_qubit_quantum_error(error2, ['Cx'])

backend = Aer.get_backend("qasm_simulator")
shots = 1000

def expect_value_function(parameters, backend, G, shots,p , noise_model = None):
    gamma = parameters[:p]
    beta = parameters[p:]
    #print(f'The parameters are : {parameters}')
    counts = execute_circuit(G, gamma, beta, backend, shots, p=p, noise_model = noise_model)
    avr_cost = get_expectval(counts, shots)
    return -avr_cost

# setting the bounds for gamma and beta
#bounds = ((0, np.pi), (0, 2*np.pi)) 
#initial values
np.random.seed(10)
x0 = np.random.randn(2,p)

# Nelder-Mead optimizer:
print("Before optimizer")
max_expect_value = minimize(expect_value_function, x0=x0,args=(backend,G,shots,p,noise_model), options={'disp': True}, method = 'Nelder-Mead')
print("After optimizer")

# SLQP optimizer:
#max_expect_value = minimize(expect_value_function, x0=np.random.randn(2),args=(backend,G,shots), bounds = bounds, options={'disp': True}, method = 'SLQP')

optimal_gamma, optimal_beta = max_expect_value['x'][:p], max_expect_value['x'][p:]

counts = execute_circuit(G, optimal_gamma, optimal_beta, backend, shots, p=p)
solution, solution_cost = get_solution(counts)

print('Optimal gamma, beta = ', optimal_gamma, optimal_beta)
print('Expectation value of the cost function = ', -max_expect_value.get('fun'))
print('Approximation ratio = ', -max_expect_value.get('fun')/solution_cost )

plot_histogram(counts,figsize = (8,6),bar_labels = False)
plt.show()