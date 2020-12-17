import networkx as nx

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TKAgg')
import pylab as pl

from qiskit import Aer
from qiskit.providers.aer import QasmSimulator
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, execute
from qiskit.tools.monitor       import job_monitor
from qiskit.visualization import *

#Optimizers
from scipy.optimize import minimize, LinearConstraint, differential_evolution

#Noises
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise import QuantumError
from qiskit.providers.aer.noise import thermal_relaxation_error

#Importing time
import time as t
tic = t.process_time()

# Generating the graph G(V,N)
def diamond_graph(): #Starmon 5 shape
    n     = 5
    V     = np.arange(0,n,1)
    E     =[(0,1,1.0),(0,2,1.0),(0,3,1.0),(4,0,1.0)]
    G     = nx.Graph()
    G.add_nodes_from(V)
    G.add_weighted_edges_from(E)
    return G

def triangle():
    n     = 3
    V     = np.arange(0,n,1)
    E     =[(0,1,1.0),(1,2,1.0),(2,0,1.0)]
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

def weighted_fournodes_graph(): #4-node 3-regular yutsis graph
    n     = 4
    V     = np.arange(0,n,1)
    E     =[(0,1,2.0), (1,2,1.0), (2,3,1.0), (3,0,1.0), (0,2,1.0), (1,3,1.0)]
    G     = nx.Graph()
    G.add_nodes_from(V)
    G.add_weighted_edges_from(E)
    return G


# Choose your fighter
G = fournodes_3reg_graph()
n = len(G.nodes())
E = G.edges()

# Generate plot of the Graph
colors = ['g' for node in G.nodes()]
nx.draw_networkx(G, node_color=colors)

toc = t.process_time()
print("It took " + str(toc - tic) + " seconds to generate graph")
tic = t.process_time()

p = 2

def circuit_ansatz(G, gamma, beta, p=1): #gamma and beta are p-lists
    n = len(G.nodes())
    E = G.edges()

    QAOA = QuantumCircuit(n, n)
    QAOA.h(range(n))
    QAOA.barrier()
    for i in range(p):
        for edge in E:
            k = edge[0]
            l = edge[1]
            QAOA.cu1(-2*gamma[i], k, l) #Controlled-Z gate with a -2*gamma phase
            QAOA.u1(gamma[i], k) #Rotation of gamma around the z axis
            QAOA.u1(gamma[i], l) #
        QAOA.barrier()
        QAOA.rx(2*beta[i], range(n)) #X rotation
        QAOA.barrier()

    #QAOA.measure(range(n),range(n)) #measures the circuit
    return QAOA

#circuit_ansatz(G, 'gamma', 'beta').draw(output = 'mpl') #draw the circuit

def execute_circuit(G, gamma, beta, backend, p = 1, noise_model = None):
    QAOA = circuit_ansatz(G, gamma, beta, p = p) #creates the circuit
    result = execute(QAOA, backend=backend).result()
    #job_monitor(job)
    statevector = result.get_statevector(QAOA)
    amplitudes = ([abs(i) for i in statevector])
    state_dictionary = {bin(i)[2:].zfill(n) : amplitudes[i]**2 for i in range(len(amplitudes))}
    #plot_histogram(state_dictionary,figsize = (8,6),bar_labels = False)
    return state_dictionary #Dictionary holding all the amplitudes for the states

def cost_function_C(x,G): #x is a list
    E = G.edges()
    C = 0
    for vertice in E:
        e1 = vertice[0]
        e2 = vertice[1]
        w = G[e1][e2]['weight']
        C = C + w*x[e1]*(1-x[e2]) + w*x[e2]*(1-x[e1])
    return C

def get_expectval(state_dict): #state_dict = dictionary holding 'state' and amplitudes
    total_cost = 0
    for state in list(state_dict.keys()):
        x = [int(bit_num) for bit_num in list(state)] #the bit string is saved as a list
        cost_x = cost_function_C(x,G)
        total_cost += (state_dict[state]**2)*cost_x
    return total_cost

def get_solution(state_dict): #takes as the solution the state with the highest cost within all the measured states
    solution_cost = 0
    for state in list(state_dict.keys()):
        x = [int(bit_num) for bit_num in list(state)] #the bit string is saved as a list
        cost_x = cost_function_C(x,G)
        if cost_x > solution_cost:
            solution = x
            solution_cost = cost_x
    print(f'The solution is the state {solution} with a cost value of {solution_cost}')
    return solution, solution_cost

from qiskit.providers.aer import StatevectorSimulator
backend = StatevectorSimulator()

def expect_value_function(parameters, backend, G, p = 1, noise_model = None):
    gamma = parameters[0:p]
    beta = parameters[p:]
    state_dictionary = execute_circuit(G, gamma, beta, backend, p = p, noise_model = noise_model)
    avr_cost = get_expectval(state_dictionary)
    return -avr_cost

'''
# Making the grid -----------------------------------------------------------------------------------
gamma_max = 2*np.pi ;beta_max = 2*np.pi
gamma_list = np.linspace(0,gamma_max,20) ; beta_list = np.linspace(0,beta_max,20)
avr_cost_grid = np.zeros((len(gamma_list),len(beta_list)))

for i in range(len(gamma_list)): #gamma and beta grid between 2pi and 2pi
    gamma = gamma_list[i]
    for j in range(len(beta_list)):
        beta = beta_list[j]
        state_dictionary = execute_circuit(G, gamma, beta, backend)

        avr_cost = get_expectval(state_dictionary)

        avr_cost_grid[i,j] = avr_cost

f = pl.figure(facecolor='w', edgecolor='k')
pl.imshow(avr_cost_grid, interpolation = 'bicubic', extent = [0,beta_max,0,gamma_max])
pl.title('Ideal parameter grid', fontsize=8)
pl.colorbar()
pl.xlabel(r'$\beta$')
pl.ylabel(r'$\gamma$')
#pl.show()
#--------------------------------------------------------------------------------------------------------------------------------------------------
'''

toc = t.process_time()
print("It took " + str(toc - tic) + " seconds to generate the grid")
tic = t.process_time()

# setting the bounds for gamma and beta
#bounds = ((0, np.pi), (0, 2*np.pi))
# Nelder-Mead optimizer:
max_expect_value = minimize(expect_value_function, x0=np.random.randn(2,p), args=(backend,G,p), options={'disp': True}, method = 'Nelder-Mead')
# SLQP optimizer:
#max_expect_value = minimize(expect_value_function, x0=np.random.randn(2),args=(backend,G,shots), bounds = bounds, options={'disp': True}, method = 'SLQP')

#optimal_gamma, optimal_beta = max_expect_value['x']

toc = t.process_time()
print("It took " + str(toc - tic) + " seconds to optmize")
tic = t.process_time()

#state_dict = execute_circuit(G,optimal_gamma,optimal_beta,backend)
#solution, solution_cost = get_solution(state_dict)

#print('Optimal gamma, beta = ', optimal_gamma, optimal_beta)
print('Expectation value of the cost function = ', -max_expect_value.get('fun'))
#print('Approximation ratio = ', -max_expect_value.get('fun')/solution_cost )

#plot_histogram(counts,figsize = (8,6),bar_labels = False)


optimal_gamma, optimal_beta = max_expect_value['x'][:p], max_expect_value['x'][p:]
print("Optimal gamma and beta are :(", optimal_gamma, ", ", optimal_beta, ")")

'''
max_expect_value = differential_evolution(expect_value_function,args=(backend,G,shots,noise_model), bounds=bounds)
optimal_gamma, optimal_beta = max_expect_value['x']

counts = execute_circuit(G,optimal_gamma,optimal_beta,backend,shots)
solution, solution_cost = get_solution(counts)

print('Optimal gamma, beta = ', optimal_gamma, optimal_beta)
print('Expectation value of the cost function = ', -max_expect_value.get('fun'))
print('Approximation ratio = ', -max_expect_value.get('fun')/solution_cost )

plot_histogram(counts,figsize = (8,6),bar_labels = False)
'''

def new_cost(vertex, set, G):
    cost = 0
    Edges = [edge for edge in G.edges()]
    for V in set:
        if (V, vertex) or (vertex, V) in Edges:
            cost += 1
    return cost

def greedy_solution(G):
    set_A = [list(G.nodes())[0]]
    set_B = [list(G.nodes())[1]]
    x = [0,1]
    for vertex in list(G.nodes())[2:]:
        if new_cost(vertex,set_A, G) > new_cost(vertex,set_B, G):
            set_B.append(vertex)
            x.append(1)
        else:
            set_A.append(vertex)
            x.append(0)
    return x

greedy_sol = greedy_solution(G)
greedy_sol_cost = cost_function_C(greedy_sol, G)
print(f'The greedy solution gives cost: {greedy_sol_cost}')

states_dict = execute_circuit(G, optimal_gamma, optimal_beta, backend)

def show_amplitudes(G, gamma, beta, p =1):

    n = len(G.nodes())
    E = G.edges()

    QAOA = QuantumCircuit(n, n)
    QAOA.h(range(n))
    QAOA.barrier()

    fig = plt.figure()
    graph = plt.bar(range(2**n), [1/(2**n) for i in range(2**n)], align = 'center')
    x_ticks = [bin(i)[2:].zfill(n) for i in range(2**n)]
    plt.xticks(ticks = range(2**n), labels = x_ticks, rotation = 60)
    plt.ylim(0,0.75)
    plt.title(f'Initial State')
    fig.canvas.draw()
    plt.pause(5)
    plt.savefig(f'Initial State bad')

    for i in range(p):
        for edge in E:
            k = edge[0]
            l = edge[1]
            QAOA.cu1(-2*gamma[i], k, l) #Controlled-Z gate with a -2*gamma phase
            QAOA.u1(gamma[i], k) #Rotation of gamma around the z axis
            QAOA.u1(gamma[i], l) #

        statevector = execute(QAOA, backend = backend).result().get_statevector(QAOA)
        amplitudes = ([abs(i) for i in statevector])
        state_dict = {bin(i)[2:].zfill(n) : amplitudes[i]**2 for i in range(len(amplitudes))}

        for rectangle, ampl in zip(graph, amplitudes):
            rectangle.set_height(ampl**2)
        plt.title(f'Cost: {get_expectval(state_dict)}\n Iteration {i + 1}, after applying $U_C$\n $\gamma$ = {optimal_gamma[i]}')
        fig.canvas.draw()
        plt.savefig(f'{i} gamma bad')
        plt.pause(0.5)

        QAOA.barrier()
        QAOA.rx(2*beta[i], range(n)) #X rotation
        statevector = execute(QAOA, backend = backend).result().get_statevector(QAOA)

        amplitudes = ([abs(i) for i in statevector])
        state_dict = {bin(i)[2:].zfill(n) : amplitudes[i]**2 for i in range(len(amplitudes))}

        for rectangle, ampl in zip(graph, amplitudes):
            rectangle.set_height(ampl**2)
        plt.title(f'Cost: {get_expectval(state_dict)}\n Iteration {i + 1}, after applying $U_B$\n $\\beta$ = {optimal_gamma[i]}')
        fig.canvas.draw()
        plt.savefig(f'{i} beta bad')
        plt.pause(0.5)

        QAOA.barrier()
    plt.show()

show_amplitudes(G, optimal_gamma, optimal_beta, p = p)
