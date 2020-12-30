from qiskit import execute, QuantumCircuit

def circuit_ansatz(G, gamma, beta, prev_gamma, prev_beta, p): #gamma and beta are p-arrays or lists
    n = len(G.nodes())
    E = G.edges()
    temp_gamma = prev_gamma[:]
    temp_gamma.append(gamma)
    gamma = temp_gamma[:]
    temp_beta = prev_beta[:]
    temp_beta.append(beta)
    beta = temp_beta[:]
    QAOA = QuantumCircuit(n, n)
    for i in range(p):
        QAOA.h(range(n))

        QAOA.barrier()
        for edge in E:
            k = edge[0]
            l = edge[1]
            QAOA.cu1(-2*gamma[i], k, l) 
            QAOA.u1(gamma[i], k) 
            QAOA.u1(gamma[i], l)

        QAOA.barrier()
        QAOA.rx(2*beta[i], range(n)) 

    QAOA.barrier()
    QAOA.measure(range(n),range(n)) 

    return QAOA

def execute_circuit(G, gamma, beta, prev_gamma, prev_beta, backend, shots, p, noise_model): #returns an instance of the Results class

    QAOA = circuit_ansatz(G, gamma, beta, prev_gamma, prev_beta, p)

    if noise_model== None:
        job = execute(QAOA, backend=backend, shots=shots)

    else:
        basis_gates=noise_model.basis_gates 
        job = execute(QAOA, backend=backend, shots=shots, noise_model=noise_model, basis_gates=basis_gates)

    results = job.result()
    counts = results.get_counts() #dictionary with keys 'bit string x' and items 'counts of x'
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

def get_expectval(counts, shots, G):
    total_cost = 0
    for sample in list(counts.keys()):
        x = [int(bit_num) for bit_num in list(sample)] #the bit string is saved as a list
        cost_x = cost_function_C(x,G)
        total_cost += counts[sample]*cost_x
    avr_cost = total_cost/shots
    return avr_cost

def get_solution(counts, G): #takes as the solution the state with the highest cost within all the measured states
    solution_cost = 0
    for sample in list(counts.keys()):
        x = [int(bit_num) for bit_num in list(sample)] #the bit string is saved as a list
        cost_x = cost_function_C(x,G)
        if cost_x > solution_cost:
            solution = x
            solution_cost = cost_x
    return solution, solution_cost

def expect_value_function(parameters,prev_gamma,prev_beta, backend, G, shots, p, noise_model):
    gamma = parameters[0]
    beta = parameters[1]
    counts = execute_circuit(G, gamma, beta, prev_gamma, prev_beta, backend, shots, p=p, noise_model = noise_model)
    avr_cost = get_expectval(counts, shots, G)
    return -avr_cost


''' Stuff idk where to put and don't want to lose
gamma = [str(f'gamma{layer+1}') for layer in range(p)]
beta = [str(f'beta{layer+1}') for layer in range(p)]
circuit_ansatz(G, 'gamma', 'beta', p = p).draw(output = 'mpl') #draw the circuit
'''