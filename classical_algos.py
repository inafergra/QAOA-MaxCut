
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