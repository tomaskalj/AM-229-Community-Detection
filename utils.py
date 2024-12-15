import cvxpy as cp
import networkx as nx
from itertools import combinations
import numpy as np
import gurobipy

def avg_clustering_union(graphs):
    union_graph = nx.Graph()
    for graph in graphs:
        union_graph = nx.compose(union_graph, graph)
    return nx.average_clustering(union_graph) 

def postprocess_communities(community_sequence):
    for level in range(len(community_sequence)):
        for l in range(level - 1, -1, -1):
            comms = community_sequence[l]
            for (graph, is_leaf) in comms:
                if is_leaf:
                    already_has = any(nx.utils.graphs_equal(graph, g) for (g, _) in community_sequence[level])
                    if not already_has:
                        community_sequence[level].append((graph, is_leaf))
    return community_sequence

def community_detect(G, alpha, beta, feasible, level=0, community_sequence={}):
    if level not in community_sequence:
        community_sequence[level] = []
    
    if len(G.nodes()) <= beta or not feasible:
        community_sequence[level].append((G, True))
        return [G], community_sequence
    
    community_sequence[level].append((G, False))
    
    components, is_feasible = cut_and_disconnect(G, alpha)
    communities = []

    if is_feasible:
        for component in components:
            comms, _ = community_detect(component, alpha, beta, is_feasible, level + 1, community_sequence)
            communities.extend(comms)
    else:
        comms, _ = community_detect(components[0], alpha, beta, is_feasible, level + 1, community_sequence)
        communities.extend(comms)

    return communities, community_sequence

def cut_and_disconnect(G, alpha):
    if not nx.is_directed(G):
        G = G.to_directed()
    
    node_pairs = list(combinations(G.nodes(), 2))
    
    min_int_result = float('inf')
    opt_int_edge_variables = None

    min_float_result = float('inf')
    opt_float_edge_variables = None
    
    for (src, sink) in node_pairs:
        result, edge_variables, _ = perform_min_cut(G, src, sink, alpha)
        if result is None:
            continue
        
        all_edge_ints = all(var.value.item().is_integer() for _, var in edge_variables.items())

        if result <= min_int_result and all_edge_ints:
            min_int_result = result
            opt_int_edge_variables = edge_variables
        elif result <= min_float_result and result.is_integer():
            min_float_result = result
            opt_float_edge_variables = edge_variables

    opt_edge_variables = None
    if opt_int_edge_variables is not None:
        opt_edge_variables = opt_int_edge_variables
    else:
        opt_edge_variables = opt_float_edge_variables
    
    # if for some reason we get to this point and the optimal edges and
    # nodes are still None, we just assume the community is optimal
    if opt_edge_variables is None:
        return [G.to_undirected().copy()], False

    # print('Optimal')
    # print(f'Result:', min_result)
    # for _, var in opt_edge_variables.items():
    #     print(f'{var.name()} = {var.value}')
    # for _, var in opt_node_variables.items():
    #     print(f'{var.name()} = {var.value}')
    # print()

    # convert to undirected for edge removal and
    # so we can get connected components
    G = G.to_undirected()

    opt_all_edge_ints = all(var.value.item().is_integer() for _, var in opt_edge_variables.items())
    
    for edge, var in opt_edge_variables.items():
        # if not all the edge variables are ints, let's just remove
        # all the nonzero edges so we can do something with the results
        if var.value == 1 or (not opt_all_edge_ints and var.value != 0):
            G.remove_edge(edge[0], edge[1])

    components = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    
    return components, True

def perform_min_cut(G, src, sink, alpha):
    edge_variables = {(u, v): cp.Variable(name=f'x_{u},{v}') for u, v in G.edges()}
    node_variables = {v: cp.Variable(name=f'y_{v}') for v in G.nodes()}
    
    constraints = []
    constraints += [node_variables[src] == 0];
    constraints += [node_variables[sink] == 1];
    
    for u, v in G.edges():
      e = (u, v)
      constraints += [node_variables[v] <= node_variables[u] + edge_variables[e]]
      constraints += [edge_variables[e] >= 0]
      constraints += [edge_variables[e] <= 1]
    
    for v in G.nodes():
      constraints += [node_variables[v] >= 0]
      constraints += [node_variables[v] <= 1]

    # these constraints prevent TUM
    constraints += [sum(node_variables[v] for v in G.nodes()) >= alpha]
    constraints += [sum(1 - node_variables[v] for v in G.nodes()) >= alpha]
    
    objective = cp.Minimize(sum(edge_variables[e] for e in G.edges()))
    problem = cp.Problem(objective, constraints)
    result = problem.solve(solver=cp.GUROBI)
    
    return result, edge_variables, node_variables
    