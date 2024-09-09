import gc
import os
import sys
import numpy as np
import itertools
import math
from pyscipopt import quicksum, Model
import networkx as nx
import branch_and_bound as bnb


problem_class = ''

def generate(problem, parameters):
    global problem_class
    # np.random.seed(0)

    problem_class = problem
    model = globals()[problem_class](parameters)
    dummy_file = f"test_{problem}_{np.random.rand()}.mps"
    model.writeProblem(dummy_file)
    # model.optimize()
    # 
    # if model.getStatus() != "optimal":
    #     os.remove(dummy_file)
    #     print("not optimal")
    #     return generate(problem, parameters)

    # size = bnb.BranchAndBound(dummy_file, "strong_branching").solve_bnb()[0]
    #
    # if size < 50 :
    #     print(f"re-generating, sb size = {size}")
    #     os.remove(dummy_file)
    #     return generate(problem, parameters)
    #
    # print(f"SB tree size = {size}")

    del model
    gc.collect()

    pyopt = Model()
    pyopt.readParams("scip_parameters.txt")
    pyopt.hideOutput()
    pyopt.readProblem(dummy_file)
    os.remove(dummy_file)

    return pyopt


def PackingIP(parameters):
    n, p, sparsity = parameters
    items = range(n)

    weights = np.random.randint(1, 10 * n, (p, n))
    p_matrix = np.random.rand(p, n)
    weights[p_matrix > sparsity] = 0

    capacities = np.ceil(0.5 * np.sum(weights, 1))

    values = np.random.randint(1, 10 * n + 1, n).astype(float)  # obj: value of items

    m = Model(f"{problem_class} {parameters}")
    m.readParams("scip_parameters.txt")

    x = {}
    for i in items:
        x[i] = m.addVar(lb=0, ub=1, vtype='B', name=f"x({i})")

    for j in range(p):
        m.addCons(quicksum(weights[j, i] * x[i] for i in items) <= capacities[j], f"packing({j})")

    m.setObjective(quicksum(values[i] * x[i] for i in items), "maximize")
    m.hideOutput()

    return m


def ContPackingIP(parameters):
    n, p, sparsity, f = parameters
    items = range(n)

    weights = np.random.random((p, n))
    p_matrix = np.random.rand(p, n)
    weights[p_matrix > sparsity] = 0

    capacities = f * np.sum(weights, 1)
    
    np.round(capacities, 3, capacities)
    capacities *= 1000

    np.round(weights, 3, weights)
    weights *= 1000

    values = np.random.random(n)  # obj: value of items

    m = Model(f"{problem_class} {parameters}")
    m.readParams("scip_parameters.txt")

    x = {}
    for i in items:
        x[i] = m.addVar(lb=0, ub=1, vtype='B', name=f"x({i})")

    for j in range(p):
        m.addCons(quicksum(weights[j, i] * x[i] for i in items) <= capacities[j], f"packing({j})")

    m.setObjective(quicksum(values[i] * x[i] for i in items), "maximize")
    m.hideOutput()

    return m


def CoveringIP(parameters):
    n, c, sparsity = parameters
    items = range(n)

    weights = np.random.randint(1, 10 * n, (c, n))
    p_matrix = np.random.rand(c, n)
    weights[p_matrix > sparsity] = 0

    capacities = np.ceil(0.5 * np.sum(weights, 1))

    values = np.random.randint(1, 10 * n + 1, n).astype(float)  # obj: value of items

    m = Model(f"{problem_class} {parameters}")
    m.readParams("scip_parameters.txt")

    x = {}
    for i in items:
        x[i] = m.addVar(lb=0, ub=1, vtype='B', name=f"x({i})")

    for j in range(c):
        m.addCons(quicksum(weights[j, i] * x[i] for i in items) >= capacities[j], f"covering({j})")

    m.setObjective(quicksum(values[i] * x[i] for i in items), "minimize")
    m.hideOutput()

    return m


def CCPPower(parameters):
    # bertsimas pg 10 (Multiperiod planning of electric p ower capacity) + Simge ipco'21 talk pg 75
    time_periods, scenarios = parameters
    T = range(time_periods)
    S = range(scenarios)

    coal_plant_life = 15
    nuc_plant_life = 10
    nuc_perc_lim = 0.2

    epsilon = 0.1  # 90% confidence
    scenario_prob = [1 / scenarios] * scenarios

    coal_capex_perMW = np.random.randint(200, 300, time_periods).astype('float')
    nuc_capex_perMW = np.random.randint(100, 200, time_periods).astype('float')

    existing_cap = np.zeros(time_periods).astype('float')
    C = np.random.randint(100, 500)
    f = np.random.randint(70, 100) / 100
    for i in range(round(time_periods / 2)):
        existing_cap[i] = C * (f ** i)

    demand_t_s = np.random.randint(300, 700, (time_periods, scenarios)).astype('float')

    m = Model(f"{problem_class} {parameters}")
    m.readParams("scip_parameters.txt")

    new_coal, new_nuc, tot_coal, tot_nuc, y, z = {}, {}, {}, {}, {}, {}
    for t in T:
        new_coal[t] = m.addVar(lb=0, vtype='C', name=f"new_coal({t})")
        new_nuc[t] = m.addVar(lb=0, vtype='C', name=f"new_nuc({t})")
        tot_coal[t] = m.addVar(lb=0, vtype='C', name=f"tot_coal({t})")
        tot_nuc[t] = m.addVar(lb=0, vtype='C', name=f"tot_nuc({t})")
        y[t] = m.addVar(lb=0, vtype='C', name=f"y({t})")

    for s in S:
        z[s] = m.addVar(lb=0, ub=1, vtype='B', name=f"z({s})")

    for t in T:
        m.addCons(tot_coal[t] - quicksum(new_coal[j] for j in range(max(0, t - coal_plant_life), t+1)) == 0,
                  name=f"account_coal({t})")
        m.addCons(tot_nuc[t] - quicksum(new_nuc[j] for j in range(max(0, t - nuc_plant_life), t+1)) == 0,
                  name=f"account_nuc({t})")
        m.addCons((1 - nuc_perc_lim) * tot_nuc[t] - nuc_perc_lim * tot_coal[t] <= nuc_perc_lim * existing_cap[t], 
                  name=f"nuc_lim({t})")
        m.addCons(y[t] - tot_coal[t] - tot_nuc[t] == existing_cap[t],
                  name=f"total_power({t})")

    for t in T:
        for s in S:
            m.addCons((y[t] + demand_t_s[t, s] * z[s] >= demand_t_s[t, s]), name=f"mixing({t},{s})")

    m.addCons(quicksum(z[s] * scenario_prob[s] for s in S) <= epsilon, name='allowed_violation')

    m.setObjective(quicksum(coal_capex_perMW[t] * new_coal[t] for t in T) +
                   quicksum(nuc_capex_perMW[t] * new_nuc[t] for t in T),
                   "minimize")

    m.hideOutput()
    
    return m


def CCPPortfolio(parameters):
    # scenarios = m
    # lin_var = n = number of variables in a'x >= r
    # k = number of scenarios out of m that can be violated
    # budget constraint = True if {1'x = 1} included

    lin_var, scenarios, k, budget_con = parameters
    A = np.round(np.random.uniform(0.8, 1.5, (scenarios, lin_var)), 6)
    r = 1.1
    
    V = range(lin_var)
    S = range(scenarios)
    
    if budget_con:
        c = np.random.randint(1, 100, lin_var)
    else:
        c = np.ones(lin_var)

    m = Model(f"{problem_class} {parameters}")
    m.readParams("scip_parameters.txt")
    
    x, z = {}, {}
    
    for v in V: 
        x[v] = m.addVar(lb=0, vtype='C', name=f"x({v})")
    
    for s in S:
        z[s] = m.addVar(lb=0, ub=1, vtype='B', name=f"z({s})")

    for s in S:
        m.addCons(quicksum(A[s, v] * x[v] for v in V) + r * z[s] >= r, name=f"scenarios({s})")
    
    m.addCons(quicksum(z[s] for s in S) <= k, name='k_violation')

    if budget_con:
        m.addCons(quicksum(x[v] for v in V) == 1, name='budget')

    m.setObjective(quicksum(c[v] * x[v] for v in V), "minimize")

    m.hideOutput()

    return m


def FixedChargeFlowProblem(parameters):
    num_nodes, num_edges, num_commodities = parameters  # edges > 2 * nodes

    G = nx.random_regular_graph(2, num_nodes, 0)
    nodes = np.fromiter(G.nodes, dtype=int)

    while G.number_of_edges() < num_edges:
        new_edge = np.random.choice(nodes, 2, False)
        G.add_edge(*new_edge)

    while not nx.is_connected(G):

        components = list(nx.connected_components(G))

        select_components = np.random.choice(len(components), 2, False)

        edge_set_1 = G.edges(components[select_components[0]])
        edge_set_2 = G.edges(components[select_components[1]])

        edge_1 = edge_set_1[np.random.choice(len(edge_set_1))]
        edge_2 = edge_set_2[np.random.choice(len(edge_set_2))]

        if np.random.choice(2):
            new_edge_1 = (edge_1[0], edge_2[0])
            new_edge_2 = (edge_1[1], edge_2[1])
        else:
            new_edge_1 = (edge_1[0], edge_2[1])
            new_edge_2 = (edge_1[0], edge_2[1])

        G.add_edges_from([new_edge_1, new_edge_2])
        G.remove_edges_from([edge_1, edge_2])

    for _ in range(5):

        edge_set = G.edges()
        edges = list(edge_set)

        select_edges = np.random.choice(len(edge_set), 2, False)

        edge_1 = edges[select_edges[0]]
        edge_2 = edges[select_edges[1]]

        if np.random.choice(2):
            new_edge_1 = (edge_1[0], edge_2[0])
            new_edge_2 = (edge_1[1], edge_2[1])
        else:
            new_edge_1 = (edge_1[0], edge_2[1])
            new_edge_2 = (edge_1[1], edge_2[0])

        if new_edge_1 not in edge_set and new_edge_2 not in edge_set and \
                new_edge_1[0] != new_edge_1[1] and new_edge_2[0] != new_edge_2[1]:
            G.remove_edges_from([edge_1, edge_2])
            G.add_edges_from([new_edge_1, new_edge_2])
    
            if not nx.is_connected(G):
                G.remove_edges_from([new_edge_1, new_edge_2])
                G.add_edges_from([edge_1, edge_2])

    source_sinks = np.random.choice(G.nodes(), 2*num_commodities, False)
    sources = source_sinks[:num_commodities]
    sinks = source_sinks[num_commodities:]
    demands = np.random.randint(100, 300, num_commodities)

    unit_costs = np.random.randint(3, 10, num_edges)
    fixed_costs = np.random.randint(3*np.sum(demands), 8*np.sum(demands), num_edges)
    capacities = np.random.randint(30*num_commodities, 80*num_commodities, num_edges)

    edges = list(G.edges())

    commodities = range(num_commodities)
    m = Model(f"{problem_class} {parameters}")
    m.readParams("scip_parameters.txt")

    delta = np.zeros((num_commodities, num_nodes))
    delta[commodities, sources] = 1
    delta[commodities, sinks] = -1

    x, y, z = {}, {}, {}
    for e in edges:
        for k in commodities:
            x[(k, e)] = m.addVar(lb=-1, ub=1, vtype='C', name=f"x_{k}_({e[0]},{e[1]})")
            z[(k, e)] = m.addVar(lb=0, ub=1, vtype='C', name=f"z_{k}_({e[0]},{e[1]})")
            # fraction of k's demand going through e (along the fixed direction, negative if opp)
            # z = abs val of x

    for e in edges:
        y[e] = m.addVar(lb=0, ub=1, vtype='B', name=f"y({e[0]},{e[1]})") # whether edge e is used or not

    m.setObjective(quicksum(unit_costs[i] * demands[k] * z[(k, edges[i])]
                            for i in range(num_edges) for k in commodities) +
                   quicksum(fixed_costs[i] * y[edges[i]]
                            for i in range(num_edges)),
                   "minimize")

    for e in edges:
        for k in commodities:
            m.addCons(z[(k, e)] >= x[(k, e)], name=f"z_bound_pos_{k}_({e[0]},{e[1]})")
            m.addCons(z[(k, e)] >= -x[(k, e)], name=f"z_bound_neg_{k}_({e[0]},{e[1]})")

    for v in nodes:
        out_edges = [e for e in edges if e[0] == v]
        in_edges = [e for e in edges if e[1] == v]

        for k in commodities:
            m.addCons(quicksum(x[(k, e)] for e in out_edges) - quicksum(x[(k, e)] for e in in_edges)
                      == delta[k, v], name=f"flow_conservation({v},{k})")

    for i in range(num_edges):
        m.addCons(quicksum(demands[k] * z[(k, edges[i])] for k in commodities) <= capacities[i] * y[edges[i]],
                  name=f"capacity_({edges[i][0]},{edges[i][1]})")

    m.hideOutput()

    return m


