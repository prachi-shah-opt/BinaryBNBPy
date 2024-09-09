import numpy as np
from itertools import product, combinations, permutations
from scipy.spatial import distance_matrix
import math
import gurobipy as gp
from gurobipy import quicksum
import scipy.sparse

problem_class = ''


def generate(problem, parameters):
    global problem_class
    # np.random.seed(0)

    problem_class = problem
    model = globals()[problem_class](parameters)
    model.optimize()

    if model.status != 2:
        print(f"not optimal, status = {model.status}")
        return generate(problem, parameters)

    return model


def ContPackingIP(parameters):
    n, p, sparsity, f = parameters

    weights = np.random.random((p, n))
    p_matrix = np.random.rand(p, n)
    weights[p_matrix > sparsity] = 0

    capacities = f * np.sum(weights, 1)

    np.round(capacities, 3, capacities)
    capacities *= 1000

    np.round(weights, 3, weights)
    weights *= 1000

    values = np.random.random(n)  # obj: value of items

    m = gp.Model()

    x = m.addVars(range(1, n + 1), lb=0, ub=1, obj=values, vtype='B', name='x')

    m.addMConstr(weights, x.values(), '<=', capacities, name='packing')

    m.ModelSense = -1
    # m.setParam('LogToConsole', 0)
    m.update()

    return m


# PACKING COVERING
def PackingIP(parameters):
    if len(parameters) == 3:
        n, p, sparsity = parameters
        rhs_ratio = 0.5
    else:
        n, p, sparsity, rhs_ratio = parameters

    weights = np.random.randint(1, 10 * n, (p, n))
    p_matrix = np.random.rand(p, n)
    weights[p_matrix > sparsity] = 0

    capacities = np.ceil(rhs_ratio * np.sum(weights, 1))

    values = np.random.randint(1, 10 * n + 1, n).astype(float)  # obj: value of items

    m = gp.Model()

    x = m.addVars(range(1, n + 1), lb=0, ub=1, obj=values, vtype='B', name='x')

    m.addMConstr(weights, x.values(), '<=', capacities, name='packing')

    m.ModelSense = -1
    # m.setParam('LogToConsole', 0)
    m.update()

    return m


def CoveringIP(parameters):
    if len(parameters) == 3:
        n, c, sparsity = parameters
        rhs_ratio = 0.5
    else:
        n, c, sparsity, rhs_ratio = parameters

    weights = np.random.randint(1, 10 * n, (c, n))
    p_matrix = np.random.rand(c, n)
    weights[p_matrix > sparsity] = 0

    capacities = np.ceil(rhs_ratio * np.sum(weights, 1))

    values = np.random.randint(1, 10 * n + 1, n).astype(float)  # obj: value of items

    m = gp.Model()

    x = m.addVars(range(1, n + 1), lb=0, ub=1, obj=values, vtype='B', name='x')

    m.addMConstr(weights, x.values(), '>=', capacities, name='packing')

    m.ModelSense = 1
    # m.setParam('LogToConsole', 0)
    m.update()

    return m


def MixedPackingCoveringIP(parameters):
    n, p, c, sparsity = parameters

    weights = np.random.randint(1, 10 * n, (p + c, n))
    p_matrix = np.random.rand(p + c, n)
    weights[p_matrix < sparsity] = 0

    capacities = np.sum(weights, 1).astype(float)
    capacities[: p] *= 0.7
    capacities[p:] *= 0.3

    capacities.round(out=capacities)

    if c * p > 0:
        values = np.random.randint(-5 * n, 5 * n + 1, n).astype(float)  # obj: value of items
    elif c == 0:
        values = np.random.randint(1, 10 * n + 1, n).astype(float)  # obj: value of items
    else:
        values = np.random.randint(-10 * n, 0, n).astype(float)  # obj: value of items

    m = gp.Model()

    x = m.addVars(range(1, n + 1), lb=0, ub=1, obj=values, vtype='B', name='x')

    m.addMConstr(weights[:p, :], x.values(), '<=', capacities[:p], name='packing')
    m.addMConstr(weights[p:, :], x.values(), '>=', capacities[p:], name='covering')

    m.ModelSense = -1
    # m.setParam('LogToConsole', 0)
    m.update()

    return m


def YangSetCovering(parameters):
    num_rows, num_cols = parameters
    # assert num_cols == 10 * num_rows

    n_i = np.random.choice(np.arange(np.ceil(2 * num_cols / 25 + 1), np.floor(3 * num_cols / 25)), (num_rows, 1))
    uniform_matrix = np.random.rand(num_rows, num_cols)

    A_matrix = np.zeros((num_rows, num_cols))
    A_matrix[uniform_matrix < n_i / num_cols] = 1

    cost = np.random.randint(1, 101, num_cols)

    m = gp.Model()

    x = m.addVars(range(1, num_cols + 1), lb=0, ub=1, obj=cost, vtype='B', name='x')

    m.addMConstr(A_matrix, x.values(), '>=', np.ones((num_rows, 1)), name='covering')

    m.ModelSense = 1
    # m.setParam('LogToConsole', 0)
    m.update()

    return m


def YangSetPacking(parameters):
    num_rows, num_cols = parameters
    # assert num_cols == 5 * num_rows

    n_i = np.random.choice(np.arange(np.ceil(2 * num_cols / 25 + 1), np.floor(3 * num_cols / 25)), (num_rows, 1))
    uniform_matrix = np.random.rand(num_rows, num_cols)

    A_matrix = np.zeros((num_rows, num_cols))
    A_matrix[uniform_matrix < n_i / num_cols] = 1

    cost = np.random.randint(1, 101, num_cols)

    m = gp.Model()

    x = m.addVars(range(1, num_cols + 1), lb=0, ub=1, obj=cost, vtype='B', name='x')

    m.addMConstr(A_matrix, x.values(), '<=', np.ones((num_rows, 1)), name='packing')

    m.ModelSense = -1
    # m.setParam('LogToConsole', 0)
    m.update()

    return m


# CARDINALITY CONSTRAINED
def MaximumCoverage(parameters):
    num_elements, num_sets, k = parameters  # p = probability of an item being in a set
    p = 30 / num_elements
    u_outcome = np.random.random((num_sets, num_elements))
    sets = []
    for row in u_outcome:
        sets.append(np.argwhere(row < p).flatten())

    m = gp.Model()
    x_sets = m.addVars(num_sets, vtype='B')
    y_elements = m.addVars(num_elements, lb=0, ub=1, obj=np.random.uniform(0.5, 1, num_elements), vtype='C')

    for i, row in enumerate(u_outcome.transpose()):
        sets_i = np.argwhere(row < p).flatten()
        m.addConstr(y_elements[i] - quicksum(x_sets[j] for j in sets_i) <= 0, name=f"element{i}")

    m.addConstr(quicksum(x_sets.values()) <= k, name="cardinality")

    m.ModelSense = -1
    # m.setParam('LogToConsole', 0)
    m.update()

    return m


def CCPPortfolio(parameters):
    # scenarios = m
    # lin_var = n = number of variables in a'x >= r
    # k = number of scenarios out of m that can be violated
    # budget constraint = True if {1'x = 1} included

    lin_var, scenarios, k, budget_con = parameters
    A = np.round(np.random.uniform(0.8, 1.5, (scenarios, lin_var)), 6)
    if scenarios > 50:
        A[np.random.rand(scenarios, lin_var) < 0.5] = 0

    r = 1.1

    if budget_con:
        c = np.random.randint(1, 100, lin_var)
    else:
        c = np.ones(lin_var)

    m = gp.Model()

    x = m.addVars(range(1, lin_var + 1), lb=0, obj=-c, vtype='C', name='x')
    z = m.addVars(range(1, scenarios + 1), lb=0, ub=1, obj=0, vtype='B', name='z')

    m.addConstrs((quicksum(A[i - 1, j - 1] * x[j] for j in range(1, lin_var + 1)) + r * z[i] >= r
                  for i in range(1, 1 + scenarios)), name=scenarios)
    m.addConstr(quicksum(z[i] for i in range(1, 1 + scenarios)) <= k, name='k_violation')

    if budget_con:
        m.addConstr(quicksum(x[i] for i in range(1, 1 + lin_var)) == 1, name='budget')

    m.ModelSense = -1
    # m.setParam('LogToConsole', 0)
    m.update()

    return m


def CCPPower(parameters):
    # bertsimas pg 10 (Multiperiod planning of electric p ower capacity) + Simge ipco'21 talk pg 75
    time_periods, scenarios = parameters

    coal_plant_life = 15
    nuc_plant_life = 10
    nuc_perc_lim = 0.2

    epsilon = 0.25  # 80% confidence
    scenario_prob = [1 / scenarios] * scenarios  # np.random.randint(10, 21, scenarios) and divide by sum

    coal_capex_perMW = np.random.randint(200, 300, time_periods).astype('float')
    nuc_capex_perMW = np.random.randint(100, 200, time_periods).astype('float')

    existing_cap = np.zeros(time_periods).astype('float')
    C = np.random.randint(100, 500)
    f = np.random.randint(70, 100) / 100
    for i in range(round(time_periods / 2)):
        existing_cap[i] = C * (f ** i)

    demand_t_s = np.random.randint(300, 700, (time_periods, scenarios)).astype('float')

    m = gp.Model()

    new_coal = m.addVars(range(1, 1 + time_periods), lb=0, obj=-coal_capex_perMW, vtype='C', name='new_coal')
    new_nuc = m.addVars(range(1, 1 + time_periods), lb=0, obj=-nuc_capex_perMW, vtype='C', name='new_nuc')
    tot_coal = m.addVars(range(1, 1 + time_periods), lb=0, obj=0, vtype='C', name='tot_coal')
    tot_nuc = m.addVars(range(1, 1 + time_periods), lb=0, obj=0, vtype='C', name='tot_nuc')
    y = m.addVars(range(1, 1 + time_periods), lb=0, obj=0, vtype='C', name='y')
    z = m.addVars(range(1, 1 + scenarios), lb=0, ub=1, obj=0, vtype='B', name='z')

    m.addConstrs((tot_coal[i] - quicksum(new_coal[j] for j in range(max(1, i + 1 - coal_plant_life), i + 1)) == 0
                  for i in range(1, time_periods + 1)), name='account_coal')
    m.addConstrs((tot_nuc[i] - quicksum(new_nuc[j] for j in range(max(1, i + 1 - nuc_plant_life), i + 1)) == 0
                  for i in range(1, time_periods + 1)), name='account_nuc')
    m.addConstrs(((1 - nuc_perc_lim) * tot_nuc[i] - nuc_perc_lim * tot_coal[i] <= nuc_perc_lim * existing_cap[i - 1]
                  for i in range(1, time_periods + 1)), name='nuc_lim')
    m.addConstrs((y[i] - tot_coal[i] - tot_nuc[i] == existing_cap[i - 1]
                  for i in range(1, time_periods + 1)), name='total_power')
    m.addConstrs((y[t + 1] + demand_t_s[t, s] * z[s + 1] >= demand_t_s[t, s]
                  for t in range(time_periods) for s in range(scenarios)), name='mixing')
    m.addConstr(quicksum(z[s + 1] * scenario_prob[s] for s in range(scenarios)) <= epsilon, name='allowed_violation')

    m.ModelSense = -1
    # m.setParam('LogToConsole', 0)
    m.update()

    return m


# NETWORK
def FixedChargeFlowProblem(parameters):
    # bin vars = num_edges
    num_nodes, num_edges, num_commodities = parameters  # edges > 2 * nodes

    # G = nx.random_regular_graph(2, num_nodes, 0)
    # nodes = np.fromiter(G.nodes, dtype=int)
    #
    # while G.number_of_edges() < num_edges:
    #     new_edge = np.random.choice(nodes, 2, False)
    #     G.add_edge(*new_edge)
    #
    # while not nx.is_connected(G):
    #
    #     components = list(nx.connected_components(G))
    #
    #     select_components = np.random.choice(len(components), 2, False)
    #
    #     edge_set_1 = G.edges(components[select_components[0]])
    #     edge_set_2 = G.edges(components[select_components[1]])
    #
    #     edge_1 = edge_set_1[np.random.choice(len(edge_set_1))]
    #     edge_2 = edge_set_2[np.random.choice(len(edge_set_2))]
    #
    #     if np.random.choice(2):
    #         new_edge_1 = (edge_1[0], edge_2[0])
    #         new_edge_2 = (edge_1[1], edge_2[1])
    #     else:
    #         new_edge_1 = (edge_1[0], edge_2[1])
    #         new_edge_2 = (edge_1[0], edge_2[1])
    #
    #     G.add_edges_from([new_edge_1, new_edge_2])
    #     G.remove_edges_from([edge_1, edge_2])
    #
    # for _ in range(5):
    #
    #     edge_set = G.edges()
    #     edges = list(edge_set)
    #
    #     select_edges = np.random.choice(len(edge_set), 2, False)
    #
    #     edge_1 = edges[select_edges[0]]
    #     edge_2 = edges[select_edges[1]]
    #
    #     if np.random.choice(2):
    #         new_edge_1 = (edge_1[0], edge_2[0])
    #         new_edge_2 = (edge_1[1], edge_2[1])
    #     else:
    #         new_edge_1 = (edge_1[0], edge_2[1])
    #         new_edge_2 = (edge_1[1], edge_2[0])
    #
    #     if new_edge_1 not in edge_set and new_edge_2 not in edge_set and \
    #             new_edge_1[0] != new_edge_1[1] and new_edge_2[0] != new_edge_2[1]:
    #         G.remove_edges_from([edge_1, edge_2])
    #         G.add_edges_from([new_edge_1, new_edge_2])
    #
    #         if not nx.is_connected(G):
    #             G.remove_edges_from([new_edge_1, new_edge_2])
    #             G.add_edges_from([edge_1, edge_2])

    nodes = range(num_nodes)
    edges = get_random_geometric_graph(num_nodes, num_edges)

    source_sinks = np.random.choice(nodes, 2 * num_commodities, False)
    sources = source_sinks[:num_commodities]
    sinks = source_sinks[num_commodities:]
    if num_commodities == 3:
        demands = np.random.randint(100, 300, num_commodities)

    unit_costs = np.random.randint(3, 10, num_edges)
    fixed_costs = np.random.randint(3 * np.sum(demands), 8 * np.sum(demands), num_edges)
    capacities = np.random.randint(30 * num_commodities, 80 * num_commodities, num_edges)

    # edges = list(G.edges())

    commodities = range(num_commodities)
    m = gp.Model()

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
        y[e] = m.addVar(lb=0, ub=1, vtype='B', name=f"y({e[0]},{e[1]})")  # whether edge e is used or not

    m.setObjective(quicksum(
        unit_costs[i] * demands[k] * z[(k, edges[i])] for i in range(num_edges) for k in commodities) + quicksum(
        fixed_costs[i] * y[edges[i]] for i in range(num_edges)))

    for e in edges:
        for k in commodities:
            m.addConstr(z[(k, e)] >= x[(k, e)], name=f"z_bound_pos_{k}_({e[0]},{e[1]})")
            m.addConstr(z[(k, e)] >= -x[(k, e)], name=f"z_bound_neg_{k}_({e[0]},{e[1]})")

    for v in nodes:
        out_edges = [e for e in edges if e[0] == v]
        in_edges = [e for e in edges if e[1] == v]

        for k in commodities:
            m.addConstr(quicksum(x[(k, e)] for e in out_edges) - quicksum(x[(k, e)] for e in in_edges)
                        == delta[k, v], name=f"flow_conservation({v},{k})")

    for i in range(num_edges):
        m.addConstr(quicksum(demands[k] * z[(k, edges[i])] for k in commodities) <= capacities[i] * y[edges[i]],
                    name=f"capacity_({edges[i][0]},{edges[i][1]})")

    m.ModelSense = 1
    # m.setParam('LogToConsole', 0)
    m.update()

    return m


def LotSizing(parameters):
    time_periods, capacitated = parameters

    d = np.random.randint(50, 100, time_periods)
    p_cost = np.random.randint(-10, -1, time_periods)
    f_cost = np.random.randint(-20*time_periods, -10*time_periods, time_periods)
    h_cost = np.random.randint(-10, -1, time_periods - 1)

    if capacitated == 0:
        prod_cap = np.array([sum(d[i:]) for i in range(time_periods)])
    else:
        prod_cap = np.random.randint(150, 250, time_periods).astype(float)  # np.ceil(2*demand.mean())


    m = gp.Model()

    x = m.addVars(range(1, time_periods + 1), lb=0, obj=p_cost, vtype='C', name='x')
    y = m.addVars(range(1, time_periods + 1), lb=0, ub=1, obj=f_cost, vtype='B', name='y')
    s = m.addVars(range(1, time_periods), lb=0, obj=h_cost, vtype='C', name='s')

    m.addConstr(x[1] - s[1] == d[0], name='balance_1')
    m.addConstrs((s[i - 1] + x[i] - s[i] == d[i - 1] for i in range(2, time_periods)), name='balance')
    m.addConstr(s[time_periods - 1] + x[time_periods] == d[time_periods - 1], name='balance_n')

    m.addConstrs((x[i] - prod_cap[i - 1] * y[i] <= 0 for i in range(1, time_periods + 1)), 'x_y_relation')

    m.ModelSense = -1
    # m.setParam('LogToConsole', 0)
    m.update()

    return m


def LotSizingPathological(parameters):
    time_periods, capacitated = parameters

    # d = np.random.randint(50, 100, time_periods)
    # p_cost = np.random.randint(-10, -1, time_periods)
    # f_cost = np.random.randint(-400, -200, time_periods)
    # h_cost = np.random.randint(-10, -1, time_periods - 1)

    d = np.ones(time_periods).astype(float)
    p_cost = -np.array([time_periods - i for i in range(time_periods)]).astype(float)
    f_cost = -np.ones(time_periods).astype(float)
    h_cost = -np.zeros(time_periods - 1).astype(float)

    if capacitated == 0:
        prod_cap = np.array([sum(d[i:]) for i in range(time_periods)])
    else:
        prod_cap = np.random.randint(150, 250, time_periods).astype(float)  # np.ceil(2*demand.mean())

    m = gp.Model()

    x = m.addVars(range(1, time_periods + 1), lb=0, obj=p_cost, vtype='C', name='x')
    y = m.addVars(range(1, time_periods + 1), lb=0, ub=1, obj=f_cost, vtype='B', name='y')
    s = m.addVars(range(1, time_periods), lb=0, obj=h_cost, vtype='C', name='s')

    m.addConstr(x[1] - s[1] == d[0], name='balance_1')
    m.addConstrs((s[i - 1] + x[i] - s[i] == d[i - 1] for i in range(2, time_periods)), name='balance')
    m.addConstr(s[time_periods - 1] + x[time_periods] == d[time_periods - 1], name='balance_n')

    m.addConstrs((x[i] - prod_cap[i - 1] * y[i] <= 0 for i in range(1, time_periods + 1)), 'x_y_relation')

    m.ModelSense = -1
    # m.setParam('LogToConsole', 0)
    m.update()

    return m


def BigBucketLotSizing(parameters):
    time_periods, items = parameters

    d = np.random.randint(0, 100, (time_periods, items))
    p_cost = -np.zeros((time_periods, items))  # np.random.randint(-10, -1, (time_periods, items)).astype(float)
    f_cost = -np.random.randint(10*time_periods*items, 20*time_periods*items, (time_periods, items))
    h_cost = -np.random.randint(1, 10, (time_periods - 1, items))

    prod_cap = np.random.randint(1000 * items * time_periods/20, 2000 * items * time_periods/20, time_periods)
    setup_time = np.random.randint(200, 500, (time_periods, items))
    process_time = np.random.randint(1, 10, (time_periods, items))

    init_inv = np.random.randint(0, 200, items)

    m = gp.Model()

    x = m.addVars(range(1, 1 + time_periods), range(1, items + 1), lb=0, obj=p_cost, vtype='C', name='x')
    s = m.addVars(range(1, time_periods), range(1, items + 1), lb=0, obj=h_cost, vtype='C', name='s')
    y = m.addVars(range(1, 1 + time_periods), range(1, items + 1), lb=0, ub=1, obj=f_cost, vtype='B', name='y')

    m.addConstrs((x[1, j] - s[1, j] == d[0, j - 1] - init_inv[j - 1] for j in range(1, items + 1)), name='balance_1')
    m.addConstrs((s[i - 1, j] + x[i, j] - s[i, j] == d[i - 1, j - 1]
                  for i in range(2, time_periods) for j in range(1, items + 1)),
                 name='balance')
    m.addConstrs((s[time_periods - 1, j] + x[time_periods, j] == d[time_periods - 1, j - 1]
                  for j in range(1, items + 1)), name='balance_n')

    m.addConstrs((x[i, j] - d[i - 1:, j - 1].sum() * y[i, j] <= 0
                  for i in range(1, time_periods + 1) for j in range(1, 1 + items)), name='x_y_relation')
    m.addConstrs((quicksum(process_time[i - 1, j - 1] * x[i, j] + setup_time[i - 1, j - 1] * y[i, j]
                           for j in range(1, 1 + items)) <= prod_cap[i - 1]
                  for i in range(1, 1 + time_periods)), name='prod_cap')

    m.ModelSense = -1
    # m.setParam('LogToConsole', 0)
    m.update()

    return m


def CLSPTrigeiro(parameters):
    time_periods, items, demand_var, proc_time_var, setup_time_mean, setup_time_range, \
    f_to_h_mean, f_h_range, cap_utilization = parameters

    demand_mean = 100
    proc_time_mean = 1

    demand_var_map = {'low': 0.09, 'medium': 0.244, 'high': 0.57}
    proc_time_var_map = {'none': (1.0, 1.0), 'high': (0.5, 1.5)}
    setup_time_range_map = {'low': 0.2667, 'high': 1.333}
    f_h_range_map = {'low': 0.2667, 'high': 1.333}
    cap_utilization_map = {'low': 0.75, 'medium': 1.0, 'high': 1.1}

    # demand
    demand_range = np.sqrt(12) * demand_var_map[demand_var] * demand_mean

    d = np.random.randint(np.floor(demand_mean - 0.5 * demand_range), np.ceil(demand_mean + 0.5 * demand_range),
                          (time_periods, items)).astype(np.float)

    a = [list(product(range(4), range(items)))[i] for i in np.random.choice(range(4 * items), items)]
    for i, j in a:
        d[i, j] = 0
    d[4:, :] += demand_mean * 0.25 * 4 / (time_periods - 4)
    init_inv = np.zeros(items)

    # cost
    p_cost = -np.zeros((time_periods, items)).astype(float)
    h_cost = -np.tile(np.random.uniform(1 * (1 - 0.5 * f_h_range_map[f_h_range]),
                                        1 * (1 + 0.5 * f_h_range_map[f_h_range]),
                                        items),  # h
                      (time_periods - 1, 1))
    f_cost = -np.tile(np.random.uniform(f_to_h_mean * 1 * (1 - 0.5 * f_h_range_map[f_h_range]),
                                        f_to_h_mean * 1 * (1 + 0.5 * f_h_range_map[f_h_range]),
                                        items),  # f
                      (time_periods, 1))

    # time
    proc_time_lb, proc_time_ub = (proc_time_mean * i for i in proc_time_var_map[proc_time_var])
    process_time = np.tile(np.random.uniform(proc_time_ub, proc_time_lb, items).astype(float),
                           (time_periods, 1))
    setup_time = np.tile(np.random.uniform(setup_time_mean * (1 - 0.5 * setup_time_range_map[setup_time_range]),
                                           setup_time_mean * (1 + 0.5 * setup_time_range_map[setup_time_range]),
                                           items).astype(float),
                         (time_periods, 1))

    lot_for_lot_cap_mean = (setup_time + process_time * d).sum(axis=1).mean()
    prod_cap = np.ones(time_periods).astype(float) * lot_for_lot_cap_mean / cap_utilization_map[cap_utilization]

    m = gp.Model()

    x = m.addVars(range(1, 1 + time_periods), range(1, items + 1), lb=0, obj=p_cost, vtype='C', name='x')
    s = m.addVars(range(1, time_periods), range(1, items + 1), lb=0, obj=h_cost, vtype='C', name='s')
    y = m.addVars(range(1, 1 + time_periods), range(1, items + 1), lb=0, ub=1, obj=f_cost, vtype='B', name='y')

    m.addConstrs((x[1, j] - s[1, j] == d[0, j - 1] - init_inv[j - 1] for j in range(1, items + 1)), name='balance_1')
    m.addConstrs((s[i - 1, j] + x[i, j] - s[i, j] == d[i - 1, j - 1]
                  for i in range(2, time_periods) for j in range(1, items + 1)),
                 name='balance')
    m.addConstrs((s[time_periods - 1, j] + x[time_periods, j] == d[time_periods - 1, j - 1]
                  for j in range(1, items + 1)), name='balance_n')

    m.addConstrs((x[i, j] - d[i - 1:, j - 1].sum() * y[i, j] <= 0
                  for i in range(1, time_periods + 1) for j in range(1, 1 + items)), name='x_y_relation')
    m.addConstrs((quicksum(process_time[i - 1, j - 1] * x[i, j] + setup_time[i - 1, j - 1] * y[i, j]
                           for j in range(1, 1 + items)) <= prod_cap[i - 1]
                  for i in range(1, 1 + time_periods)), name='prod_cap')

    m.ModelSense = -1
    # m.setParam('LogToConsole', 0)
    m.update()

    return m


def CLSPSural(parameters):
    time_periods, items, demand_var, proc_time_var, setup_time_mean, setup_time_range, \
    f_to_h_mean, f_h_range, cap_utilization = parameters

    demand_mean = 100
    proc_time_mean = 1

    demand_var_map = {'low': 0.09, 'medium': 0.244, 'high': 0.57}
    proc_time_var_map = {'none': (1.0, 1.0), 'high': (0.5, 1.5)}
    setup_time_range_map = {'low': 0.2667, 'high': 1.333}
    f_h_range_map = {'low': 0.2667, 'high': 1.333}
    cap_utilization_map = {'low': 0.75, 'medium': 1.0, 'high': 1.1}

    # demand
    demand_range = np.sqrt(12) * demand_var_map[demand_var] * demand_mean

    d = np.random.randint(np.floor(demand_mean - 0.5 * demand_range), np.ceil(demand_mean + 0.5 * demand_range),
                          (time_periods, items)).astype(np.float)

    a = [list(product(range(4), range(items)))[i] for i in np.random.choice(range(4 * items), items)]
    for i, j in a:
        d[i, j] = 2
    d[4:, :] += demand_mean * 0.25 * 4 / (time_periods - 4)
    init_inv = np.zeros(items)

    # cost
    p_cost = -np.zeros((time_periods, items)).astype(float)
    h_cost = -np.tile(np.random.uniform(1 * (1 - 0.5 * f_h_range_map[f_h_range]),
                                        1 * (1 + 0.5 * f_h_range_map[f_h_range]),
                                        items),  # h
                      (time_periods - 1, 1))
    f_cost = -np.tile(np.random.uniform(0, 0, items),  # f
                      (time_periods, 1))

    # time
    proc_time_lb, proc_time_ub = (proc_time_mean * i for i in proc_time_var_map[proc_time_var])
    process_time = np.tile(np.random.uniform(proc_time_ub, proc_time_lb, items).astype(float),
                           (time_periods, 1))
    setup_time = np.tile(np.random.uniform(setup_time_mean * (1 - 0.5 * setup_time_range_map[setup_time_range]),
                                           setup_time_mean * (1 + 0.5 * setup_time_range_map[setup_time_range]),
                                           items).astype(float),
                         (time_periods, 1))

    lot_for_lot_cap_mean = (setup_time + process_time * d).sum(axis=1).mean()
    prod_cap = np.ones(time_periods).astype(float) * lot_for_lot_cap_mean / cap_utilization_map[cap_utilization]

    m = gp.Model()

    x = m.addVars(range(1, 1 + time_periods), range(1, items + 1), lb=0, obj=p_cost, vtype='C', name='x')
    s = m.addVars(range(1, time_periods), range(1, items + 1), lb=0, obj=h_cost, vtype='C', name='s')
    y = m.addVars(range(1, 1 + time_periods), range(1, items + 1), lb=0, ub=1, obj=f_cost, vtype='B', name='y')

    m.addConstrs((x[1, j] - s[1, j] == d[0, j - 1] - init_inv[j - 1] for j in range(1, items + 1)),
                 name='balance_1')
    m.addConstrs((s[i - 1, j] + x[i, j] - s[i, j] == d[i - 1, j - 1]
                  for i in range(2, time_periods) for j in range(1, items + 1)),
                 name='balance')
    m.addConstrs((s[time_periods - 1, j] + x[time_periods, j] == d[time_periods - 1, j - 1]
                  for j in range(1, items + 1)), name='balance_n')

    m.addConstrs((x[i, j] - d[i - 1:, j - 1].sum() * y[i, j] <= 0
                  for i in range(1, time_periods + 1) for j in range(1, 1 + items)), name='x_y_relation')
    m.addConstrs((quicksum(process_time[i - 1, j - 1] * x[i, j] + setup_time[i - 1, j - 1] * y[i, j]
                           for j in range(1, 1 + items)) <= prod_cap[i - 1]
                  for i in range(1, 1 + time_periods)), name='prod_cap')

    m.ModelSense = -1
    # m.setParam('LogToConsole', 0)
    m.update()

    return m


def TravelingSalesmanProblem(parameters):
    # Dantzig-Fulkerson-Johnson
    n_nodes, = parameters
    nodes = list(range(n_nodes))

    # pos = np.random.uniform(0, 1, (n_nodes, 2)) #+ (1j * np.random.uniform(0, 1, (n_nodes, 1)))
    pos = np.random.multivariate_normal([0, 0, 0], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], n_nodes)
    distance = distance_matrix(pos, pos, 2)

    arcs = list(permutations(nodes, 2))
    m = gp.Model()
    x = m.addVars([(i, j) for i, j in arcs], vtype='B',
                  obj=[distance[i, j] for i, j in arcs],
                  name=[f"x_{i}_{j}" for i, j in arcs], )

    m.addConstrs((quicksum(x[i, j] for j in nodes if j != i) == 1 for i in nodes), name="outgoing")
    m.addConstrs((quicksum(x[j, i] for j in nodes if j != i) == 1 for i in nodes), name="incoming")

    for k in range(2, n_nodes):
        for S in combinations(nodes, k):
            m.addConstr(quicksum(x[i, j] for i in S for j in S if i != j) <= k - 1)

    # m.ModelSense = 1
    # m.setParam('LogToConsole', 0)
    m.update()

    return m


# GRAPHS

def CutOffTU(parameters):  # independent set on bipartite graph + knapsack that cuts off optimal solution
    # m = number of constraints = number of edges = number of nodes - 1
    # n = number of s-t pairs <= m+1C2
    n_edges, n_nodes = parameters

    f = 0.5
    nodes1 = range(1, 1 + math.floor(f * n_nodes))
    nodes2 = range(1 + math.floor(f * n_nodes), n_nodes + 1)

    if n_edges == 'all':
        n_edges = len(nodes1) * len(nodes2)
        edges = list(product(nodes1, nodes2))
    else:
        edges = [list(product(nodes1, nodes2))[i] for i in
                 np.random.choice(range(len(nodes1) * len(nodes2)), n_edges, replace=False)]

    A = np.zeros((n_edges, n_nodes))
    for i in range(n_edges):
        n1, n2 = edges[i]
        A[i, [n1 - 1, n2 - 1]] = 1

    b = np.ones(n_edges)  # np.random.randint(0, 3, m).astype(float)

    c = np.random.randint(1, 100, n_nodes)

    m = gp.Model()

    x = m.addVars(range(1, n_nodes + 1), lb=0, ub=1, obj=c, vtype='B', name='node')

    m.addMConstr(A, x.values(), '<=', b, 'edge_con')

    m.ModelSense = -1
    # m.setParam('LogToConsole', 0)
    m.optimize()
    val = m.getAttr('ObjVal')

    r = 0.8
    m.addLConstr(quicksum(c[i] * x[i + 1] for i in range(n_nodes)), sense='<=', rhs=val * r, name='cut_off')
    m.update()

    return m


def get_random_geometric_graph(n_nodes, n_edges):
    pos = np.random.uniform(0, 1, (n_nodes, 2))
    # for i in range(n_nodes):
    #     for j in range(i + 1, n_nodes):
    #         if np.sqrt(np.sum((pos[i, :] - pos[j, :])**2)) <= d:
    #             edges.append((i, j))
    # n_edges = len(edges)

    dist = lambda p1, p2: np.sqrt(((p1 - p2) ** 2).sum())
    dist_edges = sorted([(dist(pos[i, :], pos[j, :]), i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)])
    edges = [(i, j) for _, i, j in dist_edges[: n_edges]]
    return edges


def get_random_erdos_renyi_graph(n_nodes, n_edges):
    # Erdos - Reyni
    nodes = range(n_nodes)
    all_edges = list(combinations(nodes, 2))
    edges = [all_edges[i] for i in np.random.choice(range(len(all_edges)), n_edges, replace=False)]
    return edges


def MaximumMatching(parameters):
    n_nodes, n_edges = parameters

    # Random Geometric
    edges = get_random_geometric_graph(n_nodes, n_edges)

    A = np.zeros((n_nodes, n_edges))
    for i, e in enumerate(edges):
        A[e[0], i] = 1
        A[e[1], i] = 1

    m = gp.Model()
    x_e = m.addVars(n_edges, vtype='B', obj=np.random.uniform(0.5, 1, n_edges))

    m.addMConstr(A, x_e.values(), "<=", np.full(n_nodes, 1), "matching")

    m.ModelSense = -1
    # m.setParam('LogToConsole', 0)
    m.update()

    return m


def MaximumIndependentSet(parameters):
    n_nodes, n_edges = parameters
    edges = get_random_geometric_graph(n_nodes, n_edges)

    m = gp.Model()
    x_n = m.addVars(n_nodes, vtype='B', obj=np.random.uniform(0.5, 1, n_nodes))

    for u, v in edges:
        m.addConstr(x_n[u] + x_n[v] <= 1)

    m.ModelSense = -1
    # m.setParam('LogToConsole', 0)
    m.update()

    return m


def PMedian(parameters):
    n_facilities, n_customers, p = parameters

    customers = range(n_customers)
    facilities = range(n_facilities)

    customer_pos = np.random.uniform(0, 1, (n_customers, 3))
    facility_pos = np.random.uniform(0.2, 0.8, (n_facilities, 3))
    demand = np.random.randint(10, 100, (n_customers, 1))

    distance = distance_matrix(customer_pos, facility_pos)
    cost = distance * demand

    mean_cost = np.mean(cost, axis=0)
    mean_cost /= np.min(mean_cost)

    capacity = np.random.uniform(0.5, 1., (n_facilities)) * mean_cost * demand.sum() / p
    m = gp.Model()

    x = m.addVars(n_customers, n_facilities, ub=1, obj=cost, name="x")
    y = m.addVars(n_facilities, vtype='B', name="y")

    m.addConstrs((x.sum(i, '*') >= 1 for i in customers), name='assignment')
    m.addConstrs((x[i, j] - y[j] <= 0 for i in customers for j in facilities), name='facility_open')
    m.addConstrs((quicksum(x[i, j] * demand[i] for i in customers) - capacity[j] * y[j] <= 0 for j in facilities),
                 name='facility_open')
    m.addConstr(y.sum('*') <= p, name="num_facilities")

    m.ModelSense = 1
    # m.setParam('LogToConsole', 0)
    m.update()

    return m


def SparsePackingIP(parameters):
    n, p, sparsity, k = parameters

    weights = np.random.randint(1, 10 * n, (p, n))
    p_matrix = np.random.rand(p, n)
    weights[p_matrix > sparsity] = 0

    capacities = np.ceil(0.99 * (k / n) * np.sum(weights, 1))

    values = np.random.randint(1, 10 * n + 1, n).astype(float)  # obj: value of items

    m = gp.Model()

    x = m.addVars(range(1, n + 1), lb=0, ub=1, obj=values, name='x')
    y = m.addVars(range(1, n + 1), lb=0, ub=1, obj=values, vtype='B', name='y')

    m.addMConstr(weights, x.values(), '<=', capacities, name='packing')
    m.addConstrs((x[i] - y[i] <= 0
                  for i in range(1, 1 + n)), name="switch_on_off")
    m.addConstr(y.sum('*') <= k, name="sparse_solution")

    m.ModelSense = -1
    # m.setParam('LogToConsole', 0)
    m.update()

    return m


def SparseCoveringIP(parameters):
    n, p, sparsity, k = parameters

    weights = np.random.randint(1, 10 * n, (p, n))
    p_matrix = np.random.rand(p, n)
    weights[p_matrix > sparsity] = 0

    capacities = np.ceil(0.01 * (k / n) * np.sum(weights, 1))

    values = np.random.randint(1, 10 * n + 1, n).astype(float)  # obj: value of items

    m = gp.Model()

    x = m.addVars(range(1, n + 1), lb=0, ub=1, obj=values, name='x')
    y = m.addVars(range(1, n + 1), lb=0, ub=1, obj=values, vtype='B', name='y')

    m.addMConstr(weights, x.values(), '>=', capacities, name='packing')
    m.addConstrs((x[i] - y[i] <= 0
                  for i in range(1, 1 + n)), name="switch_on_off")
    m.addConstr(y.sum('*') <= k, name="sparse_solution")

    m.ModelSense = 1
    # m.setParam('LogToConsole', 0)
    m.update()

    return m


# Gasse et al instances

class Graph:
    """
    Container for a graph.

    Parameters
    ----------
    number_of_nodes : int
        The number of nodes in the graph.
    edges : set of tuples (int, int)
        The edges of the graph, where the integers refer to the nodes.
    degrees : numpy array of integers
        The degrees of the nodes in the graph.
    neighbors : dictionary of type {int: set of ints}
        The neighbors of each node in the graph.
    """
    def __init__(self, number_of_nodes, edges, degrees, neighbors):
        self.number_of_nodes = number_of_nodes
        self.edges = edges
        self.degrees = degrees
        self.neighbors = neighbors

    def __len__(self):
        """
        The number of nodes in the graph.
        """
        return self.number_of_nodes

    def greedy_clique_partition(self):
        """
        Partition the graph into cliques using a greedy algorithm.

        Returns
        -------
        list of sets
            The resulting clique partition.
        """
        cliques = []
        leftover_nodes = (-self.degrees).argsort().tolist()

        while leftover_nodes:
            clique_center, leftover_nodes = leftover_nodes[0], leftover_nodes[1:]
            clique = {clique_center}
            neighbors = self.neighbors[clique_center].intersection(leftover_nodes)
            densest_neighbors = sorted(neighbors, key=lambda x: -self.degrees[x])
            for neighbor in densest_neighbors:
                # Can you add it to the clique, and maintain cliqueness?
                if all([neighbor in self.neighbors[clique_node] for clique_node in clique]):
                    clique.add(neighbor)
            cliques.append(clique)
            leftover_nodes = [node for node in leftover_nodes if node not in clique]

        return cliques

    @staticmethod
    def erdos_renyi(number_of_nodes, edge_probability):
        """
        Generate an Erdös-Rényi random graph with a given edge probability.

        Parameters
        ----------
        number_of_nodes : int
            The number of nodes in the graph.
        edge_probability : float in [0,1]
            The probability of generating each edge.
        random : numpy.random.RandomState
            A random number generator.

        Returns
        -------
        Graph
            The generated graph.
        """
        edges = set()
        degrees = np.zeros(number_of_nodes, dtype=int)
        neighbors = {node: set() for node in range(number_of_nodes)}
        for edge in combinations(np.arange(number_of_nodes), 2):
            if np.random.uniform() < edge_probability:
                edges.add(edge)
                degrees[edge[0]] += 1
                degrees[edge[1]] += 1
                neighbors[edge[0]].add(edge[1])
                neighbors[edge[1]].add(edge[0])
        graph = Graph(number_of_nodes, edges, degrees, neighbors)
        return graph

    @staticmethod
    def barabasi_albert(number_of_nodes, affinity):
        """
        Generate a Barabási-Albert random graph with a given edge probability.

        Parameters
        ----------
        number_of_nodes : int
            The number of nodes in the graph.
        affinity : integer >= 1
            The number of nodes each new node will be attached to, in the sampling scheme.
        random : numpy.random.RandomState
            A random number generator.

        Returns
        -------
        Graph
            The generated graph.
        """
        assert affinity >= 1 and affinity < number_of_nodes

        edges = set()
        degrees = np.zeros(number_of_nodes, dtype=int)
        neighbors = {node: set() for node in range(number_of_nodes)}
        for new_node in range(affinity, number_of_nodes):
            # first node is connected to all previous ones (star-shape)
            if new_node == affinity:
                neighborhood = np.arange(new_node)
            # remaining nodes are picked stochastically
            else:
                neighbor_prob = degrees[:new_node] / (2*len(edges))
                neighborhood = np.random.choice(new_node, affinity, replace=False, p=neighbor_prob)
            for node in neighborhood:
                edges.add((node, new_node))
                degrees[node] += 1
                degrees[new_node] += 1
                neighbors[node].add(new_node)
                neighbors[new_node].add(node)

        graph = Graph(number_of_nodes, edges, degrees, neighbors)
        return graph


def GasseIndSet(parameters):
    """
    Generate a Maximum Independent Set (also known as Maximum Stable Set) instance
    in CPLEX LP format from a previously generated graph.

    Parameters
    ----------
    graph : Graph
        The graph from which to build the independent set problem.
    filename : str
        Path to the file to save.
    """

    nnodes, affinity = parameters
    graph = Graph.barabasi_albert(nnodes, affinity)

    cliques = graph.greedy_clique_partition()
    inequalities = set(graph.edges)
    for clique in cliques:
        clique = tuple(sorted(clique))
        for edge in combinations(clique, 2):
            inequalities.remove(edge)
        if len(clique) > 1:
            inequalities.add(clique)

    # Put trivial inequalities for nodes that didn't appear
    # in the constraints, otherwise SCIP will complain
    used_nodes = set()
    for group in inequalities:
        used_nodes.update(group)
    for node in range(10):
        if node not in used_nodes:
            inequalities.add((node,))

    m = gp.Model()

    x = m.addVars(range(1, nnodes + 1), lb=0, ub=1, obj=1, vtype='B', name='x')
    m.addConstrs((quicksum(x[j + 1] for j in sorted(group)) <= 1
                  for group in inequalities), name="cliques")


    m.ModelSense = -1
    # m.setParam('LogToConsole', 0)
    m.update()


    # with open(filename, 'w') as lp_file:
    #     lp_file.write("maximize\nOBJ:" + "".join([f" + 1 x{node+1}" for node in range(len(graph))]) + "\n")
    #     lp_file.write("\nsubject to\n")
    #     for count, group in enumerate(inequalities):
    #         lp_file.write(f"C{count+1}:" + "".join([f" + x{node+1}" for node in sorted(group)]) + " <= 1\n")
    #     lp_file.write("\nbinary\n" + " ".join([f"x{node+1}" for node in range(len(graph))]) + "\n")
    return m


def GasseSetCover(parameters):

    """
    Generates a setcover instance with specified characteristics, and writes
    it to a file in the LP format.

    Approach described in:
    E.Balas and A.Ho, Set covering algorithms using cutting planes, heuristics,
    and subgradient optimization: A computational study, Mathematical
    Programming, 12 (1980), 37-60.

    Parameters
    ----------
    nrows : int
        Desired number of rows
    ncols : int
        Desired number of columns
    density: float between 0 (excluded) and 1 (included)
        Desired density of the constraint matrix
    filename: str
        File to which the LP will be written
    rng: numpy.random.RandomState
        Random number generator
    max_coef: int
        Maximum objective coefficient (>=1)
    """

    nrows, ncols, density = parameters
    max_coef = 100

    nnzrs = int(nrows * ncols * density)

    assert nnzrs >= nrows  # at least 1 col per row
    assert nnzrs >= 2 * ncols  # at leats 2 rows per col

    # compute number of rows per column
    indices = np.random.choice(ncols, size=nnzrs)  # random column indexes
    indices[:2 * ncols] = np.repeat(np.arange(ncols), 2)  # force at leats 2 rows per col
    _, col_nrows = np.unique(indices, return_counts=True)

    # for each column, sample random rows
    indices[:nrows] = np.random.permutation(nrows) # force at least 1 column per row
    i = 0
    indptr = [0]
    for n in col_nrows:

        # empty column, fill with random rows
        if i >= nrows:
            indices[i:i+n] = np.random.choice(nrows, size=n, replace=False)

        # partially filled column, complete with random rows among remaining ones
        elif i + n > nrows:
            remaining_rows = np.setdiff1d(np.arange(nrows), indices[i:nrows], assume_unique=True)
            indices[nrows:i+n] = np.random.choice(remaining_rows, size=i+n-nrows, replace=False)

        i += n
        indptr.append(i)

    # objective coefficients
    c = np.random.randint(max_coef, size=ncols) + 1

    # sparce CSC to sparse CSR matrix
    A = scipy.sparse.csc_matrix(
            (np.ones(len(indices), dtype=int), indices, indptr),
            shape=(nrows, ncols)).tocsr()
    indices = A.indices
    indptr = A.indptr

    # write problem
    m = gp.Model()

    x = m.addVars(range(1, ncols + 1), lb=0, ub=1, obj=c, vtype='B', name='x')
    m.addConstrs((quicksum(x[j+1] for j in indices[indptr[i]:indptr[i+1]]) >= 1
                  for i in range(nrows)), name="covering")
    m.ModelSense = 1
    # m.setParam('LogToConsole', 0)
    m.update()


    # with open(filename, 'w') as file:
    #     file.write("minimize\nOBJ:")
    #     file.write("".join([f" +{c[j]} x{j+1}" for j in range(ncols)]))
    #
    #     file.write("\n\nsubject to\n")
    #     for i in range(nrows):
    #         row_cols_str = "".join([f" +1 x{j+1}" for j in indices[indptr[i]:indptr[i+1]]])
    #         file.write(f"C{i}:" + row_cols_str + f" >= 1\n")
    #
    #     file.write("\nbinary\n")
    #     file.write("".join([f" x{j+1}" for j in range(ncols)]))
    return m


def GasseCombAuctions(parameters):
    """
    Generate a Combinatorial Auction problem following the 'arbitrary' scheme found in section 4.3. of
        Kevin Leyton-Brown, Mark Pearson, and Yoav Shoham. (2000).
        Towards a universal test suite for combinatorial auction algorithms.
        Proceedings of ACM Conference on Electronic Commerce (EC-00) 66-76.

    Saves it as a CPLEX LP file.

    Parameters
    ----------
    random : numpy.random.RandomState
        A random number generator.
    filename : str
        Path to the file to save.
    n_items : int
        The number of items.
    n_bids : int
        The number of bids.
    min_value : int
        The minimum resale value for an item.
    max_value : int
        The maximum resale value for an item.
    value_deviation : int
        The deviation allowed for each bidder's private value of an item, relative from max_value.
    add_item_prob : float in [0, 1]
        The probability of adding a new item to an existing bundle.
    max_n_sub_bids : int
        The maximum number of substitutable bids per bidder (+1 gives the maximum number of bids per bidder).
    additivity : float
        Additivity parameter for bundle prices. Note that additivity < 0 gives sub-additive bids, while additivity > 0 gives super-additive bids.
    budget_factor : float
        The budget factor for each bidder, relative to their initial bid's price.
    resale_factor : float
        The resale factor for each bidder, relative to their initial bid's resale value.
    integers : logical
        Should bid's prices be integral ?
    warnings : logical
        Should warnings be printed ?
    """

    min_value = 1
    max_value = 100
    value_deviation = 0.5
    add_item_prob = 0.7
    max_n_sub_bids = 5
    additivity = 0.2
    budget_factor = 1.5
    resale_factor = 0.5
    integers = False
    warnings = False

    n_items, n_bids = parameters

    assert min_value >= 0 and max_value >= min_value
    assert add_item_prob >= 0 and add_item_prob <= 1

    def choose_next_item(bundle_mask, interests, compats):
        n_items = len(interests)
        prob = (1 - bundle_mask) * interests * compats[bundle_mask, :].mean(axis=0)
        prob /= prob.sum()
        return np.random.choice(n_items, p=prob)

    # common item values (resale price)
    values = min_value + (max_value - min_value) * np.random.rand(n_items)

    # item compatibilities
    compats = np.triu(np.random.rand(n_items, n_items), k=1)
    compats = compats + compats.transpose()
    compats = compats / compats.sum(1)

    bids = []
    n_dummy_items = 0

    # create bids, one bidder at a time
    while len(bids) < n_bids:

        # bidder item values (buy price) and interests
        private_interests = np.random.rand(n_items)
        private_values = values + max_value * value_deviation * (2 * private_interests - 1)

        # substitutable bids of this bidder
        bidder_bids = {}

        # generate initial bundle, choose first item according to bidder interests
        prob = private_interests / private_interests.sum()
        item = np.random.choice(n_items, p=prob)
        bundle_mask = np.full(n_items, 0)
        bundle_mask[item] = 1

        # add additional items, according to bidder interests and item compatibilities
        while np.random.rand() < add_item_prob:
            # stop when bundle full (no item left)
            if bundle_mask.sum() == n_items:
                break
            item = choose_next_item(bundle_mask, private_interests, compats)
            bundle_mask[item] = 1

        bundle = np.nonzero(bundle_mask)[0]

        # compute bundle price with value additivity
        price = private_values[bundle].sum() + np.power(len(bundle), 1 + additivity)
        if integers:
            price = int(price)

        # drop negativaly priced bundles
        if price < 0:
            if warnings:
                print("warning: negatively priced bundle avoided")
            continue

        # bid on initial bundle
        bidder_bids[frozenset(bundle)] = price

        # generate candidates substitutable bundles
        sub_candidates = []
        for item in bundle:

            # at least one item must be shared with initial bundle
            bundle_mask = np.full(n_items, 0)
            bundle_mask[item] = 1

            # add additional items, according to bidder interests and item compatibilities
            while bundle_mask.sum() < len(bundle):
                item = choose_next_item(bundle_mask, private_interests, compats)
                bundle_mask[item] = 1

            sub_bundle = np.nonzero(bundle_mask)[0]

            # compute bundle price with value additivity
            sub_price = private_values[sub_bundle].sum() + np.power(len(sub_bundle), 1 + additivity)
            if integers:
                sub_price = int(sub_price)

            sub_candidates.append((sub_bundle, sub_price))

        # filter valid candidates, higher priced candidates first
        budget = budget_factor * price
        min_resale_value = resale_factor * values[bundle].sum()
        for bundle, price in [
            sub_candidates[i] for i in np.argsort([-price for bundle, price in sub_candidates])]:

            if len(bidder_bids) >= max_n_sub_bids + 1 or len(bids) + len(bidder_bids) >= n_bids:
                break

            if price < 0:
                if warnings:
                    print("warning: negatively priced substitutable bundle avoided")
                continue

            if price > budget:
                if warnings:
                    print("warning: over priced substitutable bundle avoided")
                continue

            if values[bundle].sum() < min_resale_value:
                if warnings:
                    print("warning: substitutable bundle below min resale value avoided")
                continue

            if frozenset(bundle) in bidder_bids:
                if warnings:
                    print("warning: duplicated substitutable bundle avoided")
                continue

            bidder_bids[frozenset(bundle)] = price

        # add XOR constraint if needed (dummy item)
        if len(bidder_bids) > 2:
            dummy_item = [n_items + n_dummy_items]
            n_dummy_items += 1
        else:
            dummy_item = []

        # place bids
        for bundle, price in bidder_bids.items():
            bids.append((list(bundle) + dummy_item, price))

    bids_per_item = [[] for _ in range(n_items + n_dummy_items)]
    for i, bid in enumerate(bids):
        bundle, price = bid
        for item in bundle:
            bids_per_item[item].append(i)

    m = gp.Model()
    x = m.addVars(range(1, len(bids) + 1), lb=0, ub=1, obj=[bid[1] for bid in bids], vtype='B', name='x')

    for con_i, item_bids in enumerate(bids_per_item):
        if len(item_bids) > 0:
            m.addConstr(quicksum(x[i + 1] for i in item_bids) <= 1, name=f"item_{con_i}")

    m.ModelSense = -1
    # m.setParam('LogToConsole', 0)
    m.update()

    return m

    # generate the LP file
    # with open(filename, 'w') as file:
    #     bids_per_item = [[] for item in range(n_items + n_dummy_items)]
    #
    #     file.write("maximize\nOBJ:")
    #     for i, bid in enumerate(bids):
    #         bundle, price = bid
    #         file.write(f" +{price} x{i+1}")
    #         for item in bundle:
    #             bids_per_item[item].append(i)
    #
    #     file.write("\n\nsubject to\n")
    #     for item_bids in bids_per_item:
    #         if item_bids:
    #             for i in item_bids:
    #                 file.write(f" +1 x{i+1}")
    #             file.write(f" <= 1\n")
    #
    #     file.write("\nbinary\n")
    #     for i in range(len(bids)):
    #         file.write(f" x{i+1}")


def GasseCapacitedFacilityLocation(parameters):
    """
    Generate a Capacited Facility Location problem following
        Cornuejols G, Sridharan R, Thizy J-M (1991)
        A Comparison of Heuristics and Relaxations for the Capacitated Plant Location Problem.
        European Journal of Operations Research 50:280-297.

    Saves it as a CPLEX LP file.

    Parameters
    ----------
    random : numpy.random.RandomState
        A random number generator.
    filename : str
        Path to the file to save.
    n_customers: int
        The desired number of customers.
    n_facilities: int
        The desired number of facilities.
    ratio: float
        The desired capacity / demand ratio.
    """

    n_customers, n_facilities, ratio = parameters

    c_x = np.random.rand(n_customers)
    c_y = np.random.rand(n_customers)

    f_x = np.random.rand(n_facilities)
    f_y = np.random.rand(n_facilities)

    demands = np.random.randint(5, 35+1, size=n_customers)
    capacities = np.random.randint(10, 160+1, size=n_facilities)
    fixed_costs = np.random.randint(100, 110+1, size=n_facilities) * np.sqrt(capacities) \
            + np.random.randint(90+1, size=n_facilities)
    fixed_costs = fixed_costs.astype(int)

    total_demand = demands.sum()
    total_capacity = capacities.sum()

    # adjust capacities according to ratio
    capacities = capacities * ratio * total_demand / total_capacity
    capacities = capacities.astype(int)
    total_capacity = capacities.sum()

    # transportation costs
    trans_costs = np.sqrt(
            (c_x.reshape((-1, 1)) - f_x.reshape((1, -1))) ** 2 \
            + (c_y.reshape((-1, 1)) - f_y.reshape((1, -1))) ** 2) * 10 * demands.reshape((-1, 1))

    # write problem

    m = gp.Model()
    x = m.addVars(range(1, n_customers + 1), range(1, n_facilities + 1), lb=0, ub=1, obj=trans_costs, name='x')
    y = m.addVars(range(1, n_facilities + 1), lb=0, ub=1, obj=fixed_costs, vtype='B', name='y')

    m.addConstrs((quicksum(x[i+1, j+1] for j in range(n_facilities)) >= 1
                  for i in range(n_customers)), name="demand")
    m.addConstrs((quicksum(demands[i]*x[i + 1, j + 1] for i in range(n_customers)) - capacities[j] * y[j+1] <= 0
                  for j in range(n_facilities)), name="capacity")

    # optional constraints for LP relaxation tightening
    m.addConstr(quicksum(capacities[j]*y[j+1] for j in range(n_facilities)) >= total_demand, name="total demand")
    m.addConstrs((x[i+1, j+1] <= y[j+1] for j in range(n_facilities) for i in range(n_customers)), name="affectation")

    m.ModelSense = 1
    # m.setParam('LogToConsole', 0)
    m.update()

    return m

    # with open(filename, 'w') as file:
    #     file.write("minimize\nobj:")
    #     file.write("".join([f" +{trans_costs[i, j]} x_{i+1}_{j+1}" for i in range(n_customers) for j in range(n_facilities)]))
    #     file.write("".join([f" +{fixed_costs[j]} y_{j+1}" for j in range(n_facilities)]))
    #
    #     file.write("\n\nsubject to\n")
    #     for i in range(n_customers):
    #         file.write(f"demand_{i+1}:" + "".join([f" -1 x_{i+1}_{j+1}" for j in range(n_facilities)]) + f" <= -1\n")
    #     for j in range(n_facilities):
    #         file.write(f"capacity_{j+1}:" + "".join([f" +{demands[i]} x_{i+1}_{j+1}" for i in range(n_customers)]) + f" -{capacities[j]} y_{j+1} <= 0\n")
    #
    #     # optional constraints for LP relaxation tightening
    #     file.write("total_capacity:" + "".join([f" -{capacities[j]} y_{j+1}" for j in range(n_facilities)]) + f" <= -{total_demand}\n")
    #     for i in range(n_customers):
    #         for j in range(n_facilities):
    #             file.write(f"affectation_{i+1}_{j+1}: +1 x_{i+1}_{j+1} -1 y_{j+1} <= 0")
    #
    #     file.write("\nbounds\n")
    #     for i in range(n_customers):
    #         for j in range(n_facilities):
    #             file.write(f"0 <= x_{i+1}_{j+1} <= 1\n")
    #
    #     file.write("\nbinary\n")
    #     file.write("".join([f" y_{j+1}" for j in range(n_facilities)]))