# Extract Khalil features

import gc
import numpy as np
from dataclasses import dataclass, field
from operator import itemgetter
from itertools import groupby

@dataclass
class DataCollection:
    n: int
    collect: bool
    predict: bool
    ml_model: object
    ml_label: str
    nodes_limit : int

    static_cols: list = field(
        default_factory=lambda: ['variable', 'cost_sign', 'cost_rel_pos', 'cost_rel_neg',
                                 'cost_rel_posbin', 'cost_rel_negbin'] +
                                ['constr_deg_{0}'.format(stat) for stat in
                                 ['min', 'median', 'max', 'mean', 'std']] +
                                ['{0}_{1}_M{2}_{3}'.format(f, stat, measure, sign)
                                 for measure in [1, 2] for sign in [1, 0] for stat in
                                 ['min', 'median', 'max', 'mean', 'std'] for f in
                                 ['sign', 'abs']] +
                                ['{0}_{1}_M3_{2}{3}'.format(f, stat, sign1, sign2)
                                 for sign1 in [1, 0] for sign2 in [1, 0] for stat in
                                 ['min', 'median', 'max', 'mean', 'std'] for f in
                                 ['sign', 'abs']]
    )

    dynamic_cols: list = field(
        default_factory=lambda: ['node', 'variable', 'frac_fixed_var', 'frac_up', 'frac_down',
                                 'reduced_cost', 'c_sa_up_sign', 'c_sa_down_sign',
                                 'c_sa_up_log_delta', 'c_sa_down_log_delta'] +
                                ['{0}_{1}_{2}'.format(stat, measure, branch) for measure in
                                 ['q_by_c_abs', 'pseudo_cost', 'q_rel'] for branch in range(2)
                                 for stat in
                                 ['min', 'max', 'mean', 'std', 'first_quartile', 'median', 'third_quartile']] +
                                ['active_con_{0}_{1}'.format(w, stat) for w in
                                 ['b', 'pi', 'A_sum', 'A_prosp_sum'] for stat in
                                 ['min', 'median', 'max', 'mean', 'std']] +
                                ['frac_active_con', 'frac_selected']
    )

    num_cons: int = 0
    senses: list = field(default_factory=list)
    A_matrix: np.ndarray = None
    c: list = field(default_factory=list)
    c_abs: np.ndarray = None
    c_bin: np.ndarray = None
    b: np.ndarray = None
    bin_index: list = field(default_factory=list)

    branch_on_int: bool = False
    var_selection_count: np.ndarray = None

    def __post_init__(self):
        if self.predict or self.collect:
            self.var_selection_count = np.zeros(self.n, dtype=int)
            self.nodes_list: list = []
            self.static_dataset: np.ndarray = np.zeros((self.n, len(self.static_cols)))
            self.dynamic_dataset: np.ndarray = np.empty((0, len(self.dynamic_cols) + 1))

            self.obj_gain_metrics = np.empty((6, self.n, int(self.nodes_limit / 1)))
            self.obj_gain_metrics[:] = np.nan
            self.obj_gain_stats = np.zeros((6, self.n, 7))  # metric * variables * stats


    def initialize_model_attr(self, grb_model):

        A_matrix = grb_model.model.getA().toarray()
        self.num_cons = A_matrix.shape[0]

        self.c = [v.obj for v in grb_model.var]

        rhs_senses = [(c.RHS, c.sense) for c in grb_model.model.getConstrs()]
        b, self.senses = zip(*rhs_senses)
        b = np.array(b)

        var_name = [v.varname for v in grb_model.var]
        self.bin_index = list(map(var_name.index, grb_model.bin_var_name))

        Ab = np.hstack((A_matrix, b[:, None]))
        Ab = (Ab.T * [-1 if s == '>' else 1 for s in self.senses]).T
        Ab = np.vstack((Ab, -Ab[[s == '=' for s in self.senses]]))

        self.A_matrix, self.b = np.hsplit(Ab, [-1])
        self.b = self.b.flatten()

        self.c_abs = np.abs(self.c).sum()
        self.c_bin = np.array([v.obj for v in grb_model.bin_var])


    def chg_model(self, grb_model):
        self.n = grb_model.n
        self.__post_init__()
        self.initialize_model_attr(grb_model)


    def collect_dynamic_data(self, node, var_scores_dict=None):

        if self.branch_on_int:
            prospects = [i for i in range(self.n) if node.x[i] == 100]
        else:
            prospects = [i for i in range(self.n) if node.bin_sol[i] % 1 > 0]

        dynamic_data = np.zeros((len(prospects), len(self.dynamic_cols)))

        col = 1
        dynamic_data[:, col] = prospects
        col += 1

        dynamic_data[:, col] = np.count_nonzero(node.x <= 1) / self.n
        col += 1
        dynamic_data[:, col] = 1 - node.bin_sol[prospects]
        col += 1
        dynamic_data[:, col] = node.bin_sol[prospects]
        col += 1
        dynamic_data[:, col] = node.red_cost[prospects]
        col += 1

        dynamic_data[:, col] = np.sign(node.sa_obj_up[prospects])
        col += 1
        dynamic_data[:, col] = np.sign(node.sa_obj_low[prospects])
        col += 1

        dynamic_data[:, col] = np.where(self.c_bin[prospects] == 0, 0,
                                        np.log((node.sa_obj_up[prospects] - self.c_bin[prospects]) / np.abs(
                                            self.c_bin[prospects])))
        col += 1

        dynamic_data[:, col] = np.where(self.c_bin[prospects] == 0, 0,
                                        np.log((self.c_bin[prospects] - node.sa_obj_low[prospects]) / np.abs(
                                            self.c_bin[prospects])))
        col += 1

        for metric in range(6):
            for stat in range(7):
                dynamic_data[:, col] = self.obj_gain_stats[metric, prospects, stat]
                col += 1

        # active constraints features
        active_cons = (node.slack == 0).nonzero()[0]
        b_index = np.asarray(self.bin_index, dtype=int)
        A_active = self.A_matrix[active_cons, :]
        b_active = self.b[active_cons]

        weight_fn = [np.where(b_active == 0, 0, 1 / np.abs(b_active)),
                     np.abs(node.dual[active_cons]) / self.c_abs,
                     1 / np.nansum(np.abs(A_active), axis=1),
                     1 / np.nansum(np.abs(A_active[:, b_index[prospects]]), axis=1)]

        stat = [np.nanmin, np.nanmedian, np.nanmax, np.nanmean, np.nanstd]

        for w in weight_fn:
            for f in stat:
                dynamic_data[:, col] = f(w * np.abs(A_active[:, b_index[prospects]].transpose()), axis=1)
                col += 1

        dynamic_data[:, col] = np.count_nonzero(~np.isnan(A_active[:, b_index[prospects]]),
                                                axis=0) / active_cons.size
        col += 1

        f_selected = self.var_selection_count[prospects]
        f_selected_sum = max(f_selected.sum(), 1)
        
        dynamic_data[:, col] = f_selected / f_selected_sum
        col += 1

        if self.collect:
            dynamic_data = np.hstack((dynamic_data, np.array([[var_scores_dict[i]] for i in prospects])))
            self.dynamic_dataset = np.vstack((self.dynamic_dataset, dynamic_data))
            self.nodes_list.extend([tuple(node.x)] * len(prospects))

        if self.predict:
            return dynamic_data


    def collect_static_data(self):

        A_abs = np.abs(self.A_matrix).sum(axis=1)
        m = self.b.size

        col = 0
        self.static_dataset[:, col] = range(self.n)
        col += 1
        self.static_dataset[:, col] = np.sign(self.c_bin)
        col += 1

        for cost in [self.c, self.c_bin]:
            sum_c_pos = sum(filter(lambda x: x > 0, cost))
            sum_c_neg = sum_c_pos - sum(cost)
            for sum_c in [sum_c_pos, sum_c_neg]:
                if sum_c == 0:
                    self.static_dataset[:, col] = 0
                    col += 1
                else:
                    self.static_dataset[:, col] = [abs(ci) / sum_c for ci in self.c_bin]
                    col += 1

        A_deg = np.count_nonzero(self.A_matrix, axis=1) / self.A_matrix.shape[1]
        non_zero_indices = {k: [each[0] for each in g]
                            for k, g in
                            groupby(
                                sorted(zip(*np.nonzero(self.A_matrix)), key=itemgetter(1)), itemgetter(1)
                            )}
        for f in [min, np.median, max, np.mean, np.std]:
            f_stat = [f(A_deg[non_zero_indices[k]]) for k in self.bin_index]
            self.static_dataset[:, col] = f_stat
            col += 1

        M1_sets = [[self.A_matrix[j, i] / self.b[j] if self.b[j] > 0 else 0
                    for j in range(m) if self.b[j] >= 0]
                   for i in self.bin_index]
        for f in [min, np.median, max, np.mean, np.std]:
            f_stat = [f(m1_i) if m1_i else 0 for m1_i in M1_sets]
            self.static_dataset[:, col] = np.sign(f_stat).flat
            col += 1
            self.static_dataset[:, col] = np.abs(f_stat).flat
            col += 1

        M1_sets = [[self.A_matrix[j, i] / -self.b[j]
                    for j in range(m) if self.b[j] < 0]
                   for i in self.bin_index]
        for f in [min, np.median, max, np.mean, np.std]:
            f_stat = [f(m1_i) if m1_i else 0 for m1_i in M1_sets]
            self.static_dataset[:, col] = np.sign(f_stat).flat
            col += 1
            self.static_dataset[:, col] = np.abs(f_stat).flat
            col += 1

        M2_sets = [[self.c[i] * A_abs[j] / (self.A_matrix[j, i] * self.c_abs) if self.A_matrix[j, i] != 0 else 0
                    for j in range(m)] if self.c[i] >= 0 else []
                   for i in self.bin_index]
        for f in [min, np.median, max, np.mean, np.std]:
            f_stat = [f(m2_i) if m2_i else 0 for m2_i in M2_sets]
            self.static_dataset[:, col] = list(map(lambda x: 0 if x == [] else np.sign(x), f_stat))
            col += 1
            self.static_dataset[:, col] = list(map(lambda x: -1 if x == [] else np.abs(x), f_stat))
            col += 1

        M2_sets = [[-self.c[i] * A_abs[j] / (self.A_matrix[j, i] * self.c_abs) if self.A_matrix[j, i] != 0 else 0
                    for j in range(m)] if self.c[i] < 0 else []
                   for i in self.bin_index]
        for f in [min, np.median, max, np.mean, np.std]:
            f_stat = [f(m2_i) if m2_i else [] for m2_i in M2_sets]
            self.static_dataset[:, col] = list(map(lambda x: 0 if x == [] else np.sign(x), f_stat))
            col += 1
            self.static_dataset[:, col] = list(map(lambda x: -1 if x == [] else np.abs(x), f_stat))
            col += 1

        A_pos_sum = np.clip(self.A_matrix, 0, None).sum(axis=1)
        M3_sets = [[self.A_matrix[j, i] / A_pos_sum[j] if A_pos_sum[j] != 0 else 0
                    for j in range(m) if self.A_matrix[j, i] >= 0]
                   for i in self.bin_index]
        for f in [min, np.median, max, np.mean, np.std]:
            f_stat = [f(m3_i) if m3_i else 0 for m3_i in M3_sets]
            self.static_dataset[:, col] = np.sign(f_stat)
            col += 1
            self.static_dataset[:, col] = np.abs(f_stat)
            col += 1

        M3_sets = [[-self.A_matrix[j, i] / A_pos_sum[j] if A_pos_sum[j] != 0 else 0
                    for j in range(m) if self.A_matrix[j, i] < 0]
                   for i in self.bin_index]
        for f in [min, np.median, max, np.mean, np.std]:
            f_stat = [f(m3_i) if m3_i else 0 for m3_i in M3_sets]
            self.static_dataset[:, col] = np.sign(f_stat)
            col += 1
            self.static_dataset[:, col] = np.abs(f_stat)
            col += 1

        A_neg_sum = -np.clip(self.A_matrix, None, 0).sum(axis=1)
        M3_sets = [[self.A_matrix[j, i] / A_neg_sum[j] if A_neg_sum[j] != 0 else 0
                    for j in range(m) if self.A_matrix[j, i] >= 0]
                   for i in self.bin_index]
        for f in [min, np.median, max, np.mean, np.std]:
            f_stat = [f(m3_i) if m3_i else 0 for m3_i in M3_sets]
            self.static_dataset[:, col] = np.sign(f_stat)
            col += 1
            self.static_dataset[:, col] = np.abs(f_stat)
            col += 1

        M3_sets = [[-self.A_matrix[j, i] / A_neg_sum[j] if A_neg_sum[j] != 0 else 0
                    for j in range(m) if self.A_matrix[j, i] < 0]
                   for i in self.bin_index]
        for f in [min, np.median, max, np.mean, np.std]:
            f_stat = [f(m3_i) if m3_i else 0 for m3_i in M3_sets]
            self.static_dataset[:, col] = np.sign(f_stat)
            col += 1
            self.static_dataset[:, col] = np.abs(f_stat)
            col += 1

        self.A_matrix[self.A_matrix == 0] = np.nan

    
    def predict_branching_var(self, node):

        def agg_min(x):
            if x[(x != -np.inf) & (x != np.inf)].size == 0:
                return 0
            else:
                return np.nanmin(x[x != -np.inf])

        def agg_max(x):
            if x[(x != -np.inf) & (x != np.inf)].size == 0:
                return 1
            else:
                return np.nanmax(x[x != np.inf])
        
        dynamic_data = self.collect_dynamic_data(node)[:, 1:]  # not including node
        prospects = dynamic_data[:, 0].astype(int)
        node_data = np.hstack((dynamic_data[:, 1:], self.static_dataset[prospects, 1:]))  # remove variable columns
        del dynamic_data

        node_data[np.isnan(node_data)] = 0

        # Rescaling all but first feature (frac of fixed vars)
        node_min = np.apply_along_axis(agg_min, 0, node_data[:, 1:])
        node_diff = np.apply_along_axis(agg_max, 0, node_data[:, 1:]) - node_min
        node_diff[node_diff == 0] = 1
        node_data[:, 1:] = (node_data[:, 1:] - node_min) / node_diff

        node_data[node_data == np.Inf] = 2
        node_data[node_data == -np.Inf] = -1
        node_data[np.isnan(node_data)] = 0

        if '0-1' in self.ml_label[1]:
            score = self.ml_model(node_data)[:, 1]
            branch_on = prospects[np.argmax(score)]
        elif self.ml_label[0] == 'opt':
            score = self.ml_model(node_data)
            branch_on = prospects[np.argmin(score)]
        else:
            score = self.ml_model(node_data)
            branch_on = prospects[np.argmax(score)]

        del node_data, node_diff, node_min
        gc.collect()
        
        return branch_on
    
    
    def update_lp_gains(self, branch_on, child_bounds, parent):
        gain_metrics = [lambda v: abs(child_bounds[v] - parent.ub) / self.c_abs,
                        lambda v: abs(child_bounds[v] - parent.ub) / (self.c_abs * abs(v - parent.bin_sol[branch_on])) if
                        parent.bin_sol[branch_on] != v else 0,
                        lambda v: abs(child_bounds[v] - parent.ub) / abs(parent.ub) if parent.ub != 0 else abs(
                            child_bounds[v] - parent.ub)]
        self.obj_gain_metrics[:, branch_on, self.var_selection_count[branch_on] - 1] = \
            [f(val) for f in gain_metrics for val in [0, 1]]

        stat_id = 0
        for f in (np.nanmin, np.nanmax, np.nanmean, np.nanstd):
            self.obj_gain_stats[:, branch_on, stat_id] = f(
                self.obj_gain_metrics[:, branch_on, :self.var_selection_count[branch_on]], axis=1)
            stat_id += 1

        for p in [0.25, 0.5, 0.75]:
            self.obj_gain_stats[:, branch_on, stat_id] = np.nanpercentile(
                self.obj_gain_metrics[:, branch_on, :self.var_selection_count[branch_on]], p, axis=1)
            stat_id += 1