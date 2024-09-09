import pdb
import sys
import numpy as np
from operator import itemgetter


class BranchingRules:

    def __init__(self, branching_rule, sb_score_fn, n, grb_model, opt_tree, tree_estimate_fn):
        self.branching_rule: str = branching_rule
        self.sb_score_fn: str = sb_score_fn
        self.tree_estimate_fn : str = tree_estimate_fn

        self.M: int = 10 ** 8
        self.sb_score_ratio: int = 5
        self.reliability_lim: int = 8
        self.sb_limit = 100
        self.lookahead_lim = 8
        self.rbpc_include_inf = False
        self.limit_lp_iters_bool = False
        self.lp_iters_limit = 500

        self.n = n
        self.grb_model = grb_model
        self.opt_tree = opt_tree

        self.pack_or_cover : str = ""
        self.cardinality_constrained = False
        self.K = n

        self.rules = {
            "optimal_branching": self.optimal_branching,
            "most_inf_branching": self.most_inf_branching,
            "random_branching": self.random_branching,
            "strong_branching": self.strong_branching,
            "strong_branching_lexi": self.strong_branching_lexi,
            "strong_branching_dominated_set": self.strong_branching_dominated_set,
            "tree_estimate_branching": self.tree_estimate_branching,
            "reliability_pc": self.reliability_branching,
            "hybrid_branching": self.hybrid_branching,
            "naive_structural_strong_branching": self.naive_structural_strong_branching,
            "structural_strong_branching": self.structural_strong_branching,
            "cardinality_strong_branching": self.cardinality_strong_branching,
        }

        self.branching_fn = self.rules[branching_rule]

        if branching_rule == "optimal_branching":
            assert opt_tree is not None

        if "pc" in branching_rule:
            self.pseudo_costs = np.empty((2, n, self.reliability_lim))
            self.pseudo_costs[:] = np.nan
            self.reliability_counter = np.zeros((2, n), dtype=np.int)
            self.data_counter = np.zeros((2, n), dtype=np.int)

        self.lp_gain_fns = {
            "product": self.get_prod_score,
            "inv_ratio": self.get_inv_ratio_score,
            "ratio": self.get_inv_ratio_score,
            "linear": self.get_linear_score,
            "min": self.get_min,
            "max": self.get_max,
            "random_combo": self.get_combo_score,
            "log_combo": self.get_log_score,
            "harmonic": self.get_harmonic_score
        }
        self.get_sb_scores = self.lp_gain_fns[self.sb_score_fn]

        self.tree_gap_fns = {
            "polynomial": self.polynomial_estimate,
            "exponential": self.exponential_estimate,
        }
        self.get_tree_estimates = self.tree_gap_fns[self.tree_estimate_fn]
        if branching_rule == "tree_estimate_branching":
            self.coefs = np.array([1, 1, 4, 4])

        if sb_score_fn == "random_combo":
            self.coefs = np.random.random(len(self.lp_gain_fns) - 1)
        elif sb_score_fn == "linear":
            self.coefs = np.array([0, 0, self.sb_score_ratio, 1])
        else:
            self.coefs = np.array([0, 0, 1, 1])

        self.ip_val = -float('inf')
        self.root_lp_gap = float('inf')

        self.perfect_info = False
        self.update_pc = False


    def init_struct_info(self, pack_or_cover):
        self.pack_or_cover = pack_or_cover
        self.original_coefs = self.coefs.copy()

        if self.branching_rule == "naive_structural_strong_branching":
            if self.pack_or_cover == "P":
                self.coefs[:2] = [0.15, 0., ]
            elif self.pack_or_cover == "C":
                self.coefs[:2] = [0., 0.15, ]
            else:
                sys.exit(f" 101: not packing or covering - {self.pack_or_cover}")

            self.branching_rule = "strong_branching"
            self.branching_fn = self.rules[self.branching_rule]
            return

        if self.branching_rule != "structural_strong_branching":
            return

        self.grb_constrs = self.grb_model.model.getConstrs()
        A_matrix = self.grb_model.model.getA().toarray()
        b = np.fromiter((c.RHS for c in self.grb_model.model.getConstrs()), float)

        var_name = [v.varname for v in self.grb_model.var]
        bin_var_name = [v.VarName for v in self.grb_model.bin_var]
        self.bin_index = np.fromiter(map(var_name.index, bin_var_name), int)
        # A_matrix = A_matrix[:, self.bin_index]

        self.pack_or_cover = pack_or_cover
        if self.pack_or_cover == "P":
            self.relevant_cons = [i for i, c in enumerate(self.grb_constrs)
                                  if (c.sense == "<" and np.min(A_matrix[i, :]) >= 0)
                                  or (c.sense == ">" and np.max(A_matrix[i, :]) <= 0)]
            self.packing_constraints = self.relevant_cons
            self.covering_constraints = []
        elif self.pack_or_cover == "C":
            self.relevant_cons = [i for i, c in enumerate(self.grb_constrs)
                                  if (c.sense == ">" and np.min(A_matrix[i, :]) >= 0)
                                  or (c.sense == "<" and np.max(A_matrix[i, :]) <= 0)]
            self.covering_constraints = self.relevant_cons
            self.packing_constraints = []
        else:
            self.packing_constraints = []
            self.covering_constraints = []
            self.mixed_constraints = []

            a_min = np.min(A_matrix, axis=1)
            a_max = np.max(A_matrix, axis=1)


            for i, c in enumerate(self.grb_model.model.getConstrs()):
                sense = c.sense
                if a_min[i] >= 0:
                    if sense == "<": self.packing_constraints.append(i)
                    elif sense == ">": self.covering_constraints.append(i)
                elif a_max[i] <= 0:
                    if sense == ">": self.packing_constraints.append(i)
                    elif sense == "<": self.covering_constraints.append(i)
                else:
                    self.mixed_constraints.append(i)

                self.relevant_cons = self.packing_constraints + self.covering_constraints + self.mixed_constraints

        self.A_matrix = A_matrix[self.relevant_cons, :]
        self.b = b[self.relevant_cons]

    def choose_branching_var(self, node):
        return self.branching_fn(node)

    def optimal_branching(self, node):
        var_score_dict = {}
        for i in [i for i in range(self.n) if node.x[i] == 100]:
            new_tree_size = 1
            for val in [0, 1]:
                child = list(node.x)
                child[i] = val
                size = self.opt_tree.get(tuple(child))
                if size: new_tree_size += size[0]
            var_score_dict.update({i: new_tree_size})
        try:
            return self.opt_tree.get(tuple(node.x))[1], var_score_dict
        except TypeError:
            return None, var_score_dict

    def most_inf_branching(self, node):
        var_score_dict = {i: abs(0.5 - node.bin_sol[i]) for i in range(self.n)}
        return min(var_score_dict, key=var_score_dict.get), var_score_dict

    def reliability_branching(self, node):

        var_prospects = [i for i in range(self.n) if node.bin_sol[i] % 1 > 0]

        if self.perfect_info:
            assert self.ip_val > -float('inf')
            self.threshold = max(node.ub - self.ip_val, 1e-8)
        else:
            self.threshold = float('inf')

        if self.rbpc_include_inf:
            pseudocosts = np.empty_like(self.pseudo_costs)
            pseudocosts[0, :, :] = np.clip(self.pseudo_costs[0, :, :], 1e-8/node.bin_sol[:, None], self.threshold/node.bin_sol[:, None])
            pseudocosts[1, :, :] = np.clip(self.pseudo_costs[1, :, :], 1e-8/(1-node.bin_sol[:, None]), self.threshold/(1-node.bin_sol[:, None]))
        else:
            pseudocosts = self.pseudo_costs

        pc = np.nanmean(pseudocosts, axis=2)
        for branch in [0, 1]:
            mean_pc = np.nanmean(pc[branch, :])
            if np.any(np.isnan(mean_pc)):
                pc[branch, :] = 1
            else:
                pc[branch, np.argwhere(np.isnan(pc[branch, :]).flatten())] = mean_pc

        # rel_count = np.count_nonzero(~np.isnan(pc), axis=2)
        min_rel_count = np.min(self.reliability_counter, axis=0)

        lp_gain_est = np.empty((2, len(var_prospects)))
        lp_gain_est[0, :] = pc[0, var_prospects] * node.bin_sol[var_prospects]
        lp_gain_est[1, :] = pc[1, var_prospects] * (1 - node.bin_sol[var_prospects])

        lp_gain_est[0, :] = np.clip(lp_gain_est[0, :], 1e-8, self.threshold)
        lp_gain_est[1, :] = np.clip(lp_gain_est[1, :], 1e-8, self.threshold)

        if np.max(self.data_counter) >= pseudocosts.shape[2] - 1:
            append_arr = np.empty_like(self.pseudo_costs)
            append_arr[:] = np.nan
            self.pseudo_costs = np.concatenate((self.pseudo_costs, append_arr), axis=2)

        score_estimates = self.get_sb_scores(lp_gain_est.transpose())
        assert len(score_estimates) == len(var_prospects)
        sb_est_pc = sorted(zip(var_prospects, score_estimates, np.random.random(len(var_prospects)) ),
                           key=itemgetter(1, 2), reverse=True)
        sb_est_pc = [l[:2] for l in sb_est_pc]

        sb_counter = 0
        branch_on = 0
        score = -float('inf')
        lookahead_counter = 0

        if self.limit_lp_iters_bool:
            self.grb_model.model.setParam("IterationLimit", self.lp_iters_limit)

        for i in range(len(var_prospects)):

            var = sb_est_pc[i][0]

            if min_rel_count[var] < self.reliability_lim:
                update_pc = False
                sb_counter += 1
                delta = np.empty((1, 2))
                new_face = node.x.copy()

                for j in [0, 1]:

                    new_face[var] = j
                    lp_val = min(self.grb_model.solve_lp(new_face)[0], node.ub)

                    if lp_val == -float('inf'):
                        lp_val = -self.M
                        if self.rbpc_include_inf:
                            self.pseudo_costs[j, var, self.data_counter[j, var]] = np.inf
                            self.data_counter[j, var] += 1
                    else:
                        self.pseudo_costs[j, var, self.data_counter[j, var]] = (node.ub - lp_val) / abs(j - node.bin_sol[var])
                        self.reliability_counter[j, var] += 1
                        self.data_counter[j, var] += 1

                    delta[0, j] = min(node.ub - lp_val, self.threshold)

                sb_est_pc[i] = (var, self.get_sb_scores(delta).item())
            else:
                update_pc = True

            if sb_est_pc[i][1] > score:
                branch_on = var
                lookahead_counter = 0
                score = sb_est_pc[i][1]
                self.update_pc = update_pc
            else:
                lookahead_counter += 1

            if sb_counter == self.sb_limit or lookahead_counter == self.lookahead_lim:
                break

        if self.limit_lp_iters_bool:
            self.grb_model.model.setParam("IterationLimit", float('inf'))

        var_score_dict = dict(sb_est_pc[:i + 1])

        return branch_on, var_score_dict

    def random_branching(self, node):
        var_score_dict = {i: np.random.random() for i in [i for i in range(self.n) if node.bin_sol[i] % 1 > 0]}
        return max(var_score_dict, key=var_score_dict.get), var_score_dict

    def get_children_lp_vals(self, var_prospects, face):

        lp_val = np.empty((len(var_prospects), 2))

        for i in range(len(var_prospects)):
            for j in [0, 1]:
                new_face = face.copy()
                new_face[var_prospects[i]] = j
                lp_val[i][j] = self.grb_model.solve_lp(new_face)[0]

        lp_val[lp_val == -float('inf')] = -self.M
        return lp_val

    def strong_branching(self, node):

        var_prospects = [i for i in range(self.n) if node.bin_sol[i] % 1 > 0]
        lp_val = self.get_children_lp_vals(var_prospects, node.x)
        lp_gains = node.ub - lp_val

        while lp_gains.max() >= 10 * self.M:
            self.M *= 10

        if self.perfect_info:
            assert self.ip_val > -float('inf')

            divisor = node.ub - self.ip_val
            if divisor == 0:
                divisor += 1

            lp_gains /= divisor
            lp_gains = np.clip(lp_gains, 1e-8, 1)

            var_score = self.get_sb_scores(lp_gains)
        else:
            np.clip(lp_gains, 1e-8, None, lp_gains)
            var_score = self.get_sb_scores(lp_gains)

        var_score_dict = {var_prospects[i]: var_score[i] for i in range(len(var_prospects))}
        return var_prospects[np.nanargmax(var_score)], var_score_dict

    def cardinality_strong_branching(self, node):

        k = self.K - np.count_nonzero(node.x == 1)
        n = np.count_nonzero(node.x == 2)
        if n - k <= k:
            self.coefs[:2] = [0, 0]
        else:
            self.coefs[:2] = [(n-k)/(n), k/(n)]

        return self.strong_branching(node)

    def naive_structural_strong_branching(self, node):

        if self.pack_or_cover == "P":
            self.coefs[:2] = [0.15, 0., ]
        elif self.pack_or_cover == "C":
            self.coefs[:2] = [0., 0.15,]
        else:
            sys.exit(f" 344: not packing or covering - {self.pack_or_cover}")

        return self.strong_branching(node)

    def structural_strong_branching(self, node):


        var_prospects = [i for i in range(self.n) if node.bin_sol[i] % 1 > 0]
        fixed_1 = np.argwhere(node.x == 1).flatten()
        unfixed = np.argwhere(node.x == 2).flatten()

        remaining_cap = (self.b - np.sum(self.A_matrix[:, self.bin_index[fixed_1]], axis=1))[:, None]
        A_unfixed = self.A_matrix[:, self.bin_index[unfixed]]
        A_prospects = self.A_matrix[:, self.bin_index[var_prospects]]

        
        # row sums children = sum_{j \neq j*} a_ij
        row_sums_children = np.sum(A_unfixed, axis=1)[:, None] - A_prospects
        # row_sums_children[row_sums_children == 0] = float('inf')

        if self.pack_or_cover == "P":
            rp = (remaining_cap - A_prospects)/row_sums_children
            r = np.mean(rp[(rp > 0) & (rp < 1)])
            a_0 = (0.3/1.15)*(1-r) # 0.15 * np.exp(0.5 - r)
            # a_0 = max(0.15 * (1 - r), 0)
            # a_1 = - a_0
            self.coefs[:] = [a_0, 0, 0.3*(1-a_0), 0.7*(1-a_0)]

        elif self.pack_or_cover == "C":
            rp = (row_sums_children - remaining_cap)/row_sums_children
            r = np.mean(rp[(rp > 0) & (rp < 1)])
            a_1 = (0.3/1.15)*(1-r) #0.15 * np.exp(0.5 - r)
            # a_1 = max(0.15 * (1 - r), 0)
            # a_0 = - a_1
            self.coefs[:] = [0, a_1, 0.3*(1-a_1), 0.7*(1-a_1)]
        else:
            r = np.nan

        if np.isnan(r):
            self.coefs[:] = self.original_coefs.copy()

        """
        # zero_rhs = np.tile(remaining_cap[:, None], (1, len(var_prospects)))
        # one_rhs = zero_rhs - A_prospects
        # zero_rhs = np.mean(zero_rhs / row_sums_children, axis=0)
        # one_rhs = np.mean(one_rhs / row_sums_children, axis=0)
        # sum_rhs = (zero_rhs + one_rhs)

        # self.coefs = [(zero_rhs) / sum_rhs, (one_rhs) / sum_rhs, self.original_coefs[2], self.original_coefs[3]]
        # add_factor = np.clip(zero_rhs - one_rhs, 0, 0.3)/2
        
        if self.pack_or_cover == "P":
            self.coefs[:2] = [add_factor, -add_factor, ]
        elif self.pack_or_cover == "C":
            self.coefs[:2] = [-add_factor, add_factor, ]
        """

        return self.strong_branching(node)


    def strong_branching_lexi(self, node):

        var_prospects = [i for i in range(self.n) if node.bin_sol[i] % 1 > 0]

        lp_val = np.empty((len(var_prospects), 2))
        face = node.x
        for i in range(len(var_prospects)):
            for j in [0, 1]:
                new_face = face.copy()
                new_face[var_prospects[i]] = j
                lp_val[i][j] = self.grb_model.solve_lp(new_face)[0]
                if lp_val[i][j] == -float('inf') or (self.perfect_info and lp_val[i][j] <= self.ip_val):
                    return var_prospects[i], None

        lp_gains = node.ub - lp_val

        np.clip(lp_gains, 1e-8, None, lp_gains)
        var_score = self.get_sb_scores(lp_gains)

        var_score_dict = {var_prospects[i]: var_score[i] for i in range(len(var_prospects))}
        return var_prospects[np.nanargmax(var_score)], var_score_dict


    def strong_branching_dominated_set(self, node):

        var_prospects = [i for i in range(self.n) if node.bin_sol[i] % 1 > 0]

        lp_val = np.empty((len(var_prospects), 2))
        face = node.x
        pruning_set = []
        for i in range(len(var_prospects)):
            for j in [0, 1]:
                new_face = face.copy()
                new_face[var_prospects[i]] = j
                lp_val[i][j] = self.grb_model.solve_lp(new_face)[0]
                if lp_val[i][j] == -float('inf') or (self.perfect_info and lp_val[i][j] <= self.ip_val + 1e-6):
                    pruning_set.append(i)

        lp_gains = node.ub - lp_val
        if len(pruning_set) == 0:

            np.clip(lp_gains, 1e-8, None, lp_gains)
            var_score = self.get_sb_scores(lp_gains)

            var_score_dict = {var_prospects[i]: var_score[i] for i in range(len(var_prospects))}
            return var_prospects[np.nanargmax(var_score)], var_score_dict
        else: # pick var where min lp gain is the max - we know the max side is going to be pruned
            dom_var_score = np.min(lp_gains[pruning_set, :], axis=1)
            if self.perfect_info: pdb.set_trace()
            return var_prospects[pruning_set[np.argmax(dom_var_score)]], None


    def hybrid_branching(self, node):
        var_prospects = [i for i in range(self.n) if node.bin_sol[i] % 1 > 0]

        lp_val = np.empty((len(var_prospects), 2))
        distances = np.zeros((len(var_prospects), 2))
        face = node.x

        for i in range(len(var_prospects)):
            for j in [0, 1]:
                new_face = face.copy()
                new_face[var_prospects[i]] = j
                lp_val[i][j], sol = self.grb_model.solve_lp(new_face)
                if sol is not None:
                    distances[i][j] = np.sum(np.abs(sol - node.bin_sol))

        lp_val[lp_val == -float('inf')] = -self.M

        lp_gains = node.ub - lp_val

        while lp_gains.max() >= 10 * self.M:
            self.M *= 10

        if self.perfect_info:
            assert self.ip_val > -float('inf')

            divisor = node.ub - self.ip_val
            if divisor == 0:
                divisor += 1
            np.clip(lp_gains / divisor, 1e-8, 1, lp_gains)

        else:
            np.clip(lp_gains, 1e-8, None, lp_gains)

        # alternately coefs = 1/(1+distances), 0, 0

        obj_scores = self.get_sb_scores(lp_gains)
        dist_scores = distances.sum(axis=1)
        
        if obj_scores.max() == obj_scores.min():
            obj_scores[:] = 1
        else:
            obj_scores = (obj_scores - obj_scores.min())/(obj_scores.max() - obj_scores.min())

        if dist_scores.max() == dist_scores.min():
            dist_scores[:] = 1
        else:
            dist_scores = (dist_scores - dist_scores.min()) / (dist_scores.max() - dist_scores.min())

        var_score = obj_scores + 0.1 * dist_scores

        var_score_dict = {var_prospects[i]: var_score[i] for i in range(len(var_prospects))}
        return var_prospects[np.nanargmax(var_score)], var_score_dict

    def tree_estimate_branching(self, node):

        var_prospects = [i for i in range(self.n) if node.bin_sol[i] % 1 > 0]
        lp_val = self.get_children_lp_vals(var_prospects, node.x)

        assert self.ip_val > -float('inf')

        int_gap_at_children = (lp_val - self.ip_val)/self.root_lp_gap

        var_score = self.get_tree_estimates(int_gap_at_children)
        var_score_dict = {var_prospects[i]: var_score[i] for i in range(len(var_prospects))}

        return var_prospects[np.nanargmin(var_score)], var_score_dict

    def exponential_estimate(self, gap_arr):
        cols = gap_arr.shape[1]
        return (np.exp(gap_arr * self.coefs[cols:]) @ self.coefs[:cols])

    def polynomial_estimate(self, gap_arr):
        cols = gap_arr.shape[1]
        return (gap_arr ** self.coefs[cols:]) @ self.coefs[:cols]

    def get_inv_ratio_score(self, arr):
        l = np.min(arr, axis=1)
        r = np.max(arr, axis=1)

        l_by_r = np.clip(l / r, 1e-8, None)

        x, x_prev = np.full_like(l, 2), np.full_like(l, 0)

        for _ in range(1000):
            x_prev[:] = x[:]
            x = 1 + 1 / (x ** l_by_r - 1)
            if np.all(np.abs(x - x_prev) / np.abs(x) < 1e-8):
                break

        return r / np.log2(x)  # (1/x)**(1/r) #

    def get_prod_score(self, arr):
        return np.sqrt(np.prod(arr, axis=1))

    def get_linear_score(self, arr):
        # return (np.max(arr, axis=1) + self.sb_score_ratio * np.min(arr, axis=1)) / (1 + self.sb_score_ratio)
        score = np.zeros(arr.shape[0], dtype=float)

        score += self.coefs[0] * arr[:, 0]
        score += self.coefs[1] * arr[:, 1]
        score += self.coefs[2] * np.min(arr, axis=1)
        score += self.coefs[3] * np.max(arr, axis=1)

        return score

    def get_harmonic_score(self, arr):
        # return 1/np.sum(1/arr, axis=1)
        score = np.zeros(arr.shape[0], dtype=float)

        score += (self.coefs[0] / arr[:, 0])
        score += (self.coefs[1] / arr[:, 1])
        score += (self.coefs[2] / np.min(arr, axis=1))
        score += (self.coefs[3] / np.max(arr, axis=1))

        return 1/score

    def get_min(self, arr):
        return np.min(arr, axis=1)

    def get_max(self, arr):
        return np.max(arr, axis=1)

    def get_combo_score(self, arr):
        score = np.zeros(arr.shape[0], dtype=float)

        # coefs order = ratio, min, max, prod, min2max1, min1max2
        min_arr = self.get_min(arr)
        max_arr = self.get_max(arr)

        if self.coefs[0] > 1e-6: score += self.coefs[0] * self.get_inv_ratio_score(arr)
        score += self.coefs[1] * min_arr
        score += self.coefs[2] * max_arr
        score += self.coefs[3] * (min_arr * max_arr) ** (1 / 2)
        score += self.coefs[4] * (min_arr * min_arr * max_arr) ** (1 / 3)
        score += self.coefs[5] * (min_arr * max_arr * max_arr) ** (1 / 3)

        return score

    def get_log_score(self, arr):
        arr = np.log2(arr)

        score = np.zeros(arr.shape[0], dtype=float)

        score += self.coefs[0] * arr[:, 0]
        score += self.coefs[1] * arr[:, 1]
        score += self.coefs[2] * np.min(arr, axis=1)
        score += self.coefs[3] * np.max(arr, axis=1)

        return score
