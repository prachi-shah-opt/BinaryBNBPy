import gc
from itertools import product, chain
import multiprocessing as mp
from gurobipy import GRB
import numpy as np
import gurobipy as gp
import sys


def opt_tree(mps_file : str, binvar_name=None):

    global split_dim, model, var, bin_var, ip_opt_val, n, bin_var_name, mps_path

    mps_path = mps_file

    env = gp.Env(empty=True)
    env.setParam("OutputFlag",0)
    env.start()
    model = gp.read(mps_path, env=env)

    change_sense = 1
    if model.ModelSense == 1:
        model.ModelSense = -1
        model.setObjective(gp.quicksum(-v * v.Obj for v in model.getVars()))
        model.setParam('LogToConsole', 0)
        model.update()
        change_sense = -1

    if binvar_name is not None:
        bin_var = [model.getVarByName(v) for v in binvar_name]
        bin_var_name = binvar_name
    else:
        bin_var = [v for v in model.getVars() if v.vtype == 'B' or v.vtype == 'I']
        bin_var_name = model.getAttr(GRB.Attr.VarName, bin_var)


    n = len(bin_var)
    split_dim = min(int(np.floor(n / 2)), 8)

    model.optimize()
    if model.status == 3:
        print('infeasible IP')
        ip_opt_val = -float('inf')
    else:
        ip_opt_val = round(model.objVal, 8)
        # print(f"ip_opt_val = {ip_opt_val}")

    for x in bin_var:
        x.vtype = GRB.CONTINUOUS
    model.update()
    model.write(mps_path)

    # Execute phase 1
    non_zero_faces = phase_1()

    # Execute phase 2
    all_faces = phase2(set(non_zero_faces))

    if all_faces == {}:
        all_faces.update({tuple([2] * n): (0, 0, None)})
        opt_tree_size = 0
    else:
        opt_tree_size = all_faces[tuple([2] * n)][0]

    rem_nodes = [tuple([2] * n)]
    opt_tree = {}

    while rem_nodes:
        node = rem_nodes.pop()
        attrs = all_faces.get(node)

        if attrs:
            opt_tree.update({node: attrs})

            if attrs[0] > 1:
                var = attrs[1]
                node_l = list(node)

                for val in [0, 1]:
                    node_l[var] = val
                    rem_nodes.append(tuple(node_l))

    del model, non_zero_faces
    gc.collect()

    if len(opt_tree) != opt_tree_size: sys.exit(f"opt tree = {len(opt_tree)}, computed size = {opt_tree_size}")

    return all_faces, opt_tree_size, bin_var_name, ip_opt_val*change_sense, opt_tree


def phase_1():

    faces_0x = list(product([1, 0, 2], repeat=split_dim))

    # pool = mp.Pool(min(mp.cpu_count(), 16))
    # results_obj = [pool.apply_async(eval_faces, args=(each, mps_path, ip_opt_val, bin_var_name, n, split_dim))
    #                for each in faces_0x]
    # pool.close()
    # pool.join()
    with mp.Pool(min(mp.cpu_count(), 16)) as pool:
        results_obj = [pool.apply_async(eval_faces,
                                        args=(each, mps_path, ip_opt_val, bin_var_name, n, split_dim))
                       for each in faces_0x]
        pool.close()
        pool.join()

    del faces_0x
    gc.collect()

    # results = [r.get() for r in results_obj]
    # return list(chain.from_iterable([x for x in results]))

    return list(chain.from_iterable([r.get() for r in results_obj]))


def eval_faces(face_0x, mps_path, ip_opt_val, bin_var_name, n, split_dim):

    faces_xn = list(sorted(product([1, 0, 2], repeat=n - split_dim), key=lambda x: x.count(2)))

    env = gp.Env(empty=True)
    env.setParam("OutputFlag",0)
    env.start()
    m = gp.read(mps_path, env=env)

    b_var = [m.getVarByName(name) for name in bin_var_name]

    non_zeros = []

    while faces_xn:

        f_xn = faces_xn.pop()
        face = face_0x + f_xn

        face_cons = m.addConstrs((b_var[i] == face[i] for i in range(n) if face[i] <= 1), name='face')
        m.optimize()

        if m.status == 3 or round(m.objVal, 8) <= ip_opt_val:
            fix_var = [i for i in range(len(f_xn)) if f_xn[i] != 2]
            fix_val = [f_xn[i] for i in fix_var]
            faces_xn = list(filter(lambda x: [x[i] for i in fix_var] != fix_val, faces_xn))
        else:
            non_zeros.append(face)

        # if face.count(2) < n:
        #     m.remove(m.getConstrs()[-(n - face.count(2)):])
        m.remove(face_cons)

    return non_zeros


def phase2(non_zero_faces):

    def var_scores_1d(idx):
        var_scores = np.empty(n)
        var_scores[:] = np.nan
        var_scores[idx] = 1
        return var_scores

    curr_dim_faces = set(filter(lambda y: y.count(2) == 1, non_zero_faces))

    opt = {F: (1, F.index(2), var_scores_1d(F.index(2))) for F in curr_dim_faces}
    Face_list = non_zero_faces - curr_dim_faces
    del non_zero_faces
    gc.collect()

    for dim in range(2, n+1):

        curr_dim_faces = set(filter(lambda y: y.count(2) == dim, Face_list))
        Face_list = Face_list - curr_dim_faces

        curr_face_size = []
        for F in curr_dim_faces:

            # iterate over free variables and create facets
            free_var = [i for i in range(n) if F[i] == 2]
            branch_on_free = [1] * len(free_var)

            for k in range(len(free_var)):
                i = free_var[k]
                for x in [0, 1]:
                    g = list(F)
                    g[i] = x
                    if tuple(g) in opt.keys():
                        branch_on_free[k] += opt[tuple(g)][0]
            min_size, min_k = min([val, idx] for idx, val in enumerate(branch_on_free))

            var_scores = np.empty(n); var_scores[:] = np.nan
            var_scores[free_var] = branch_on_free
            # store as a list of tuples - curr_face_size
            curr_face_size.append((F, [min_size, free_var[min_k], var_scores]))

        # update in opt
        opt.update(curr_face_size)

    del Face_list, curr_dim_faces
    gc.collect()
    return opt
