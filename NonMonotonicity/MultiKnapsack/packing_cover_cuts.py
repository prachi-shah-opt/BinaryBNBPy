import pickle
import numpy as np
import gurobipy as gp
from pathlib import Path
from matplotlib import pyplot as plt, ticker as mtick
from BranchAndBound import branch_and_bound as bnb
import BranchAndBound.instance_generator_grb as instance_generator

problem = "PackingIP"
nvars, ncons, density, rhs_ratio = (20, 50, 0.75, 0.5)
NUM_INSTANCES = 100

def Generator(gen_params):
    while True:
        model = instance_generator.generate(*gen_params)
        yield model


params = (nvars, ncons, density, rhs_ratio)
generator = Generator((problem, params))

instance_dir = f"instances/{problem}_{'_'.join(map(str, params))}/"
instances_w_cuts_dir = f"instances_with_cuts/{problem}_{','.join(map(str, params))}/"
Path(instance_dir).mkdir(parents=True, exist_ok=True)
Path(instances_w_cuts_dir).mkdir(parents=True, exist_ok=True)

# stats for instances with individual cuts
cut_nodes = np.zeros((NUM_INSTANCES, ncons), dtype=np.uint16)
cut_distances = np.empty((NUM_INSTANCES, ncons), dtype=float)

cut_root_lps = np.empty((NUM_INSTANCES, ncons), dtype=float)
cut_root_lps[:] = np.nan

# stats for original model
model_nodes = np.empty(NUM_INSTANCES, dtype=np.uint16)
model_root_lps = np.empty(NUM_INSTANCES)
model_ip_vals = np.empty(NUM_INSTANCES)

# stats for model after adding all cuts
allcuts_nodes = np.empty(NUM_INSTANCES, dtype=np.uint16)
allcuts_rootlp = np.empty(NUM_INSTANCES)

for i_inst in range(NUM_INSTANCES):

    env = gp.Env()
    env.setParam("OutputFlag", 0)

    mps_file = f"{instance_dir}instance_{i_inst+1}.mps"

    if not Path(mps_file).exists():
        instance = next(generator)
        instance.write(mps_file)

    # Create and solve linear relaxation
    lp_model = gp.read(mps_file, env)
    lp_vars = lp_model.getVars()

    for v in lp_vars:
        v.vtype = gp.GRB.CONTINUOUS

    lp_model.optimize()
    x_lp = np.array([v.x for v in lp_vars])
    model_root_lps[i_inst] = lp_model.objVal

    # solve ip to get number of cuts
    model_nodes[i_inst] = bnb.BranchAndBound(mps_file, "strong_branching", "product").solve_bnb()[0]

    # IP model to generate mps files with cuts
    ip_model = gp.read(mps_file, env)
    ip_vars = ip_model.getVars()
    ip_model.optimize()
    model_ip_vals[i_inst] = ip_model.objVal

    # Cut generating LP and matrices required for getting cuts
    cut_generator = gp.Model(env=env)
    cut_gen_vars = cut_generator.addVars(len(lp_vars), obj = [1 - x for x in x_lp], vtype='B')
    A_matrix = lp_model.getA()
    rhs = np.array(lp_model.getAttr("RHS", lp_model.getConstrs()), dtype=np.uint16)

    # Tracking all covers found
    all_covers = []

    # Find a cover for all packing constraints
    for i_con in range(A_matrix.shape[0]):

        # add constr to cut gen model, a x >= b + 1
        con = cut_generator.addLConstr(
            gp.LinExpr(A_matrix[i_con, :].data, [cut_gen_vars[i] for i in A_matrix[i_con, :].indices]) >= rhs[i_con] + 1
        )
        cut_generator.optimize()

        # if 1 - obj <= 0 : no good cover cut for this row
        if cut_generator.objVal - 1 >= -1e-8:
            cut_generator.remove(con)
            continue

        # if 1 - obj > 0 : get sol wherever var is 1 -> add contr sum_x <= C - 1
        cover = [i for i in range(nvars) if cut_gen_vars[i].x == 1]
        all_covers.append(cover)

        # add constr to lp model and get bound
        lp_con = lp_model.addConstr(gp.quicksum(lp_vars[i] for i in cover) <= len(cover) - 1)
        lp_model.optimize()

        # add constr to IP model and save in saved_mps_dir
        ip_con = ip_model.addConstr(gp.quicksum(ip_vars[i] for i in cover) <= len(cover) - 1)
        new_mps_file = f"{instances_w_cuts_dir}instance_{i_inst + 1}_{i_con}.mps"
        ip_model.update()
        ip_model.write(new_mps_file)

        # Update results for this cut
        cut_root_lps[i_inst, i_con] = lp_model.objVal
        cut_nodes[i_inst, i_con] = bnb.BranchAndBound(new_mps_file, "strong_branching", "product").solve_bnb()[0]
        cut_distances[i_inst, i_con] = (1 - cut_generator.objVal)/np.sqrt(len(cover))

        # Reset the models for the next constraint
        cut_generator.remove(con)
        lp_model.remove(lp_con)
        ip_model.remove(ip_con)

    # Solve while adding all cover cuts obtained for the model
    for cover in all_covers:
        lp_model.addConstr(gp.quicksum(lp_vars[i] for i in cover) <= len(cover) - 1)
        ip_model.addConstr(gp.quicksum(ip_vars[i] for i in cover) <= len(cover) - 1)

    # add constr to lp model and get bound
    lp_model.optimize()

    # add constr to IP model and save in saved_mps_dir
    new_mps_file = f"{instances_w_cuts_dir}instance_{i_inst + 1}_allcovers.mps"
    ip_model.update()
    ip_model.write(new_mps_file)

    # Update results arrays
    allcuts_rootlp[i_inst] = lp_model.objVal
    allcuts_nodes[i_inst] = bnb.BranchAndBound(new_mps_file, "strong_branching", "product").solve_bnb()[0]

pkl_file = f"results.pkl"
with open(pkl_file, "wb") as f:
    pickle.dump([model_root_lps, model_nodes, model_ip_vals, cut_distances, cut_root_lps, cut_nodes, allcuts_rootlp, allcuts_nodes, ], f)


""" Plot Results """
pkl_file = f"results.pkl"
with open(pkl_file, "rb") as f:
    (model_root_lps, model_nodes, model_ip_vals, cut_distances, cut_root_lps, cut_nodes, allcuts_rootlp, allcuts_nodes) = pickle.load(f)

num_cuts_applied = np.sum(cut_nodes != 0, axis=1)
num_cuts_bigger_tree = np.sum((cut_nodes != 0) & (cut_nodes > model_nodes[:, None]), axis=1)
all_cuts_bigger_tree = allcuts_nodes > model_nodes
nnz = np.argwhere(cut_nodes.flatten() > 0).flatten()

print(f"Across {NUM_INSTANCES} instances, {num_cuts_applied.sum()} individual cover cuts were applied out of which {num_cuts_bigger_tree.sum()} lead to larger trees")
print(f"When all cover cuts were applied for each instance, {np.sum(all_cuts_bigger_tree)/np.sum(num_cuts_applied > 0) * 100}% instances saw larger tree sizes.")

# Plot 1 - change in tree size vs gap closed for individual cuts
gap_closed = ((cut_root_lps - model_root_lps[:, None])/(model_ip_vals[:, None] - model_root_lps[:, None])).flatten()[nnz]
change_in_nnodes = (cut_nodes/model_nodes[:, None] - 1).flatten()[nnz]
plt.scatter(gap_closed, change_in_nnodes, c="orange", alpha=0.7, s=20)
plt.plot(np.linspace(0, gap_closed.max()*1.1, 11), np.full(11, 0), label='y = 0', alpha=0.5)
plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.title("Individual Cut", size=14)
plt.xlabel("Gap Closed by Cut", size=14)
plt.ylabel("Change in Tree Size", size=14)
plt.legend()
plt.subplots_adjust(bottom=0.12, top=0.92, left=0.2, right=0.95)
plt.gcf().set_size_inches(4, 4)
plt.gcf().set_dpi(100)
plt.savefig("individual_gap.png")
plt.close()


# Plot 2 - change in tree size vs depth for individual cuts
depth = cut_distances.flatten()[nnz]
change_in_nnodes = (cut_nodes/model_nodes[:, None] - 1).flatten()[nnz]
plt.scatter(depth, change_in_nnodes, c="orange", alpha=0.7, s=20)
plt.plot(np.linspace(0, depth.max()*1.1, 11), np.full(11, 0), label='y = 0', alpha=0.5)
# plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.title("Individual Cut", size=14)
plt.xlabel("Depth of Cut", size=14)
plt.ylabel("Change in Tree Size", size=14)
plt.legend()
plt.subplots_adjust(bottom=0.12, top=0.92, left=0.2, right=0.95)
plt.gcf().set_size_inches(4, 4)
plt.gcf().set_dpi(100)
plt.savefig("individual_depth.png")
plt.close()


# Plot 3 - change in tree size vs gap closed for all cuts added to an instance
ids = np.argwhere(num_cuts_applied > 0).flatten()
gap_closed = (allcuts_rootlp - model_root_lps)[ids]/(model_ip_vals - model_root_lps)[ids]
change_in_nnodes = (allcuts_nodes/model_nodes - 1).flatten()[ids]
plt.scatter(gap_closed, change_in_nnodes, c="orange", alpha=0.7, s=20)
plt.plot(np.linspace(0, gap_closed.max()*1.1, 11), np.full(11, 0), label='y = 0', alpha=0.5)
plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
plt.title("All Cuts", size=14)
plt.xlabel("Gap Closed by Cut", size=14)
plt.ylabel("Change in Tree Size", size=14)
plt.legend()
plt.subplots_adjust(bottom=0.12, top=0.92, left=0.2, right=0.95)
plt.gcf().set_size_inches(4, 4)
plt.gcf().set_dpi(100)
plt.savefig("all_cuts_gap.png")
plt.close()



