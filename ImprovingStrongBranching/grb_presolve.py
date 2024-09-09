import pandas as pd
from pathlib import Path
import gurobipy as gp

# MIPLIB Benchmark instances
instance_dir = f"../../../ConsistencyOfBranchingRules/MIPLIB/miplibFiles/"
assert Path(instance_dir).exists()

# output directory to store presolved instances
presolved_dir = "gp_presolve_instances/no_cuts/"
Path(presolved_dir).mkdir(parents=True, exist_ok=True)

# filter instances based on ease of solving and number of variables
csv_file = f"{instance_dir}TheBenchmarkSet.csv"
miplib_df = pd.read_csv(csv_file)
miplib_df = miplib_df[
    (miplib_df['Status  Sta.'] == "easy") &
    (miplib_df['Integers  Int.'] + miplib_df['Binaries  Bin.'] <= 500) &
    (miplib_df['Variables  Var.'] <= 10000) &
    (miplib_df['Objective  Obj.'] != 'Infeasible')
    ][['Instance  Ins.', 'Objective  Obj.']]

miplib_df.set_index('Instance  Ins.', inplace=True)
miplib_df['Objective  Obj.'] = pd.to_numeric(miplib_df['Objective  Obj.'])

# iterate through all instances, presolve and save files
for instance, row in miplib_df.iterrows():
    mps_file = instance_dir+instance+'.mps.gz'
    obj = row['Objective  Obj.']

    solve_model = gp.read(mps_file)
    for v in solve_model.getVars():
        v.varname = v.varname.replace('[', '_').replace(']', '_') # scip cant read vars with '[.]'
    solve_model.update()

    outfile = presolved_dir + instance + '.lp'
    presolve_model = solve_model.presolve()
    presolve_model.write(outfile)

    lp_relax = presolve_model.relax()
    lp_relax.optimize()

    if abs(lp_relax.objVal - obj) > 1e-8: # instances not solved at the root node
        with open("gp_presolve_unsolved_at_root.txt", 'a+') as f:
            print(f"{instance}", file=f)