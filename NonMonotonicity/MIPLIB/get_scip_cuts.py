# Reads miplib instances, add cuts and saves them as lp files
# Files are stored and read again while solving to synchronize the same seed at the start of solve
# Requires a folder called miplibFiles containing -
# 1) instances in lp format that are readable by scip (e.g. variable names not containing '[.]')
# 2) BenchmarkSet.csv from MIPLIB website

import sys
sys.path.append("/home/pshah398/LearningToBranch/EcoleGasseEtAl/RandomTrees/BranchAndBound")
import pyscipopt as scip


disable_permuatations = {
    "randomization/permutevars" : False,
    "randomization/permuteconss" : False,
}

disable_presolve_params = {
    "presolving/maxrounds": 0,
    "presolving/maxrestarts": 0,
}

disable_root_cuts_params = {
    "separating/maxroundsroot": 0,
    "separating/maxcutsroot": 0,
}

scip_parameters = {
    "propagating/maxrounds": 0,
    "propagating/maxroundsroot": 0,

    "separating/maxrounds": 0,
    "separating/maxcuts": 0,

    "conflict/enable": False,

    "display/verblevel": 5,
    "display/freq": 1,

    "branching/vanillafullstrong/priority": 536870911,
    "branching/vanillafullstrong/idempotent" : True,
    "branching/vanillafullstrong/scoreall": True,

    "limits/time": 3*24*3600,
    "limits/nodes": 1,
}

randomization_params = {
    "randomization/randomseedshift": 0,
    "randomization/permutationseed": 0,
    "randomization/lpseed": 0,
}

# MIPLIB Benchmark instances
instance_dir = f"miplibFiles/"
assert Path(instance_dir).exists()

# output directory to store presolved instances
presolved_dir = "scip_cuts"
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
    input_file = f"{instance_dir}{instance}.lp"

    # Repeat for 3 seeds and upto 10 rounds of cuts
    for seed in range(1, 4):
        for rounds in range(1, 11):

            # Create scip model
            model = scip.Model()
            for k in randomization_params.keys():
                randomization_params[k] = seed

            # Set params to switch off presolve, propogation, heuristics variable permutations, limiting bnb to solve only root node
            model.setParams(scip_parameters)
            model.setParams(disable_presolve_params)
            model.setParams(disable_permuatations)
            model.setPresolve(scip.SCIP_PARAMSETTING.OFF)
            model.disablePropagation()

            # Set number of root nodes rounds
            model.setParams({"separating/maxroundsroot" : cut_rounds})

            # Set global random seed
            model.setParams(randomization_params)

            # Read model and solve root node
            model.readProblem(input_file)
            model.optimize()

            new_cuts = 0
            cut_lines = [] # list of cuts formatted as strings in .lp format

            for row in model.getLPRowsData():

                if row.getOrigintype() not in [1, 3]: # identify cuts added by constraint handlers (1) or separators (3)
                    continue

                # convert row into constraint in .lp format
                new_cuts += 1
                cut_line = f" newcut_{new_cuts}:"
                for col, val in zip(row.getCols(), row.getVals()):
                    if val >= 0:
                        cut_line += f" +{val} {str(col.getVar())[2:]}"
                    else:
                        cut_line += f" {val} {str(col.getVar())[2:]}"

                if row.getLhs() > -1e20:
                    cut_line += f" >= {row.getLhs()}\n"
                else:
                    cut_line += f" <= {row.getRhs()}\n"

                cut_lines.append(cut_line)

            # Read LP file of original mip
            with open(input_file, 'r') as f:
                lp_lines = f.readlines()

            # Identify line where cuts are to be added (at the end of constraints)
            cut_i = 0
            while lp_lines[cut_i] != "Bounds\n":
                cut_i += 1

            # Write new LP file
            output_file = f"gp_presolve_instances/scip_cut_rounds/{instance}_seed{seed}_round{cut_rounds}.lp"
            with open(output_file, "w") as f:
                f.writelines(lp_lines[:cut_i])
                f.writelines(cut_lines)
                f.writelines(lp_lines[cut_i:])

