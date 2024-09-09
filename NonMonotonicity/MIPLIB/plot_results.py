from matplotlib import pyplot as plt, ticker as mtick, colors as cm
import numpy as np
from pathlib import Path
import pandas as pd

num_rounds = 10

normalize_tree_size = True
remove_incomplete_runs = True
results_dir = f"Results/"
Path(results_dir).mkdir(parents=True, exist_ok=True)


def format_plot(filename, instance=None):
    xlims = plt.gca().get_xlim()
    if normalize_tree_size:
        plt.plot(np.linspace(xlims[0], max(xlims[1], 0.05), 11), np.full(11, 0), color='gray', linestyle='dashed',
                 label='y = 0', linewidth=2)
        plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1.00))
        plt.ylabel("Change in Tree Size", size=24, labelpad=10)
    else:
        plt.plot(np.linspace(xlims[0], max(xlims[1], 0.05), 21), np.full(11, 0), alpha=0, color='gray',
                 linestyle='dashed', linewidth=2)
        plt.ylabel("Tree Size", size=24, labelpad=10)

    if instance is None:
        plt.title(f"Rounds of cuts across instances and seeds", size=24)
    else:
        plt.title(f"{instance}", size=24)

    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=0))
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.gca().tick_params(axis='both', which='major', pad=15)

    plt.xlabel("Gap Closed by Cut", size=24, labelpad=20)
    plt.legend(fontsize=16)

    plt.gcf().set_size_inches(7, 5.6)
    plt.gcf().set_dpi(100)
    plt.tight_layout()

    plt.savefig(filename)
    plt.show()
    plt.close()


instance_dir = f"miplibFiles/"
assert Path(instance_dir).exists()

csv_file = f"{instance_dir}TheBenchmarkSet.csv"
miplib_df = pd.read_csv(csv_file)
miplib_df = miplib_df[
    (miplib_df['Status  Sta.'] == "easy") &
    (miplib_df['Integers  Int.'] + miplib_df['Binaries  Bin.'] <= 500) &
    (miplib_df['Variables  Var.'] <= 10000) &
    (miplib_df['Objective  Obj.'] != 'Infeasible')][['Instance  Ins.', 'Objective  Obj.']]

miplib_df['filePath'] = instance_dir + miplib_df['Instance  Ins.'] + '.mps.gz'
miplib_df.set_index('Instance  Ins.', inplace=True)


lpbounds_dict = {}
tree_size_dict = {}
with open("Results/results.txt", "r") as f:
    results = f.readlines()
    for line in results:
        linedata = line.split(", ")
        instance, seed, round, lpbound, treesize = linedata[0], int(linedata[1]), int(linedata[2]), float(linedata[3]), int(linedata[4])
        if round > num_rounds: continue
        if instance in lpbounds_dict:
            lpbounds_dict[instance][seed][round] = lpbound
            tree_size_dict[instance][seed][round] = treesize
        else:
            lpbounds_dict[instance] = {s: np.full(1+num_rounds, np.nan) for s in range(1, 4)}
            tree_size_dict[instance] = {s: np.full(1+num_rounds, np.nan) for s in range(1, 4)}

            lpbounds_dict[instance][seed][round] = lpbound
            tree_size_dict[instance][seed][round] = treesize


for instance in lpbounds_dict.keys():
    ipval = float(miplib_df.loc[instance, 'Objective  Obj.'])

    for seed in range(1, 4):
        bounds_arr = lpbounds_dict[instance][seed]
        tree_arr = tree_size_dict[instance][seed]

        bounds_arr = (bounds_arr - bounds_arr[0])/(ipval - bounds_arr[0])
        if normalize_tree_size:
            tree_arr = (tree_arr - tree_arr[0])/tree_arr[0]

        lpbounds_dict[instance][seed] = bounds_arr
        tree_size_dict[instance][seed] = tree_arr

    if len(lpbounds_dict[instance]) == 0:
        print(f"instance {instance} is empty")
        del lpbounds_dict[instance]
        del tree_size_dict[instance]


# Plot each instance
for instance in lpbounds_dict.keys():
    for seed in lpbounds_dict[instance].keys():
        plt.plot(lpbounds_dict[instance][seed], tree_size_dict[instance][seed], linewidth=2, label=f"seed: {seed}", marker='o')

    file_name = f"{results_dir}normalized_{instance}.png" if normalize_tree_size else f"{results_dir}unnormalized_{instance}.png"
    format_plot(file_name, instance)


# Plot all instances
if normalize_tree_size:
    color = iter([cm.to_hex(plt.cm.tab20(i)) for i in range(20)])
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [1, 10]})
    fig.subplots_adjust(hspace=0.05)  # adjust space between axes
    fig.set_size_inches(12, 6)
    fig.set_dpi(100)

    for instance in lpbounds_dict.keys():
        if instance == 'ran14x18-disj-8':
            c_i = 'gold'
        else:
            c_i = next(color)
        set_label = True

        for seed in lpbounds_dict[instance].keys():
            if set_label:
                ax1.scatter(lpbounds_dict[instance][seed], tree_size_dict[instance][seed], color=c_i, label=instance, marker='o')
                ax2.scatter(lpbounds_dict[instance][seed], tree_size_dict[instance][seed], color=c_i, label=instance,
                            marker='o')
                set_label = False
            else:
                ax1.scatter(lpbounds_dict[instance][seed], tree_size_dict[instance][seed], color=c_i, marker='o')
                ax2.scatter(lpbounds_dict[instance][seed], tree_size_dict[instance][seed], color=c_i,
                            marker='o')

    ax1.set_ylim(5, 25)  # outliers only
    ax2.set_ylim(-1, 1.) # most of the data

    ax1.spines.bottom.set_visible(False)
    ax2.spines.top.set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()

    d = .5  # proportion of vertical to horizontal extent of the slanted line
    kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
                  linestyle="none", color='k', mec='k', mew=1, clip_on=False)
    ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
    file_name = f"{results_dir}all_instances_scatter.png"

    xlims = plt.gca().get_xlim()

    ax2.plot(np.linspace(0, 0.7, 11), np.full(11, 0), color='gray', linestyle='dashed',
             label='y = 0')
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.00))
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1.00))
    ax2.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    ax1.set_title(f"Rounds of cuts across instances and seeds", size=20, pad=20)
    ax2.set_ylabel("Change in Tree Size", size=20, labelpad=20)
    ax2.set_xlabel("Gap Closed by Cut", size=20, labelpad=20)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax1.tick_params(axis='y', which='major', labelsize=14)

    ax2.legend(bbox_to_anchor =(1.02,1), loc="upper left", fontsize=14, ncols=2)
    plt.tight_layout()

    plt.savefig(f"{results_dir}all_instances_scatter.png")
    plt.show()
    plt.close()
