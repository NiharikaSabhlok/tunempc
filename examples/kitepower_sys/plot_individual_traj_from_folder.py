#!/usr/bin/env python3
"""
Read (acc_reg, beta) pairs from a master CSV, find matching CSVs in a data folder,
and plot their x,y,z coordinates in 3D.

Assumptions:
- Master CSV has columns: 'acc_reg' and 'beta'.
- Data CSVs have columns for coordinates (case-insensitive): e.g. x,y,z or X,Y,Z or pos_x,pos_y,pos_z.
- Matching is done first by filename containing acc_reg/beta (e.g., "...beta=0.3_acc_reg=1e-3..."),
  then by reading candidate files and checking columns 'acc_reg'/'beta' inside (if present).
"""

from pathlib import Path
import re
import math
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (side-effect import for 3D)
import numpy as np
from matplotlib.lines import Line2D
import glob
import os

# --------------------------- Configuration ------------------------------------
# MASTER_CSV = Path("F:/Thesis/Parameter_sweep_analysis/Analysis_results/files/master_filtered.csv")
# DATA_DIR   = Path("F:/Thesis/Parameter_sweep_analysis/loop_t_27_N_54")  # contains the per-run CSVs
DATA_FOLDER = Path("F:/Thesis/Parameter_sweep_analysis/Analysis_results/data_folder_to_generate_graphs_for_param_analysis_chapter/time_val_refined")
FILENAME_GLOB = "*.csv"                             # adjust if needed
FLOAT_TOL = 1e-6                                    # tolerance for float equality
SAVE_FIG = Path("awe_xyz_3d_plot.png")
columns_to_extract = ['x_q10_0', 'x_q10_1', 'x_q10_2']
x_col = 'x_q10_0'
y_col = 'x_q10_1'
z_col = 'x_q10_2'
# folder_pattern = re.compile(r'Beta_and_acc_reg_wo_warmstarting_sol_beta([\d.eE+-]+)_acc_reg_([\d.eE+-]+)_N_(\d+)_time_(\d+\w*)')
folder_pattern = re.compile(r'Time_sweep_wo_warmstarting_sol_beta([\d.eE+-]+)_acc_reg_([\d.eE+-]+)_N_(\d+)_time_(\d+\w*)')
# ------------------------------------------------------------------------------

grouped_data = {}
main_param_val=[]

# def find_match(beta,acc_reg,T):
#     file_name = f"Beta_and_acc_reg_wo_warmstarting_sol_beta{beta}_acc_reg_{acc_reg}_N_{int(2*T)}_time_{int(T)}.csv"
#     matched_file = next(DATA_DIR.glob(file_name), None)
#     print(matched_file if matched_file else "File not found")
#     return matched_file

def parse_folder(name: str):
    m = folder_pattern.match(name)   # or .search(name) if it’s not anchored at start
    if not m:
        return None
    beta    = float(m.group(1))
    acc_reg = float(m.group(2))
    N       = int(m.group(3))
    t       = m.group(4)  # e.g. '120s' or '500ms'
    return beta, acc_reg, N, t

# master = pd.read_csv(MASTER_CSV)

for name in os.listdir(DATA_FOLDER):
    if name.lower().endswith(".csv"): 
    # if r["beta"]==0.1:
    #     beta = r["beta"]
    #     if beta in [0,1]:
    #         beta = int(beta)
    #     t = int(r["T"])
    #     acc = float(r["acc_reg"])
    #     if acc in [0,1]:
    #         acc = int(acc)
    #     else:
    #         acc=float(f"{acc:.1f}")
        # data_file=find_match(beta,acc,t)
        beta, acc, N, t = parse_folder(name)
        data_file=os.path.join(DATA_FOLDER, name)  
        try:
            df = pd.read_csv(data_file)
            df = df[[col for col in columns_to_extract if col in df.columns]]
            
            param_key = (float(beta), float(acc), int(t))
            if param_key not in grouped_data:
                grouped_data[param_key] = []
            label = f"beta= {beta}, acc={acc}, n={2*t}, t={t}"
            extracted_data = tuple(df[col] for col in columns_to_extract)
            grouped_entry = (*extracted_data, label)
            grouped_data[param_key].append(grouped_entry)
            main_param_val.append(param_key)
        except Exception as e:
            print(f"Error reading {data_file}: {e}")
        
        # --- where to save ---
# OUT_DIR = Path("F:/Thesis/Parameter_sweep_analysis/Analysis_results") / "plots_to_get_optimal_traj"
# OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR = DATA_FOLDER

# optional: params to remove
to_remove = set()  # e.g., {(0.3, 100000.0, 27)}
final_param_list = sorted(set(main_param_val) - set(to_remove), key=lambda t: (t[0], t[1], t[2]))
num_shades = max(1, len(final_param_list))

# --- style & colors ---
plt.style.use('seaborn-v0_8-muted')  # classy, readable
cmap = plt.cm.viridis
colors = cmap(np.linspace(0.2, 0.85, num_shades))  # avoid too-light colors

# --- 3D figure ---
# fig = plt.figure(figsize=(12, 8))
# ax = fig.add_subplot(111, projection='3d')

# # optional reference point
# ax.scatter(0, 0, 0, color='red', marker='o', s=40, label='origin')

legend_handles, legend_labels = [], []

# indices of your columns inside each grouped entry
ix = columns_to_extract.index(x_col)
iy = columns_to_extract.index(y_col)
iz = columns_to_extract.index(z_col)

for i, param in enumerate(final_param_list):
    beta, acc, t = param
    if param not in grouped_data:
        continue

    # plot all trajectories for this (beta, acc, t) in the same color
    first_label = True
    for entry in grouped_data[param]:
        legend_labels.clear()
        legend_handles.clear()
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # optional reference point
        ax.scatter(0, 0, 0, color='red', marker='o', s=10, label='origin')
        *data_columns, _label_text = entry
        x = data_columns[ix]
        y = data_columns[iy]
        z = data_columns[iz]

        # label only once per parameter group
        # label = f"β={beta:g}, a={acc:g}, t={t}" if first_label else None
        label = f"T={t}s" if first_label else None
        ax.plot(x, y, z, lw=1.8, color=colors[i], label=label)
        first_label = False

        # custom legend handle for consistency (in case no line got a label)
        legend_handles.append(Line2D([0], [0], color=colors[i], lw=2))
        # legend_labels.append(f"β={beta:g}, a={acc:g}, t={t}")
        legend_labels.append(f"T={t}s")
        # time_iterated=t

# axes & aesthetics
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_zlabel("z [m]")
        ax.set_title("3D Trajectories by (β, acc_reg, t)")
        ax.grid(True, which='both', alpha=0.25)

        # view angle
        ax.view_init(elev=28, azim=45)
        ax.set_box_aspect((1, 1, 1))  # equal-ish proportions

        # legend outside
        ax.legend(legend_handles, legend_labels, loc='center left',
                bbox_to_anchor=(1.02, 0.5), fontsize='small', title='Parameters', frameon=True)

        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.pane.set_edgecolor("none")
                axis.pane.set_alpha(0.05)

        # if labels:
        #     ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)


        plt.tight_layout()

        # save + show
        out_path = OUT_DIR / f"Parametric_sweep_trajectories_for_diff_T_{t}_acc_{acc}_beta_{beta}.png"
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        print(f"Saved plot to: {out_path}")
        plt.show()