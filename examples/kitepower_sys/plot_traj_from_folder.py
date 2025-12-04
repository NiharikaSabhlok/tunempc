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
DATA_FOLDER = Path("F:/Thesis/Parameter_sweep_analysis/Analysis_results/data_folder_to_generate_graphs_for_param_analysis_chapter/beta_and_acc_reg_refined")
FILENAME_GLOB = "*.csv"                             # adjust if needed
FLOAT_TOL = 1e-6                                    # tolerance for float equality
SAVE_FIG = Path("awe_xyz_3d_plot.png")
columns_to_extract = ['x_q10_0', 'x_q10_1', 'x_q10_2']
x_col = 'x_q10_0'
y_col = 'x_q10_1'
z_col = 'x_q10_2'
folder_pattern = re.compile(r'Beta_and_acc_reg_sweep_wo_warmstarting_sol_beta([\d.eE+-]+)_acc_reg_([\d.eE+-]+)_N_(\d+)_time_(\d+)')
# folder_pattern = re.compile(r'Time_discretization_sweep_wo_warmstarting_sol_beta([\d.eE+-]+)_acc_reg_([\d.eE+-]+)_N_(\d+)_time_(\d+\w*)')
# ------------------------------------------------------------------------------

beta_opti = 0.05
acc_opti = 1e7
T_opti = '25'

grouped_data = {}
main_param_val=[]

# def find_match(beta,acc_reg,T):
#     file_name = f"Beta_and_acc_reg_wo_warmstarting_sol_beta{beta}_acc_reg_{acc_reg}_N_{int(2*T)}_time_{int(T)}.csv"
#     matched_file = next(DATA_DIR.glob(file_name), None)
#     print(matched_file if matched_file else "File not found")
#     return matched_file

def latexify():
    import matplotlib
    params_MPL_Tex = {
                'text.usetex': True,
                'font.family': 'serif',
                # Use 10pt font in plots, to match 10pt font in document
                "axes.labelsize": 10,
                "font.size": 10,
                # Make the legend/label fonts a little smaller
                "legend.fontsize": 8,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8
              }
    matplotlib.rcParams.update(params_MPL_Tex)

def draw_tethers_3d(ax, X, Y, Z, x_g=0.0, y_g=0.0, z_g=0.0, step=5,color_line="gray"):
    """
    Plot very light dotted lines from ground point (x_g,y_g,z_g)
    to each (X[k], Y[k], Z[k]) every `step` samples.
    """
    for k in range(0, len(X), step):
        ax.plot(
            [x_g, X[k]],
            [y_g, Y[k]],
            [z_g, Z[k]],
            linestyle=":",
            linewidth=0.7,
            color=color_line,
            alpha=0.5,
            zorder=0,
        )

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
latexify()
for name in os.listdir(DATA_FOLDER/"plot_traj"):
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
        if acc in [1e7,5e6,5e7] and beta in [0.05,0.15,0.2,0.4] and t in[T_opti]:
            data_file=os.path.join(DATA_FOLDER,"plot_traj" ,name)  
            try:
                df = pd.read_csv(data_file)
                df = df[[col for col in columns_to_extract if col in df.columns]]
                
                param_key = (float(beta), float(acc), int(t), int(N))
                if param_key not in grouped_data:
                    grouped_data[param_key] = []
                label = f"beta= {beta}, acc={acc}, n={N}, t={t}"
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
final_param_list = sorted(set(main_param_val) - set(to_remove), key=lambda t: (t[0], t[1], t[2], t[3]))
num_shades = max(1, len(final_param_list))

# --- style & colors ---
plt.style.use('seaborn-v0_8-muted')  # classy, readable
cmap = plt.cm.viridis
colors = cmap(np.linspace(0.2, 0.85, num_shades))  # avoid too-light colors

# --- 3D figure ---
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# optional reference point
ax.scatter(0, 0, 0, color='red', marker='o', s=40, label=r'Ground Station')

legend_handles, legend_labels = [], []

# indices of your columns inside each grouped entry
ix = columns_to_extract.index(x_col)
iy = columns_to_extract.index(y_col)
iz = columns_to_extract.index(z_col)

for i, param in enumerate(final_param_list):
    beta, acc, t,N = param
    if param not in grouped_data:
        continue

    # plot all trajectories for this (beta, acc, t) in the same color
    first_label = True
    for entry in grouped_data[param]:
        *data_columns, _label_text = entry
        x = data_columns[ix]
        y = data_columns[iy]
        z = data_columns[iz]

        # label only once per parameter group
        # label = f"β={beta:g}, a={acc:g}, t={t}" if first_label else None
        # label = label = rf"$\mathrm{{weight}}_{{\mathrm{{ddq}}}} = {acc/1e8}$" if first_label else None
        label = rf"$\beta={beta}, \mathrm{{weight}}_{{\mathrm{{ddq}}}}={acc/1e8}$" if first_label else None
        if acc==acc_opti and beta==beta_opti:
            ax.plot(x, y, z, lw=1.8, color=colors[i], label=label, linewidth=2.5)
            draw_tethers_3d(ax, x, y, z, step=5, color_line=colors[i])
        else:
            ax.plot(x, y, z, lw=1.8, color=colors[i], label=label, alpha=0.6)
            # draw_tethers_3d(ax, x, y, z, step=5, color_line=colors[i])
            
        first_label = False

    # custom legend handle for consistency (in case no line got a label)
    # legend_handles.append(Line2D([0], [0], color=colors[i], lw=2))
    # # legend_labels.append(f"β={beta:g}, a={acc:g}, t={t}")
    # legend_labels.append(rf"$\beta={beta}$")
    # time_iterated=t

# axes & aesthetics
ax.set_xlabel(r"$x\ \mathrm{[m]}$")
ax.set_ylabel(r"$y\ \mathrm{[m]}$")
ax.set_zlabel(r"$z\ \mathrm{[m]}$")
# ax.set_title("3D Trajectories by (β, acc_reg, t)")
ax.grid(True, which='both', alpha=0.25)

# view angle
# ax.view_init(elev=28, azim=45)
ax.view_init(elev=13, azim=35)
ax.set_box_aspect((1, 1, 1))  # equal-ish proportions

# legend outside
# ax.legend(legend_handles, legend_labels, loc='center left',
#           bbox_to_anchor=(1.02, 0.5), fontsize='small', title='Parameters', frameon=True)
ax.legend(loc='upper right',
          bbox_to_anchor=(0.95, 0.9), frameon=True,framealpha=0.8)

for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_edgecolor("none")
        axis.pane.set_alpha(0.05)

# if labels:
#     ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=True)


plt.tight_layout()

# save + show
out_path = OUT_DIR / f"refined_traj_for_T_{T_opti}_acc_{acc_opti}_beta_{beta_opti}.png"
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"Saved plot to: {out_path}")
plt.show()