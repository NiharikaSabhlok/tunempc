import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# CSV_PATH_1 = Path("F:/Thesis/Parameter_sweep_analysis/Analysis_results/data_folder_to_generate_graphs_for_param_analysis_chapter/Time_discretization")
# CSV_PATH_1 = Path("F:/Thesis/Parameter_sweep_analysis/Analysis_results/data_folder_to_generate_graphs_for_param_analysis_chapter/Time_discretization")
# CSV_PATH = Path("F:/Thesis/Parameter_sweep_analysis/Analysis_results/Controller/T_32_N_64_beta_0.45_accreg_80000000_NMPC_66_noise_config1")
CSV_PATH = Path("F:/Thesis/Parameter_sweep_analysis/Analysis_results/files/simulation_files/T_25_N_50_beta_0.05_accreg_10000000_NMPC_50_config1_DEBUG_TMPC_2_with_disturbance_only1")
scaling_x=13.5916
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
    
# latexify()

df_1 = pd.read_csv(CSV_PATH/'ref_x.csv')  # Replace with your file path
df_2 = pd.read_csv(CSV_PATH/'EMPC_x.csv')
df_3 = pd.read_csv(CSV_PATH/'TMPC_2_x.csv')
df_4 = pd.read_csv(CSV_PATH/'TUNEMPC_2_x.csv')
df_5 = pd.read_csv(CSV_PATH/'TUNEMPC_x.csv')
# df_2 = pd.read_csv(CSV_PATH/'TUNEMPC_traj_x.csv', header=None)

fig = plt.figure()
ax  = fig.add_subplot(111, projection='3d')
ax.scatter(0, 0, 0, color='black', marker='o', s=10 ) #label='origin'
# ax.plot(df_1['x0'], df_1['x1'], df_1['x2'], label="Reference",color="#000000",  linewidth=2)

# ax.plot(df_2['x0'], df_2['x1'], df_2['x2'], color="#462e9e", label="EMPC", linewidth=1.5, linestyle='-')
# ax.plot(df_3['x0'], df_3['x1'], df_3['x2'], color="#9B3CA6", label="TMPC_2", linewidth=1.5, linestyle='--')
# ax.plot(df_4['x0'], df_4['x1'], df_4['x2'], color="#540a35", label="TUNEMPC_2", linewidth=1.5, linestyle='-.')
# ax.plot(df_5['x0'], df_5['x1'], df_5['x2'], color="#3b8522", label="TUNEMPC", linewidth=1.5, linestyle='--')
ax.plot(df_1['x0'], df_1['x1'], df_1['x2'], label="Reference",color="#000000",  linewidth=2)

ax.plot(df_2['x0'], df_2['x1'], df_2['x2'], color="#462e9e", label="EMPC", linewidth=1.5, linestyle='-')
ax.plot(df_3['x0'], df_3['x1'], df_3['x2'], color="#9B3CA6", label="TMPC_2", linewidth=1.5, linestyle='--')
ax.plot(df_4['x0'], df_4['x1'], df_4['x2'], color="#540a35", label="TUNEMPC_2", linewidth=1.5, linestyle='-.')
ax.plot(df_5['x0'], df_5['x1'], df_5['x2'], color="#3b8522", label="TUNEMPC", linewidth=1.5, linestyle='--')

# ax.plot(df_2[0], df_2[1], df_2[2], color="#37560e", label="TMPC_2", linewidth=1.5, linestyle='-')
# ax.plot(df_2.iloc[:, 0]*scaling_x, df_2.iloc[:, 1]*scaling_x, df_2.iloc[:, 2]*scaling_x,
#         color="#37560e", label="TuneMPC", linewidth=1.5, linestyle='-')


ax.set_title("3D Trajectory: controllers vs reference")
ax.set_xlabel(r"x [m]"); ax.set_ylabel(r"y[m]"); ax.set_zlabel(r"z[m]")
ax.grid(True, which="both", alpha=0.25)
ax.view_init(elev=28, azim=45) #14,38
ax.set_box_aspect((1, 1, 1))  # equal-ish proportions
ax.legend()
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
    axis.pane.set_edgecolor("none")
    axis.pane.set_alpha(0.05)
plt.tight_layout()
plt.show()