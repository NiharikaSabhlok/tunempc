import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# CSV_PATH_1 = Path("F:/Thesis/Parameter_sweep_analysis/Analysis_results/data_folder_to_generate_graphs_for_param_analysis_chapter/Time_discretization")
# CSV_PATH_1 = Path("F:/Thesis/Parameter_sweep_analysis/Analysis_results/data_folder_to_generate_graphs_for_param_analysis_chapter/Time_discretization")
# CSV_PATH = Path("F:/Thesis/Parameter_sweep_analysis/Analysis_results/Controller/T_32_N_64_beta_0.45_accreg_80000000_NMPC_66_noise_config1")
CSV_PATH = Path("F:/Thesis/Parameter_sweep_analysis/Analysis_results/files/simulation_files/T_25_N_50_beta_0.05_accreg_10000000_NMPC_50_config1_WITH_DISTURBANCE_with_step")

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

# df_1 = pd.read_csv(CSV_PATH/'TMPC_2_x.csv')  # Replace with your file path
# df_2 = pd.read_csv(CSV_PATH/'ref_x.csv')

df_1 = pd.read_csv(CSV_PATH/'ref_u.csv')  # Replace with your file path
df_2 = pd.read_csv(CSV_PATH/'EMPC_u.csv')
df_3 = pd.read_csv(CSV_PATH/'TMPC_2_u.csv')
df_4 = pd.read_csv(CSV_PATH/'TUNEMPC_2_u.csv')
df_5 = pd.read_csv(CSV_PATH/'TUNEMPC_u.csv')

fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)
# axes[0, 0].plot(df_1['t_cycle'], df_1['x0'],color="#4b738e", linewidth=2)
# axes[0, 0].plot(df_2['t_cycle'], df_2['x0'],color="#598a19", linewidth=1)
# axes[0, 0].set_title("x0")

# axes[0, 1].plot(df_1['t_cycle'], df_1['x1'],color="#4b738e", linewidth=2)
# axes[0, 1].plot(df_2['t_cycle'], df_2['x1'],color="#598a19", linewidth=1)
# axes[0, 1].set_title("x1")

# axes[0, 2].plot(df_1['t_cycle'], df_1['x2'],color="#4b738e", linewidth=2)
# axes[0, 2].plot(df_2['t_cycle'], df_2['x2'],color="#598a19", linewidth=1)
# axes[0, 2].set_title("x2")

axes[0].step(df_1['t_cycle'], df_1['u0'],color="#000000", linewidth=2)
axes[0].step(df_2['t_cycle'], df_2['u0'],color="#462e9e", linewidth=1, linestyle='-')
axes[0].step(df_3['t_cycle'], df_3['u0'],color="#9B3CA6", linewidth=1, linestyle='--')
axes[0].step(df_4['t_cycle'], df_4['u0'],color="#540a35", linewidth=1, linestyle='-')
axes[0].step(df_5['t_cycle'], df_5['u0'],color="#3b8522", linewidth=1, linestyle='--')
axes[0].set_title("u0")

axes[1].step(df_1['t_cycle'], df_1['u1'],color="#000000", linewidth=2)
axes[1].step(df_2['t_cycle'], df_2['u1'],color="#462e9e", linewidth=1, linestyle='-')
axes[1].step(df_3['t_cycle'], df_3['u1'],color="#9B3CA6", linewidth=1, linestyle='--')
axes[1].step(df_4['t_cycle'], df_4['u1'],color="#540a35", linewidth=1, linestyle='-')
axes[1].step(df_5['t_cycle'], df_5['u1'],color="#3b8522", linewidth=1, linestyle='--')
axes[1].set_title("u1")

axes[2].step(df_1['t_cycle'], df_1['u2'],color="#000000", linewidth=2)
axes[2].step(df_2['t_cycle'], df_2['u2'],color="#462e9e", linewidth=1, linestyle='-')
axes[2].step(df_2['t_cycle'], df_3['u2'],color="#9B3CA6", linewidth=1, linestyle='--')
axes[2].step(df_2['t_cycle'], df_4['u2'],color="#540a35", linewidth=1, linestyle='-')
axes[2].step(df_2['t_cycle'], df_5['u2'],color="#3b8522", linewidth=1, linestyle='--')
axes[2].set_title("u2")

# axes[1, 0].plot(df_1['t_cycle'], df_1['x3'],color="#4b738e", linewidth=2)
# axes[1, 0].plot(df_2['t_cycle'], df_2['x3'],color="#598a19", linewidth=1)
# axes[1, 0].set_title("x3")

# axes[1, 1].plot(df_1['t_cycle'], df_1['x4'],color="#4b738e", linewidth=2)
# axes[1, 1].plot(df_2['t_cycle'], df_2['x4'],color="#598a19", linewidth=1)
# axes[1, 1].set_title("x4")

# axes[1, 2].plot(df_1['t_cycle'], df_1['x5'],color="#4b738e", linewidth=2)
# axes[1, 2].plot(df_2['t_cycle'], df_2['x5'],color="#598a19", linewidth=1)
# axes[1, 2].set_title("x5")

# axes[1, 0].plot(df_2['t_cycle'], df_2['x0'],color="#598a19")
# axes[1, 0].set_title("x0")

# axes[1, 1].plot(df_2['t_cycle'], df_2['x1'],color="#598a19")
# axes[1, 1].set_title("x1")

# axes[1, 2].plot(df_2['t_cycle'], df_2['x2'],color="#598a19")
# axes[1, 2].set_title("x2")

# axes[1, 3].plot(df_2['t_cycle'], df_2['x3'],color="#598a19")
# axes[1, 3].set_title("x3")

# axes[1, 4].plot(df_2['t_cycle'], df_2['x4'],color="#598a19")
# axes[1, 4].set_title("x4")

# axes[1, 5].plot(df_2['t_cycle'], df_2['x5'],color="#598a19")
# axes[1, 5].set_title("x5")

# plt.plot(df_1['t_cycle'], df_1['x0'],color="#4b738e", linewidth=2, alpha=0.8, label='reference')  # Replace with actual column names
# plt.plot(df_2['t_cycle'], df_2['x0'],color="#598a19", linewidth=1, alpha=0.8, label='TMPC_2')  # Replace with actual column names
# plt.xlabel(r'Time [cycle]')  # label x-axis
# plt.ylabel(r'x0')  # label y-axis
plt.legend()
# plt.title('Computation time vs Number of Samples for time period of 25 sec')

# plt.ylim(15200, 17000)

plt.tight_layout()
# plt.grid(True)
plt.show()