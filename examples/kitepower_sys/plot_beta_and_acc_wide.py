import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('F:/Thesis/Parameter_sweep_analysis/Analysis_results/data_folder_to_generate_graphs_for_param_analysis_chapter/beta_and_acc_reg_refined/BETA_AND_ACC_REGULARIZATION_SWEEP_WO_WARMSTARTING_RANGE_wide.csv')  # Replace with your CSV file path

fixed_column = 'beta'#'acc_reg'
plotting_column = 'acc_reg'#'beta'
# Specify the beta value to filter
BETA_VALUE = 0.1  # Change this to your desired beta value

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

# Filter data for the specified beta
df_filtered = df[df[fixed_column] == BETA_VALUE].copy()


# Sort by acc_reg for proper line plotting
df_filtered = df_filtered.sort_values(plotting_column).reset_index(drop=True)
latexify()
# Check if data exists for this beta
if len(df_filtered) == 0:
    print(f"No data found for beta = {BETA_VALUE}")
    print(f"Available beta values: {sorted(df[fixed_column].unique())}")
    exit()

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
fig.suptitle(f'Performance Analysis for Acceleration Regularization = {BETA_VALUE}', 
             fontsize=16, fontweight='bold', y=0.995)

# Color scheme
# primary_color = '#3498db'  # Blue
# secondary_color = '#e67e22'  # Orange

# Plot 1: Average Power vs acc_reg
ax1.plot((df_filtered[plotting_column])/1e8, (df_filtered['avg_power'])/1000, 
         color="#36393b", linewidth=1.5, alpha=0.8, label=r"$\text{Average Power [kW]}$")
ax1.scatter((df_filtered[plotting_column])/1e8, (df_filtered['avg_power'])/1000, 
            c='#36393b', s=20, zorder=5, 
            edgecolors='white', linewidth=1, alpha=0.9)
ax1.set_xscale('log')
ax1.set_xlim(left=1e-5, right=1)
ax1.set_ylim(bottom=3)
ax1.set_xlabel(r"Acceleration Regularization $weight_{ddq}$")
ax1.set_ylabel(r"Average Power $P [kW]$")
ax1.set_title('Average Power vs Acceleration Regularization', 
              fontsize=13, fontweight='bold', pad=15)
# ax1.set_xscale('log')  # Logarithmic scale for x-axis
ax1.tick_params(axis='y', labelcolor='#36393b')
ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, which='both')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Highlight optimal points
# optimal_mask = df_filtered['b_optimal']
# if optimal_mask.any():
#     ax1.scatter(df_filtered[optimal_mask]['acc_reg'], 
#                 df_filtered[optimal_mask]['avg_power'],
#                 s=200, facecolors='none', edgecolors='#2ecc71', 
#                 linewidth=3, zorder=6, label='Optimal')
#     ax1.legend(loc='best', framealpha=0.9, fontsize=10, edgecolor='gray')

# Plot 2: Wall Time vs acc_reg
ax2.plot((df_filtered[plotting_column])/1e8, df_filtered['twall'], 
         color='#36393b', linewidth=1.5, alpha=0.8, label='Wall Time')
ax2.scatter((df_filtered[plotting_column])/1e8, df_filtered['twall'], 
            c='#36393b', s=20, zorder=5, 
            edgecolors='white', linewidth=1, alpha=0.9)
ax2.set_xscale('log')
ax2.set_xlim(left=1e-5, right=1)
ax2.set_ylim(bottom=0.0)
ax2.set_xlabel(r"Acceleration Regularization $weight_{ddq}$")
ax2.set_ylabel(r"$Computation Time [s]$")
ax2.set_title('Computation Time vs Acceleration Regularization', 
              fontsize=13, fontweight='bold', pad=15)
# ax2.set_xscale('log')  # Logarithmic scale for x-axis
ax2.tick_params(axis='y', labelcolor='#36393b')
ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.8, which='both')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# # Highlight optimal points
# if optimal_mask.any():
#     ax2.scatter(df_filtered[optimal_mask]['acc_reg'], 
#                 df_filtered[optimal_mask]['twall'],
#                 s=200, facecolors='none', edgecolors='#2ecc71', 
#                 linewidth=3, zorder=6, label='Optimal')
#     ax2.legend(loc='best', framealpha=0.9, fontsize=10, edgecolor='gray')

# Adjust layout and display
plt.tight_layout()
plt.show()

# Print summary statistics
print("\n" + "="*60)
print(f"SUMMARY STATISTICS FOR β = {BETA_VALUE}")
print("="*60)
print(f"Total data points: {len(df_filtered)}")
print(f"Optimal solutions: {df_filtered['b_optimal'].sum()}")
print(f"\nAcc_reg range: [{df_filtered['acc_reg'].min():.4f}, {df_filtered['acc_reg'].max():.4f}]")
print(f"Avg Power range: [{df_filtered['avg_power'].min():.4f}, {df_filtered['avg_power'].max():.4f}]")
print(f"Wall Time range: [{df_filtered['twall'].min():.4f}, {df_filtered['twall'].max():.4f}]")
print("\nCorrelation with acc_reg:")
print(f"  - Average Power: {df_filtered['acc_reg'].corr(df_filtered['avg_power']):.4f}")
print(f"  - Wall Time: {df_filtered['acc_reg'].corr(df_filtered['twall']):.4f}")
print("="*60)