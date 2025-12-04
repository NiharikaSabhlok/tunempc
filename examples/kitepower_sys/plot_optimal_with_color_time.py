import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


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
    
latexify()
ROOT=Path("F:/Thesis/Parameter_sweep_analysis/Analysis_results/data_folder_to_generate_graphs_for_param_analysis_chapter")
# Read the CSV file
df = pd.read_csv(ROOT/"time_val_refined"/"time_sweep_refined.csv")  # Replace with your CSV file path

# Sort by T to ensure proper line plotting
df = df.sort_values('T').reset_index(drop=True)

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
# fig.suptitle('Optimization Results Analysis', fontsize=16, fontweight='bold', y=0.995)

# Function to plot with conditional coloring
def plot_conditional(ax, x, y, b_optimal, ylabel):
    # Plot line segments with colors based on b_optimal
    for i in range(len(x) - 1):
        # Determine color: green if current OR next point is optimal
        if b_optimal.iloc[i] or b_optimal.iloc[i + 1]:
            color = "#127e3f"  # Green
        else:
            color = "#942c20"  # Red
        
        ax.plot(x.iloc[i:i+2], y.iloc[i:i+2], 
                color=color, linewidth=1.5, alpha=0.8)
    
    # Plot markers
    colors = ['#127e3f' if opt else '#942c20' for opt in b_optimal]
    ax.scatter(x, y, c=colors, s=100, zorder=5, 
               edgecolors='white', linewidth=1.5, alpha=0.9)
    
    # Styling
    ax.set_xlabel(r'Time Period [s]', fontsize=11 )
    ax.set_ylabel(ylabel, fontsize=11)
    # ax.set_title(title, fontsize=11, pad=15)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#127e3f", label='Feasible'),
        Patch(facecolor='#942c20', label='Infeasible')
    ]
    ax.legend(handles=legend_elements, loc='best', framealpha=0.9, 
              fontsize=10, edgecolor='gray')

# Plot 1: Trajectory Time vs T
plot_conditional(ax1, df['T'], df['twall'], df['b_optimal'],
                r'Computation Time[s]')

# Plot 2: Average Power vs T
plot_conditional(ax2, df['T'], df['avg_power']/1000, df['b_optimal'],
                r'Average Power [kW]')

# Adjust layout and display
plt.tight_layout()
plt.show()

# Optional: Print summary statistics
print("\n" + "="*50)
print("SUMMARY STATISTICS")
print("="*50)
print(f"Total data points: {len(df)}")
print(f"Optimal solutions: {df['b_optimal'].sum()} ({df['b_optimal'].sum()/len(df)*100:.1f}%)")
print(f"Trajectory Time - Mean: {df['final_time'].mean():.3f}, Std: {df['final_time'].std():.3f}")
print(f"Average Power - Mean: {df['avg_power'].mean():.3f}, Std: {df['avg_power'].std():.3f}")
print("="*50)