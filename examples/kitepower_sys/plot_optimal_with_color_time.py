import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('F:/Thesis/Parameter_sweep_analysis/Analysis_results/FINAL_PARAM_SWEEP/Time/TIME_SWEEP_WO_WARMSTARTING_RANGE_20-150.csv')  # Replace with your CSV file path

# Sort by T to ensure proper line plotting
df = df.sort_values('T').reset_index(drop=True)

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
fig.suptitle('Optimization Results Analysis', fontsize=16, fontweight='bold', y=0.995)

# Function to plot with conditional coloring
def plot_conditional(ax, x, y, b_optimal, ylabel, title):
    # Plot line segments with colors based on b_optimal
    for i in range(len(x) - 1):
        # Determine color: green if current OR next point is optimal
        if b_optimal.iloc[i] or b_optimal.iloc[i + 1]:
            color = '#2ecc71'  # Green
        else:
            color = '#e74c3c'  # Red
        
        ax.plot(x.iloc[i:i+2], y.iloc[i:i+2], 
                color=color, linewidth=2.5, alpha=0.8)
    
    # Plot markers
    colors = ['#2ecc71' if opt else '#e74c3c' for opt in b_optimal]
    ax.scatter(x, y, c=colors, s=100, zorder=5, 
               edgecolors='white', linewidth=1.5, alpha=0.9)
    
    # Styling
    ax.set_xlabel('Time Upper Bound [s]', fontsize=11 )
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11, pad=15)
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Feasible'),
        Patch(facecolor='#e74c3c', label='Infeasible')
    ]
    ax.legend(handles=legend_elements, loc='best', framealpha=0.9, 
              fontsize=10, edgecolor='gray')

# Plot 1: Trajectory Time vs T
plot_conditional(ax1, df['T'], df['final_time'], df['b_optimal'],
                'Output Time Period [s]', 'Trajectory Time vs Upper Time Bound')

# Plot 2: Average Power vs T
plot_conditional(ax2, df['T'], df['avg_power']/1000, df['b_optimal'],
                'Average Power [kW]', 'Average Power vs Upper Time Bound')

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