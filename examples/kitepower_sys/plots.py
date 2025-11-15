import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

CSV_PATH = Path("F:/Thesis/Parameter_sweep_analysis/Analysis_results/data_folder_to_generate_graphs_for_param_analysis_chapter/Time_discretization/CSVs")
df = pd.read_csv(CSV_PATH/'time_discretization_master.csv')  # Replace with your file path

# Step 2: Preview the DataFrame (optional)
print(df.head())

# Step 3: Create a scatter plot (choose your own x and y columns)
# plt.scatter(df['beta_0'], df['avg_power'])  # Replace with actual column names
# plt.xlabel('beta_0')  # label x-axis
# plt.ylabel('Average Power')  # label y-axis
# plt.title('Average Power vs beta_0')
# plt.grid(True)
# plt.show()

plt.plot(df['N'], df['twall'],marker='o')  # Replace with actual column names
plt.xlabel('Number of Samples N')  # label x-axis
plt.ylabel('Computation time [s]')  # label y-axis
plt.title('Computation time vs Number of Samples for time period of 25 sec')

# plt.ylim(15200, 17000)

plt.tight_layout()
plt.grid(True)
plt.show()