import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('time_bound_for_traj_vs_final_time.txt')  # Replace with your file path

# Step 2: Preview the DataFrame (optional)
print(df.head())

# Step 3: Create a scatter plot (choose your own x and y columns)
# plt.scatter(df['beta_0'], df['avg_power'])  # Replace with actual column names
# plt.xlabel('beta_0')  # label x-axis
# plt.ylabel('Average Power')  # label y-axis
# plt.title('Average Power vs beta_0')
# plt.grid(True)
# plt.show()

plt.plot(df['time'], df['final_time'],marker='o')  # Replace with actual column names
plt.xlabel('Applied Time bound')  # label x-axis
plt.ylabel('Actual Trajectory time')  # label y-axis
plt.title('Applied Time bound vs Actual Trajectory time')

# plt.ylim(15200, 17000)

plt.tight_layout()
plt.grid(True)
plt.show()