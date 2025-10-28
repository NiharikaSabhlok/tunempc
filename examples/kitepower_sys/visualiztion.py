import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import re
import pandas as pd
import numpy as np

## To extract data from CSV and visulaize them for parameter study ##


# --- CONFIGURATION ---
# Flag to control filtering
filter_by_optimal_flag = True

plot_type = 'multi_plot_organise_by_param_beta'
folder_path = "F:/Thesis/Parameter_sweep_analysis/"  
subfolder = "T_31_N_62"
columns_to_extract = ['x_q10_0', 'x_q10_1', 'x_q10_2']
x_col = 'x_q10_0'
y_col = 'x_q10_1'
z_col = 'x_q10_2'

# to_remove = ['1.5','2','3','5']
to_remove = []
# --- REGEX to parse folder name ---
folder_pattern = re.compile(r'Beta_and_acc_reg_wo_warmstarting_sol_beta([\d.eE+-]+)_acc_reg_([\d.eE+-]+)_N_(\d+)_time_(\d+\w*)')
# Beta_and_acc_reg_wo_warmstarting_sol_beta0.1_acc_reg_0_N_72_time_36

# --- Store data for plotting ---
grouped_data = {}
main_param_val=[]
j=0


if filter_by_optimal_flag:
    stats_file = os.path.join(folder_path, subfolder, 'avg_power_and_twall_for_beta_and_Acc_N_62_T_31.txt')
    stats_df = pd.read_csv(stats_file)
    stats_df.columns = stats_df.columns.str.strip() # Remove leading/trailing whitespace from column names
    # stats_df['Beta'] = stats_df['Beta']
    # stats_df['acc_reg'] = stats_df['acc_reg']

# --- Walk through the directory ---
data_folder_path = os.path.join(folder_path,subfolder)
for file in os.listdir(data_folder_path):
    
    if file.endswith('.csv'):
        file_path = os.path.join(data_folder_path, file)
        # --- Extract values from the folder_name ---
        match = folder_pattern.search(file)
        if not match:
            print(FileExistsError,"Folder name doesn't match")
            continue
        
        beta, acc_reg, n, time_label = match.groups()
        print(f"beta: {beta}, acc_reg: {acc_reg}, n: {n}, time: {time_label}")
        
        if filter_by_optimal_flag:
            temp_beta = float(beta)
            temp_acc_reg = float(acc_reg)
            matched_row = stats_df[(stats_df['Beta'].astype(float) == temp_beta) & (stats_df['acc_reg'].astype(float) == temp_acc_reg)]
            print("Matched rows found:")
            print(matched_row)
 
            if matched_row.empty or not matched_row.iloc[0]['b_optimal']:
                print(f"Skipping {file} as it does not match optimal criteria.")
                continue
        
        try:
            df = pd.read_csv(file_path)
            df = df[[col for col in columns_to_extract if col in df.columns]]
            
            param_key = (float(beta), float(acc_reg))
            if param_key not in grouped_data:
                grouped_data[param_key] = []
            label = f"beta= {beta}, acc={acc_reg}, n={n}, t={time_label}"
            extracted_data = tuple(df[col] for col in columns_to_extract)
            grouped_entry = (*extracted_data, label)
            grouped_data[param_key].append(grouped_entry)
            main_param_val.append(param_key)
            

        except Exception as e:
            print(f"Error reading {file_path}: {e}")
        j=j+1   
        
if plot_type == 'multi_plot_organise_by_param_beta' :        
    # --- Organize by beta ---
    beta_groups = {}
    for (beta, acc), entries in grouped_data.items():
        if beta not in beta_groups:
            beta_groups[beta] = {}
        beta_groups[beta][acc] = entries        
    for beta in sorted(beta_groups.keys()):
        acc_dict = beta_groups[beta]
        acc_values = sorted(acc_dict.keys())
        num_shades = len(acc_dict)
        cmap = plt.cm.viridis
        colors = cmap(np.linspace(0.2, 1, num_shades))

        linewidth=np.linspace(3, 1, num_shades)
        transparency=np.linspace(0.5, 1, num_shades)
        # --- Create Figure ---
        fig = plt.figure(figsize=[10,6])
        # fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.02)

        # Add origin marker
        ax.scatter(0, 0, 0, color='red', marker='o', s=50, label='origin (0,0,0)')

        legend_handles = []
        legend_labels = []

        # --- Step 3: Plot all acc_reg for this beta ---
        for i, acc in enumerate(acc_values):
            entries = acc_dict[acc]
            color = colors[i]
            for j, entry in enumerate(entries):
                *data_columns, label_text = entry
                x = data_columns[columns_to_extract.index(x_col)]
                y = data_columns[columns_to_extract.index(y_col)]
                z = data_columns[columns_to_extract.index(z_col)]

                label = f"beta={beta}" if j == 0 else None
                ax.plot(x, y, z, label=label, color=color,linewidth=linewidth[i], alpha=transparency[i])

            # For legend
            legend_handles.append(Line2D([0], [0], color=color, lw=2))
            legend_labels.append(f"acc={acc}")

        # --- Final polish ---
        ax.set_xlabel('X(m)')
        ax.set_ylabel('Y(m)')
        ax.set_zlabel('Z(m)')
        ax.set_title(f'Kite Trajectory for different acceleration regularization values of beta={beta:.2g} and N={n}, time={time_label}s')
        ax.view_init(elev=15, azim=2)
        ax.legend(legend_handles, legend_labels, loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize='small', title='N')

        # plt.tight_layout()
        # plt.show()  # Or save to file
        # plt.savefig(f"beta_{beta:.2g}_plot.png", bbox_inches='tight')        
        
elif plot_type == 'multi_plot_organise_by_param_acc' :        
    # --- Organize by beta ---
    acc_groups = {}
    for (beta, acc), entries in grouped_data.items():
        if acc not in acc_groups:
            acc_groups[acc] = {}
        acc_groups[acc][beta] = entries        
    for acc in sorted(acc_groups.keys()):
        beta_dict = acc_groups[acc]
        beta_values = sorted(beta_dict.keys())
        num_shades = len(beta_values)
        cmap = plt.cm.viridis
        colors = cmap(np.linspace(0.2, 1, num_shades))

        linewidth=np.linspace(3, 1, num_shades)
        transparency=np.linspace(0.5, 1, num_shades)
        # --- Create Figure ---
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Add origin marker
        ax.scatter(0, 0, 0, color='red', marker='o', s=50, label='origin (0,0,0)')

        legend_handles = []
        legend_labels = []

        # --- Step 3: Plot all acc_reg for this beta ---
        for i, beta in enumerate(beta_values):
            entries = beta_dict[beta]
            color = colors[i]
            for j, entry in enumerate(entries):
                *data_columns, label_text = entry
                x = data_columns[columns_to_extract.index(x_col)]
                y = data_columns[columns_to_extract.index(y_col)]
                z = data_columns[columns_to_extract.index(z_col)]

                label = f"beta={beta:.2g}" if j == 0 else None
                ax.plot(x, y, z, label=label, color=color,linewidth=linewidth[i], alpha=transparency[i])

            # For legend
            legend_handles.append(Line2D([0], [0], color=color, lw=2))
            legend_labels.append(f"beta={beta:.2g}")

        # --- Final polish ---
        ax.set_xlabel('X(m)')
        ax.set_ylabel('Y(m)')
        ax.set_zlabel('Z(m)')
        ax.set_title(f'Kite Trajectory for acc_penalization={acc:.2g}')
        ax.view_init(elev=30, azim=45)
        ax.legend(legend_handles, legend_labels, loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize='small', title='beta')

        # plt.tight_layout()
        # plt.show()  # Or save to file
        # plt.savefig(f"beta_{beta:.2g}_plot.png", bbox_inches='tight')        
                
else:         
    # --- Sort Param values ---
    # sorted_param_val = sorted(grouped_data.keys(), key=lambda b: float(b))

    # final_param_list = [x for x in sorted_param_val if x not in to_remove]
    final_param_list = sorted(set(main_param_val) - set(to_remove), key=lambda t: (t[0], t[1]))
    num_shades = len(final_param_list)

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-muted')  # or 'ggplot', 'fivethirtyeight', etc.
    cmap = plt.cm.viridis  # or 'viridis', plasma', 'Blues', 'coolwarm', etc.
    colors = cmap(np.linspace(0.2, 1, num_shades))  # avoid too light shades
    fig=plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(0, 0, 0, color='red', marker='o', s=50, label='origin (0,0,0)')

    ## Custom legend handles ##
    legend_handles = []
    legend_labels = []

    for i, param in enumerate(final_param_list):
        beta, acc = param
        for i, param in enumerate(final_param_list):
            for entry in grouped_data[param]:
                *data_columns, label = entry
                x = data_columns[columns_to_extract.index(x_col)]
                y = data_columns[columns_to_extract.index(y_col)]
                z = data_columns[columns_to_extract.index(z_col)]
                
                # Add label only for the first entry of this (beta, acc) group
                label = f"β={beta:.2g}, a={acc:.2g}" if j == 0 else None
                ax.plot(x, y,z, label=f"acc_penalty={param}", color=colors[i])
            # Add custom handle for legend
            legend_handles.append(Line2D([0], [0], color=colors[i], lw=2))
            legend_labels.append(f"β={beta:.2g}, a={acc:.2g}")

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.title(f'{y_col} vs {x_col} for different configurations')

    # --- Move Legend Outside Plot ---
    ax.legend(legend_handles, legend_labels, loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize='small', title='Parameters')
    # plt.legend()
    plt.grid(True)
    # --- View Angle ---
    ax.view_init(elev=30, azim=45)
    # plt.tight_layout()
    plt.show()


plt.tight_layout()
plt.show()  # Or save to file