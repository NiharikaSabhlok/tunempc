import awebox as awe
# import matplotlib.pyplot as plt
import numpy as np
import awebox.tools.print_operations as print_op
import awebox.opts.kite_data.kitepower_lei_data as kitepower_lei_data
import casadi as ca
import casadi.tools as ct
import time
import pandas as pd
import copy 
import pickle
import awebox.tools.integrator_routines as awe_integrators
from awebox.logger.logger import Logger as awelogger
from tunempc.logger import Logger
from pathlib import Path
# Logger.logger.setLevel('DEBUG')
# awelogger.logger.setLevel('DEBUG')


############# PARAMETERS #################
TWO_PARAM_ANALYSIS=True
avg_power_array=[]
windings=1
time_per_winding=38
intervals_per_winding = 2*time_per_winding
N=int(intervals_per_winding * windings)
beta_0 = 0.1
acc_reg = 1
filefolder = '/pfs/data6/home/fr/fr_fr/fr_ns591/code/examples/parameter_sweep_analysis/beta_and_acc_reg_refined/'
file_name = 'Beta_and_acc_reg_sweep_wo_warmstarting'
ROOT = Path("/pfs/data6/home/fr/fr_fr/fr_ns591/code/examples/parameter_sweep_analysis/beta_and_acc_reg_refined")
CSV_PATH = ROOT/f"BETA_AND_ACC_REGULARIZATION_SWEEP_WO_WARMSTARTING_RANGE_{time_per_winding}.csv"
options = {}
options['user_options.system_model.architecture'] = {1: 0}
options['user_options.kite_standard'] = kitepower_lei_data.data_dict()
options['user_options.system_model.wing_type'] = 'LEI'
options['user_options.system_model.kite_dof'] = 3

# indicate desired operation mode
options['user_options.trajectory.type'] = 'power_cycle'
options['user_options.trajectory.system_type'] = 'lift_mode'
# windings = 5
options['user_options.trajectory.lift_mode.windings'] = windings

# indicate desired environment
options['params.wind.z_ref'] = 100.0
options['params.wind.power_wind.exp_ref'] = 0.15
options['user_options.wind.model'] = 'power'
options['user_options.wind.u_ref'] = 6.

# coefficient boundaries
options['model.system_bounds.x.coeff'] =  [np.array([-0.6, 0.]), np.array([0.6, 1.])]
options['model.system_bounds.u.dcoeff'] =  [np.array([-.08, -1]), np.array([.08, 1])]

# indicate numerical nlp details
# here: nlp discretization, with a zero-order-hold control parametrization, and
# a simple phase-fixing routine. also, specify a linear solver to perform the Newton-steps
# within ipopt.
options['nlp.n_k'] = N
options['nlp.collocation.u_param'] = 'zoh'
options['user_options.trajectory.lift_mode.phase_fix'] = 'simple' # 'simple' # 'single_reelout'
options['solver.linear_solver'] = 'mumps'  # if HSL is installed, otherwise 'mumps'
options['model.system_bounds.x.ddl_t'] = [-2.0, 2.0]
options['model.system_bounds.theta.t_f'] = [0, (windings*time_per_winding)] ## +2*(time_per_winding/intervals_per_winding)
# options['nlp.phase_fix'] = 'simple'
options['nlp.phase_fix_reelout'] = 0.7
options['solver.cost.beta.0'] = beta_0
options['solver.weights.ddq'] =  acc_reg

options['model.model_bounds.acceleration.include']  = False
options['model.model_bounds.aero_validity.include']  = False
options['model.model_bounds.tether_stress.include']  = True
# (experimental) set to "True" to significantly (factor 5 to 10) decrease construction time
# note: this may result in slightly slower solution timings
options['nlp.compile_subfunctions'] = False


# initialization
options['solver.initialization.shape'] = 'lemniscate'
options['solver.initialization.lemniscate.az_width'] = 20*np.pi/180.
options['solver.initialization.lemniscate.el_width'] = 8*np.pi/180.
options['solver.initialization.inclination_deg'] = 30.
options['solver.initialization.groundspeed'] = 40.
options['solver.initialization.theta.diam_t'] = 14e-3
options['solver.initialization.l_t'] = 500.0
options['solver.max_iter_hippo'] = 500
options['solver.max_iter'] = 500
options['visualization.cosmetics.plot_ref'] = False
trial_default = awe.Trial(options, 'Kitepower_LEI')
trial_default.build()
    
    
if TWO_PARAM_ANALYSIS :        
    # for idy, no_of_intervals in enumerate([N]):
    #     for idx, time_per_interval in enumerate([25,35,45,50,60,70,80,90,100,120,140,150,160,180]):
    for idy, beta_0 in enumerate([0, 0.01, 0.02,0.05, 0.07, 0.08, 0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]):
        for idx, acc_reg in enumerate([0,1,5,1e2, 5e3, 5e4, 5e5,1e6,3e6,5e6,8e6,1e7,3e7,5e7,8e7,1e8,5e8, 1e9, 5e9]):        
            # Yaw-rate regularization
            # options['nlp.n_k'] = no_of_intervals
            # Acceleration penaization
            # options['model.system_bounds.theta.t_f'] = [0, time_per_interval]
            options['solver.cost.beta.0'] = beta_0
            options['solver.weights.ddq'] =  acc_reg

            if idx == 0 and idy == 0:
                # trial_default.optimize(options_seed = options, intermediate_solve=True)
                # intermediate_sol = copy.deepcopy(trial_default.solution_dict)
                # trial_default.optimize(options_seed = options, warmstart_file = intermediate_sol, intermediate_solve=False, recalibrate_viz = True,final_homotopy_step = 'final')
                trial_default.optimize(options_seed = options, recalibrate_viz = True, final_homotopy_step = 'final')
                plot_dict = trial_default.visualization.plot_dict
                avg_power = plot_dict['power_and_performance']['avg_power']
                final_time = trial_default.optimization.V_opt['theta','t_f']
                t_wall = trial_default.optimization.stats['t_wall_total']
                b_optimal = trial_default.optimization.stats['success']
                # t_wall = opti.stats['t_wall_total']
                b_optimal = trial_default.optimization.stats['success']
                avg_power_array.append([beta_0, acc_reg, N, time_per_winding, avg_power.full().item(),t_wall,final_time.full().item(), b_optimal])
                
                
            else:
                # trial_default.optimize(options_seed = options, warmstart_file = intermediate_sol, intermediate_solve=True)
                # intermediate_sol = copy.deepcopy(trial_default.solution_dict)
                # trial_default.optimize(options_seed = options, warmstart_file = trial_default.solution_dict, intermediate_solve=False, recalibrate_viz = True)
                trial_default.optimize(options_seed = options, recalibrate_viz = True, final_homotopy_step = 'final')
                plot_dict = trial_default.visualization.plot_dict
                avg_power = plot_dict['power_and_performance']['avg_power']
                final_time = trial_default.optimization.V_opt['theta','t_f']
                b_optimal = trial_default.optimization.stats['success']
                t_wall = trial_default.optimization.stats['t_wall_total']
                avg_power_array.append([beta_0, acc_reg, N, time_per_winding, avg_power.full().item(),t_wall,final_time.full().item(),b_optimal])
            trial_default.write_to_csv(filename=filefolder + '{}_sol_beta{}_acc_reg_{}_N_{}_time_{}'.format(file_name, beta_0, acc_reg, N, time_per_winding), frequency=10., rotation_representation='dcm')
            print("Wrote file to the folder")
        
    avg_power_df = pd.DataFrame(avg_power_array, columns=["beta","acc_reg","N", "T", "avg_power","twall","final_time","b_optimal"])
    # Write to a .txt file as comma-separated values
    avg_power_df.to_csv(CSV_PATH, index=False, header=True, sep=',',mode='a')
    print("Wrote final CSV File")
    # Plot avg_power vs beta_0
    # plt.figure()
    # plt.plot(avg_power_df["Intervals"], avg_power_df["avg_power"], marker='o')
    # plt.xlabel("No. of Intervals")
    # plt.ylabel("Average Power")
    # plt.title("Average Power vs number of samples")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    
    # # Plot avg_power vs beta_0
    # plt.figure()
    # plt.plot(avg_power_df["Intervals"], avg_power_df["twall"], marker='o')
    # plt.xlabel("No. of Intervals")
    # plt.ylabel("Computation Time")
    # plt.title("Number of Samples vs Computation time")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    
    # Plot avg_power vs beta_0
    # plt.figure()
    # plt.plot(avg_power_df["time"], avg_power_df["avg_power"], marker='o')
    # plt.xlabel("Time per winding")
    # plt.ylabel("Average Power")
    # plt.title("Time per winding vs Average Power")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    # # Plot avg_power vs acc_reg
    # plt.figure()
    # plt.plot(avg_power_df["time"], avg_power_df["twall"], marker='o')
    # plt.xlabel("Time per winding")
    # plt.ylabel("Computation Time")
    # plt.title("Time per winding vs Computation Time")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    
    # plt.figure()
    # plt.plot(avg_power_df["time"], avg_power_df["final_time"], marker='o')
    # plt.xlabel("Applied Time Bound")
    # plt.ylabel("Actual Time of the trajectory")
    # plt.title("Applied Time Bound vs Actual Time of the trajectory")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()
    
else :
    # for idx, param_val in enumerate([1e6,1e7,1e8, 2e8, 5e8, 1e9,2e9, 2e9, 4e9, 5e9, 7e9,1e10]):
    for idx, param_val in enumerate([20,25,30,35,40,45,50,55,60,70,80,90,100,110,120,130,140,150]):

        # Yaw-rate regularization
        # options['solver.weights.ddq'] =  param_val
        options['model.system_bounds.theta.t_f'] = [0, param_val]

        if idx == 0:
            # trial_default.optimize(options_seed = options, recalibrate_viz = True, final_homotopy_step = 'final')
            # intermediate_sol = copy.deepcopy(trial_default.solution_dict)
            # trial_default.optimize(options_seed = options, warmstart_file = intermediate_sol, intermediate_solve=False, recalibrate_viz = True)
            trial_default.optimize(options_seed = options, recalibrate_viz = True, final_homotopy_step = 'final')
            plot_dict = trial_default.visualization.plot_dict
            avg_power = plot_dict['power_and_performance']['avg_power']
            final_time = trial_default.optimization.V_opt['theta','t_f']
            t_wall = trial_default.optimization.stats['t_wall_total']
            b_optimal = trial_default.optimization.stats['success']
            # t_wall = opti.stats['t_wall_total']
            b_optimal = trial_default.optimization.stats['success']
            avg_power_array.append([beta_0, acc_reg, N, time_per_winding, avg_power.full().item(),t_wall,final_time.full().item(), b_optimal])
                
        
        else:
            # trial_default.optimize(options_seed = options, warmstart_file = intermediate_sol, intermediate_solve=True)
            # intermediate_sol = copy.deepcopy(trial_default.solution_dict)
            # trial_default.optimize(options_seed = options, warmstart_file = trial_default.solution_dict, intermediate_solve=False, recalibrate_viz = True)
            trial_default.optimize(options_seed = options, recalibrate_viz = True, final_homotopy_step = 'final')
            plot_dict = trial_default.visualization.plot_dict
            avg_power = plot_dict['power_and_performance']['avg_power']
            final_time = trial_default.optimization.V_opt['theta','t_f']
            b_optimal = trial_default.optimization.stats['success']
            t_wall = trial_default.optimization.stats['t_wall_total']
            avg_power_array.append([beta_0, acc_reg, N, time_per_winding, avg_power.full().item(),t_wall,final_time.full().item(),b_optimal])

        trial_default.write_to_csv(filename=filefolder + '{}_sol_beta{}_acc_reg_{}_N_{}_time_{}'.format(file_name, beta_0, acc_reg,N, param_val), frequency=10., rotation_representation='dcm')
        print("Wrote file to the folder")
    acg_power_df = pd.DataFrame(avg_power_array, columns=["beta","acc_reg","N", "T", "avg_power","twall","final_time","b_optimal"])
    # Write to a .txt file as comma-separated values
    acg_power_df.to_csv(CSV_PATH, index=False, header=True, sep=',')
    print("Wrote CSV File")
    