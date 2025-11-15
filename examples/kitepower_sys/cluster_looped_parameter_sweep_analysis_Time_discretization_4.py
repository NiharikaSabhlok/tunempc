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
TWO_PARAM_ANALYSIS=False
avg_power_array=[]
windings=1
time_per_winding=25
########## CHNAGE INTERVAL #########
intervals_per_winding = 100
####################################
N=int(intervals_per_winding * windings)
beta_0 = 0.05
acc_reg = 5e6
filefolder = '/pfs/data6/home/fr/fr_fr/fr_ns591/code/examples/parameter_sweep_analysis/Time_discretization/'
file_name = 'Time_discretization_sweep_wo_warmstarting'
ROOT = Path("/pfs/data6/home/fr/fr_fr/fr_ns591/code/examples/parameter_sweep_analysis/Time_discretization")
CSV_PATH = ROOT /f"TIME_DISCRETIZATION_SWEEP_WO_WARMSTARTING_RANGE_25-150_N_{N}.csv"
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
    for idy, beta_0 in enumerate([0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]):
        for idx, acc_reg in enumerate([0,1,1e6,5e6,1e7,5e7,8e7,1e8]):        
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
            
            ############################### PREPARE INPUTS ################################
            sol = {}
            sol['model'] = trial_default.generate_optimal_model()
            sol['l_t']   = trial_default.optimization.V_opt['x',0,'l_t']
            sol['t_f'] = trial_default.optimization.V_opt['theta','t_f']
            w_init = []
            x_val= []
            u_val =[]
            for k in range(N):
                w_init.append(trial_default.optimization.V_opt['x',k])
                w_init.append(trial_default.optimization.V_opt['u', k][3:6])
                x_val.append(trial_default.optimization.V_opt['x',k])
                u_val.append(trial_default.optimization.V_opt['u', k][3:6])
            sol['w0'] = ca.vertcat(*w_init)
            w_0 = sol['w0']
            model = sol['model']
            l_t = sol['l_t']
            x_shape = model['dae']['x'].shape
            x_shape = (x_shape[0], x_shape[1])
            x = ca.MX.sym('x',*x_shape)
            x_awe = x
            x_0 = w_0[:x_shape[0]]
            xh = ca.MX.sym('x', x_shape[0]-1, x_shape[1])
            xh_awe = ca.vertcat(xh, 0)

            # remove fictitious forces, tether jerk...
            u_shape = model['dae']['p'].shape
            u_shape = (u_shape[0]-3, u_shape[1])
            u = ca.MX.sym('u',*u_shape)
            u_awe = ct.vertcat(0.0,0.0,0.0,u)
            u_0 = ct.vertcat(ca.DM.zeros(3,1), w_0[x_shape[0]:x_shape[0]+u_shape[0]])


            nx = x.shape[0]
            nu = u_shape[0]

            # initial guess

            # save time-continuous dynamics
            xdot = ca.MX.sym('xdot', x.shape[0])
            xdot_awe = xdot
            z_0 = model['rootfinder'](0.1, x_0, u_0) #(12,1)
            z = model['rootfinder'](z_0, x_awe, u_awe) #(12,1)
            integrator = awe_integrators.rk4root(
                'F',
                model['dae'],
                model['rootfinder'],
                {'tf': 1/N, 'number_of_finite_elements':20})
            xf = integrator(x0=x_awe, p=u_awe, z0 = z_0)['xf']
            qf = integrator(x0=x_awe, p=u_awe, z0 = z_0)['qf']
            constraints = ca.vertcat(
                -model['constraints'](x_awe, u_awe,z),
                -model['var_bounds_fun'](x_awe, u_awe,z)
            )
            constraints_new = []
            for i in range(constraints.shape[0]):
                if True in ca.which_depends(constraints[i],ca.vertcat(x,u)):
                    constraints_new.append(constraints[i])

            h = ca.Function('h', [x,u], [ca.vertcat(*constraints_new)])
                

            sys = {
                'f' : ca.Function('F',[x,u],[xf,qf],['x0','p'],['xf','qf']),
                'h' : ca.Function('h', [x,u], [ca.vertcat(*constraints_new)])
            }
            
            cost = ca.Function(
                'cost',
                [x,u],
                [qf] #+ extra_regularization
            )
            rm_indeces = []
            z = ca.MX.sym('z', model['dae']['z']['z'].shape[0])
            indeces = [k for k in range(x_awe.shape[0]+z.shape[0]) if k not in rm_indeces]
            alg = model['dae']['alg'][indeces]
            alg_energy = ca.vertcat(alg[:-1], model['dae']['z']['xdot'][-1] - model['dae']['quad']/model['t_f'])
            alg_fun = ca.Function('alg_fun',[model['dae']['x'],model['dae']['p'],model['dae']['z']],[alg_energy])
            #########################################################



            dyn = ca.Function(
                'dae',
                [xdot,x,u,z],
                [alg_fun(x, u_awe, ct.vertcat(xdot, z))],
                ['xdot','x','u','z'],
                ['dyn'])

            pickle_filename = f"{filefolder}\kitepower_user_input_{windings*intervals_per_winding}_w_{windings}_tpw_{time_per_winding}_beta0_{beta_0}_acc_reg_{acc_reg}.pkl"
            x_val_np = [ca.DM(x).full().flatten().tolist() for x in x_val]
            u_val_np = [ca.DM(u).full().flatten().tolist() for u in u_val]
            with open(pickle_filename,'wb') as outfile:
                pickle.dump({
                    'f': sys['f'],
                    'l': cost,
                    'h': h,
                    'p': N,
                    'w0': w_0,
                    'dyn': dyn,
                    'ts': model['t_f']/N,
                    'x_val': x_val_np,
                    'u_val': u_val_np,
                    # 'xdot_val':xdot_val_np
                },outfile)    
    avg_power_df = pd.DataFrame(avg_power_array, columns=["Beta","acc_reg","Intervals", "time", "avg_power","twall","final_time","b_optimal"])
    # Write to a .txt file as comma-separated values
    avg_power_df.to_csv(f"looped_wo_warmstarting_avg_power_and_twall_for_beta_and_Acc_N_{N}_T_{time_per_winding}.txt", index=False, header=True, sep=',',mode='a')

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
    for idx, param_val in enumerate([25]):

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
            avg_power_array.append([beta_0, acc_reg, N, param_val, avg_power.full().item(),t_wall,final_time.full().item(), b_optimal])
                
        
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
            avg_power_array.append([beta_0, acc_reg, N, param_val, avg_power.full().item(),t_wall,final_time.full().item(),b_optimal])

        trial_default.write_to_csv(filename=filefolder + '{}_sol_beta{}_acc_reg_{}_N_{}_time_{}'.format(file_name, beta_0, acc_reg,N, param_val), frequency=10., rotation_representation='dcm')
        print("Wrote file to the folder")
    acg_power_df = pd.DataFrame(avg_power_array, columns=["beta","acc_reg","N", "T", "avg_power","twall","final_time","b_optimal"])
    # Write to a .txt file as comma-separated values
    acg_power_df.to_csv(CSV_PATH, index=False, header=True, sep=',')
    print("Wrote CSV File")
    