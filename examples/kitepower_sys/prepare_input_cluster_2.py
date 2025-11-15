#
#    This file is part of TuneMPC.
#
#    TuneMPC -- A Tool for Economic Tuning of Tracking (N)MPC Problems.
#    Copyright (C) 2020 Jochem De Schutter, Mario Zanon, Moritz Diehl (ALU Freiburg).
#
#    TuneMPC is free software; you can redistribute it and/or
#    modify it under the terms of the GNU Lesser General Public
#    License as published by the Free Software Foundation; either
#    version 3 of the License, or (at your option) any later version.
#
#    TuneMPC is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#    Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with TuneMPC; if not, write to the Free Software Foundation,
#    Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
#
#
#!/usr/bin/python3
""" awebox optimization toolbox is available at: https://github.com/awebox/awebox
commit hash: 561f1726f28357e04b2e6a1163a1b30753670784

:author: Jochem De Schutter

"""

import awebox as awe
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import awebox.opts.kite_data.kitepower_lei_data as kitepower_lei_data
import casadi as ca
import casadi.tools as ct
import pickle
import awebox.tools.integrator_routines as awe_integrators
import time
import pandas as pd
from pathlib import Path



####################### TESTS #######################################
ROOT = Path('/pfs/data6/home/fr/fr_fr/fr_ns591/code/examples/files')
TOL = 1e-10
N_reps = 1
windings=1
##################################
intervals_per_winding = 64
time_per_winding = 32
beta_0 = 0.5
acc_reg_weight = 1e6
#################################
N = windings*(intervals_per_winding)
N_sim=N
# beta_0 = 5e-2

scaling = False
d=4

def run_simulation(f_fun, l_fun, x0, controls, N,diff_integrator=False):
    # x_sim = [x0.full().squeeze()]
    x_sim = []
    x_sim.append(x0)
    l_sim = [0.0]
    timings=[]

    for k in range(N-1):
        print(f"sim_test {k=}")

        x_k = x_sim[-1]
        u_k = controls[k]

        start_time = time.time()
        if diff_integrator:
            tet_len = x_k[-3]
            res = f_fun(x_k, u_k)
            xf=res[0]
            x_next = xf.full().squeeze()
            # x_next = np.append(x_next,[tet_len,0,0])
            l_next = res[1].full().squeeze()
        else:
            x_next = f_fun(x_k, u_k).full().squeeze()
            l_next = l_fun(x_k, u_k).full().squeeze()
        print(f"x_next_shape {np.shape(x_next)}")
        elapsed_time = time.time() - start_time
        
        if np.isnan(x_next).any():
            break
        x_sim.append(x_next)
        l_sim.append(l_sim[-1] + l_next)
        timings.append(elapsed_time)
    
    return x_sim, l_sim, timings

def run_jacobian_test(integrator, x0_list, controls):
    x, u = integrator.mx_in()
    integrator_jac = ca.jacobian(integrator(x, u), ca.vertcat(x, u))
    jac_fun = ca.Function('integrator_jac', [x, u], [integrator_jac], {"record_time": True})
    jac_list = []
    N = len(x0_list)
    for rep in range(N_reps):
        timings = []
        for k in range(N-1):
            print(f"jac_test {k=}")
            jac_list.append(jac_fun(x0_list[k], controls[k]).full())
            # timings.append(get_time_casadi_fun(jac_fun))
        if rep == 0:
            timings_min = timings
        else:
            timings_min = [min(timings[i], timings_min[i]) for i in range(len(timings))]
    print(f"{timings_min=}, mean: {np.mean(timings_min)}")
    return jac_list, timings_min

def timing_comparison(timing_list, title=''):
    print(f"Timing comparison {title}")
    # print(LABELS, "speedup")
    for label, metric in [('mean', np.mean), ('median', np.median), ('max', np.max), ('min', np.min)]:
        timing_values = [1e3*metric(t) for t in timing_list]
        timing_strings = [f'{t:.4f}' for t in timing_values]
        speedup = timing_values[1] / timing_values[0]
        print(f"{label} & {' & '.join(timing_strings)}, {speedup:.2f}")

###################################################
# indicate desired system architecture

def generate_kite_model_and_orbit(windings, intervals_per_winding, time_per_winding):
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
    options['nlp.n_k'] = int(intervals_per_winding * windings)
    options['nlp.collocation.d'] = d
    options['nlp.collocation.u_param'] = 'zoh'
    options['user_options.trajectory.lift_mode.phase_fix'] = 'simple' # 'simple' # 'single_reelout'
    options['solver.linear_solver'] = 'mumps'  # if HSL is installed, otherwise 'mumps'
    options['model.system_bounds.x.ddl_t'] = [-2.0, 2.0]
    options['model.system_bounds.theta.t_f'] = [0, (windings*time_per_winding+1)] ## +2*(time_per_winding/intervals_per_winding)
    options['nlp.phase_fix_reelout'] = 0.7
    options['solver.cost.beta.0'] = beta_0
    options['solver.weights.ddq'] =  acc_reg_weight

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
    options['solver.max_iter_hippo'] = 1000
    options['solver.max_iter'] = 1000
    options['visualization.cosmetics.plot_ref'] = False

    # build and optimize the NLP (trial)
    trial = awe.Trial(options, 'Kitepower_LEI')
    trial.build()
    trial.optimize(final_homotopy_step = 'final')
    plot_names = ['states', 'controls', 'isometric', 'constraints']
    trial.plot(plot_names)
   # save all open figures created by trial.plot
    for i, num in enumerate(plt.get_fignums()):
        fig = plt.figure(num)
        # use either index-based names:
        fig.savefig(ROOT/ "plots_from_awebox" / f"plot_T_{time_per_winding}_beta_{beta_0}_acc_reg_{acc_reg_weight}_{i}.png", dpi=300, bbox_inches='tight')
        # or, if you want to tie to names:
        # fig.savefig(f"trial_plot_{plot_names[i]}.png", dpi=300, bbox_inches='tight')

    plt.close('all')
    # extract model data
    sol = {}
    sol['model'] = trial.generate_optimal_model()
    print(f"Available keys in V_opt: {trial.optimization.V_opt.keys()}")
    sol['l_t']   = trial.optimization.V_opt['x',0,'l_t']
    sol['t_f'] = trial.optimization.V_opt['theta','t_f']
    sol["avg_power_output"]=trial.visualization.plot_dict['power_and_performance']['avg_power']
    # sol["scaling"]=trial.model.scaling()
    # initial guess
    w_init = []
    x_val= []
    u_val =[]
    x_dot_val=[]
    shape_of_x = trial.optimization.V_opt['x',0].shape
    shape_of_u = trial.optimization.V_opt['u', 0][3:6].shape
    print(f"x shape : {shape_of_x}")
    print(f"u shape : {shape_of_u}")
    for k in range(N):
        x_val.append(trial.optimization.V_opt['x',k])
        u_val.append(trial.optimization.V_opt['u', k][3:6])
        x_dot_val.append(trial.optimization.V_opt['xdot',k])
        w_init.append(trial.optimization.V_opt['x',k])
        w_init.append(trial.optimization.V_opt['u', k][3:6])
    sol['w0'] = ca.vertcat(*w_init)

    return sol, x_val, u_val, x_dot_val, trial


def test_model_functions(f, l, x_opt, u_opt):

    x_test = [x_opt[0]]
    l_test = [0.0]
    for k in range(len(x_opt)):
        x_test.append(f(x_opt[k], u_opt[k])[0].full().squeeze())
        l_test.append(l(x_opt[k], u_opt[k]).full().squeeze())

# N = 100

awe_sol,awe_x_val,awe_u_val,awe_xdot_val, trial = generate_kite_model_and_orbit(windings, intervals_per_winding, time_per_winding)
w_0=awe_sol['w0']

# remove tether variables
model = awe_sol['model']
scaling_factors = model['scaling']
l_t = awe_sol['l_t']
x_shape = model['dae']['x'].shape
x_shape = (x_shape[0], x_shape[1])
x = ca.MX.sym('x',*x_shape)
x_awe = x
# x_awe = ca.vertcat(x,l_t,0,0)
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
# xdot_awe = ca.vertcat(xdot,0,0,0)
# z = ca.MX.sym('z', model['dae']['z']['z'].shape[0])
x_scale = []
u_scale = []
z_scale = []

x_scale=np.array(scaling_factors['x']).flatten()
u_scale=np.array(scaling_factors['u']).flatten()
z_scale=np.array(scaling_factors['z']).flatten()

if (scaling):  
    x_scaled = x_awe
    u_scaled = u_awe

    x_physical = x_scaled * ca.vertcat(*x_scale)
    # u_physical = ca.vertcat(ca.DM.zeros(3,1), u_scaled * ca.vertcat(*u_scale))  # zero padd for fictitious
    u_physical = u_scaled * ca.vertcat(*u_scale)
    # z_physical = z* ca.vertcat(*z_scale)

w0 = awe_sol['w0']
# if scaling:
x_scale_dm = ca.DM(x_scale)
u_scale_clean_dm = ca.DM(u_scale[3:6])  # only physical controls
xu_scale_single = ca.vertcat(x_scale_dm, u_scale_clean_dm)
xu_scale_repeated = ca.repmat(xu_scale_single, N, 1)  # creates (N × len(xu_scale_single), 1) DM
w0_physical=w0*xu_scale_repeated

# remove algebraic variable
if scaling:
    z_0 = model['rootfinder'](0.1, x_0, u_0)
    z_0 = z_0*ca.vertcat(*z_scale)          #(12,1)
    z = model['rootfinder'](z_0, x_physical, u_physical) #(12,1)
else:
    z_0 = model['rootfinder'](0.1, x_0, u_0) #(12,1)
    z = model['rootfinder'](z_0, x_awe, u_awe) #(12,1)
    z_func = ca.Function('z_fun', [x, u], [z], ['x','u'], ['z'])

# create integrator
# integrator = ca.integrator('F', 'collocation', model['dae'], {'collocation_scheme': 'radau', 'interpolation_order': 5, 'tf': 1/N, 'number_of_finite_elements': 10})
integrator = awe_integrators.rk4root(
        'F',
        model['dae'],
        model['rootfinder'],
        {'tf': 1/N, 'number_of_finite_elements':20})
if scaling:
    xf = integrator(x0=x_physical, p=u_physical, z0 = 0.1*z_scale)['xf']
    qf = integrator(x0=x_physical, p=u_physical, z0 = 0.1*z_scale)['qf']
    constraints = ca.vertcat(
        -model['constraints'](x_physical, u_physical,z),
        -model['var_bounds_fun'](x_physical, u_physical,z)
    )
else:
    xf = integrator(x0=x_awe, p=u_awe, z0 = z_0)['xf']
    qf = integrator(x0=x_awe, p=u_awe, z0 = z_0)['qf']
    constraints = ca.vertcat(
        -model['constraints'](x_awe, u_awe,z),
        -model['var_bounds_fun'](x_awe, u_awe,z)
    )
    


# remove redundant constraints
constraints_new = []
for i in range(constraints.shape[0]):
    if True in ca.which_depends(constraints[i],ca.vertcat(x,u)):
        constraints_new.append(constraints[i])

h = ca.Function('h', [x,u], [ca.vertcat(*constraints_new)])
    

sys = {
    'f' : ca.Function('F',[x,u],[xf,qf],['x0','p'],['xf','qf']),
    'h' : ca.Function('h', [x,u], [ca.vertcat(*constraints_new)])
}
sys['z'] = z_func

# cost function
# qf = sys['f'](x0=x, p=u)['qf']
# power_output = trial.optimization.p_fix_num['cost', 'power'] * qf[0] / model['t_f']
# regularization = 0 # 1/2*1e-4*ct.mtimes(u.T,u)
# power_output = -sys['f'](x0=x, p=u)['qf'][0]/model['t_f']/1e3
# yaw_rate_reg = 0 #
cost = ca.Function(
    'cost',
    [x,u],
    [qf] #+ extra_regularization
)


test_model_functions(sys['f'], cost, awe_x_val,awe_u_val)

##################################### UPDATE ##############################
 # remove algebraic variable
algf  = ca.Function('algf', [model['dae']['x'], model['dae']['p'], model['dae']['z']], [model['dae']['alg']])
A = ca.jacobian(model['dae']['alg'], model['dae']['z'])
b = - algf(model['dae']['x'], model['dae']['p'], model['dae']['z'](0.0))
rootfinder = ca.Function('rootfinder', [model['dae']['z'], model['dae']['x'], model['dae']['p']], [ca.solve(A,b)])
# z_0 = rootfinder(0.1, x_0, u_0)
# z = rootfinder(z_0, xh_awe, u_awe)    

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

pickle_filename = ROOT/"input_files"/f"kitepower_user_input_{windings*intervals_per_winding}_w_{windings}_tpw_{time_per_winding}_beta0_{beta_0}_acc_reg_{acc_reg_weight}_with_z.pkl"

x_val_np = [ca.DM(x).full().flatten().tolist() for x in awe_x_val]
u_val_np = [ca.DM(u).full().flatten().tolist() for u in awe_u_val]

if scaling :
    x_val_np_physical = x_val_np*x_scale

    u_val_np_physical = u_val_np*u_scale[3:6]
    

print(f"final_time: {model['t_f']}")
print(f"power_output: {awe_sol['avg_power_output']}")

with open(pickle_filename,'wb') as outfile:
        pickle.dump({
            'f': sys['f'],
            'l': cost,
            'h': h,
            'p': N,
            'w0': w0,
            'z': sys['z'],
            'w0_physical': w0_physical,
            'dyn': dyn,
            'ts': model['t_f']/N,
            'x_val': x_val_np,
            'u_val': u_val_np,
            'x_scale': x_scale,
            'u_scale': u_scale,
            'scaling_factor' : scaling_factors
            # 'xdot_val':xdot_val_np
        },outfile)    

######################## INTEGRATORS  #####################################
controls = u_val_np
x0 = x_val_np[0] 

collocation_opts = {
            'tf': 1/N,
            'number_of_finite_elements': 1,
            'collocation_scheme':'radau',
            # 'rootfinder': 'fast_newton',
            'interpolation_order': 4,
            'rootfinder_options':
                {'line_search': False, 'abstolStep': TOL, 'max_iter': 20, 'print_iteration': False} #, 'abstol': TOL
        }


# integrator_casados, f_casados, l_casados = acados_simulator.create_awe_casados_integrator(dyn, model['t_f']/N,collocation_opts=collocation_opts, use_cython=False)  
# x_sim_casados, l_sim_casados, timings_casados = run_simulation(f_casados, l_casados, x0, controls, N)
x_sim_casadi, l_sim_casadi, timings_casadi = run_simulation(sys['f'], cost, x0, controls, N,diff_integrator=True)
x_ref_array = np.array(x_val_np)
x_sim_casadi= np.array(x_sim_casadi)
# x_sim_casados=np.array(x_sim_casados)


len_ref = x_ref_array.shape[0]
len_sim = x_sim_casadi.shape[0]
max_len = max(len_ref, len_sim)
# x_sim_padded_casados = np.pad(x_sim_casados, ((0, max_len - len_sim), (0, 0)), mode='constant')
x_sim_casadi = np.pad(x_sim_casadi, ((0, max_len - len_sim), (0, 0)), mode='constant')


# data = {
#     'x_ref': x_ref_array[:, 0],
#     'y_ref': x_ref_array[:, 1],
#     'z_ref': x_ref_array[:, 2],
#     'x_sim_casadi': x_sim_casadi[:, 0],
#     'y_sim_casadi': x_sim_casadi[:, 1],
#     'z_sim_casadi': x_sim_casadi[:, 2]
#     # 'x_sim_casados': x_sim_padded_casados[:, 0],
#     # 'y_sim_casados': x_sim_padded_casados[:, 1],
#     # 'z_sim_casados': x_sim_padded_casados[:, 2]
# }      

# df_padded = pd.DataFrame(data)

# # Save to txt file
# padded_file_path = "trajectories_comparision.txt"
# df_padded.to_csv(padded_file_path, index=False, sep='\t')


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


ax.plot(x_ref_array[:,0], x_ref_array[:,1], x_ref_array[:,2], label='Reference Trajectory', linestyle='--')
# ax.plot(x_sim_casados[:,0], x_sim_casados[:,1], x_sim_casados[:,2], label='Casados Trajectory')
ax.plot(x_sim_casadi[:,0], x_sim_casadi[:,1], x_sim_casadi[:,2], label='RK4 Trajectory')

ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.legend()
ax.set_title('Kite Trajectories')
# plt.show()
plt.savefig(ROOT / "plots_from_awebox"/ f"trajectory_integrator_T_{time_per_winding}_beta_{beta_0}_acc_reg_{acc_reg_weight}.png", dpi=300, bbox_inches='tight')
plt.close(fig)