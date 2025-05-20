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
import awebox.opts.kite_data.ampyx_ap2_settings as ampyx_ap2_settings
import matplotlib.pyplot as plt
import numpy as np
import awebox.tools.print_operations as print_op
import awebox.opts.kite_data.kitepower_lei_data as kitepower_lei_data
import casadi as ca
import casadi.tools as ct
import pickle
import tunempc
import awebox.tools.integrator_routines as awe_integrators
import casados_integrator as casados
import acados_simulator
import time
import pandas as pd


####################### TESTS #######################################
TOL = 1e-10
N_reps = 1
windings=1
intervals_per_winding = 20
time_per_winding = 26
N=windings*intervals_per_winding
N_sim=N

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
    options['user_options.wind.u_ref'] = 10.

    # coefficient boundaries
    options['model.system_bounds.x.coeff'] =  [np.array([-1., 0.]), np.array([1., 1.])]
    options['model.system_bounds.u.dcoeff'] =  [np.array([-.04, -1]), np.array([.04, 1])]

    # indicate numerical nlp details
    # here: nlp discretization, with a zero-order-hold control parametrization, and
    # a simple phase-fixing routine. also, specify a linear solver to perform the Newton-steps
    # within ipopt.
    options['nlp.n_k'] = int(intervals_per_winding * windings)
    options['nlp.collocation.u_param'] = 'zoh'
    options['user_options.trajectory.lift_mode.phase_fix'] = 'single_reelout' # 'simple' # 'single_reelout'
    options['solver.linear_solver'] = 'mumps'  # if HSL is installed, otherwise 'mumps'
    options['model.system_bounds.x.ddl_t'] = [-2.0, 2.0]
    options['model.system_bounds.theta.t_f'] = [0.0, windings*time_per_winding]
    options['nlp.phase_fix_reelout'] = 0.7 

    options['model.model_bounds.acceleration.include']  = False
    options['model.model_bounds.aero_validity.include']  = False
    options['model.model_bounds.tether_stress.include']  = False
    # (experimental) set to "True" to significantly (factor 5 to 10) decrease construction time
    # note: this may result in slightly slower solution timings
    options['nlp.compile_subfunctions'] = False


    # initialization
    options['solver.initialization.shape'] = 'lemniscate'
    options['solver.initialization.lemniscate.az_width'] = 20*np.pi/180.
    options['solver.initialization.lemniscate.el_width'] = 8*np.pi/180.
    options['solver.initialization.inclination_deg'] = 30.
    options['solver.initialization.groundspeed'] = 20.
    options['solver.initialization.theta.diam_t'] = 5e-3
    options['solver.initialization.l_t'] = 300.0
    options['solver.max_iter_hippo'] = 1000
    options['solver.max_iter'] = 1000
    options['visualization.cosmetics.plot_ref'] = False

    # build and optimize the NLP (trial)
    trial = awe.Trial(options, 'Kitepower_LEI')
    trial.build()
    trial.optimize(final_homotopy_step = 'final')
    trial.plot(['states', 'controls', 'isometric'])
    plt.show()
    # extract model data
    sol = {}
    sol['model'] = trial.generate_optimal_model()
    print(f"Available keys in V_opt: {trial.optimization.V_opt.keys()}")
    sol['l_t']   = trial.optimization.V_opt['x',0,'l_t']

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

    return sol, x_val, u_val, x_dot_val


# N = 100

awe_sol,awe_x_val,awe_u_val,awe_xdot_val = generate_kite_model_and_orbit(windings, intervals_per_winding, time_per_winding)
w_0=awe_sol['w0']

x_val_np = [ca.DM(x).full().flatten().tolist() for x in awe_x_val]
u_val_np = [ca.DM(u).full().flatten().tolist() for u in awe_u_val]
# xdot_val_np = [ca.DM(xdot).full().flatten() for u in awe_xdot_val]
u_val_np_padded = u_val_padded_list = [np.concatenate([np.zeros(3), np.array(u)]).tolist() for u in u_val_np]
# remove tether variables
model = awe_sol['model']
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

# remove algebraic variable
z_0 = model['rootfinder'](0.1, x_0, u_0) #(12,1)
# z_0 = rootfinder(0.1, x_0, u_0)
z = model['rootfinder'](z_0, x_awe, u_awe) #(12,1)
nx = x.shape[0]
nu = u_shape[0]
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

# initial guess
w0 = awe_sol['w0']

# save time-continuous dynamics
xdot = ca.MX.sym('xdot', x.shape[0])
xdot_awe = xdot
# xdot_awe = ca.vertcat(xdot,0,0,0)
# z = ca.MX.sym('z', model['dae']['z']['z'].shape[0])

# create integrator
integrator = awe_integrators.rk4root(
        'F',
        model['dae'],
        model['rootfinder'],
        {'tf': 1/N, 'number_of_finite_elements':10})
xf = integrator(x0=x_awe, p=u_awe, z0 = 0.1)['xf']
qf = integrator(x0=x_awe, p=u_awe, z0 = 0.1)['qf']

sys = {
    'f' : ca.Function('F',[x,u],[xf,qf],['x0','p'],['xf','qf']),
    'h' : ca.Function('h', [x,u], [ca.vertcat(*constraints_new)])
}

# cost function
power_output = -sys['f'](x0=x, p=u)['qf']/model['t_f']/1e3
regularization = 1/2*1e-4*ct.mtimes(u.T,u)

cost = ca.Function(
    'cost',
    [x,u],
    [power_output + regularization] #+ extra_regularization
)



##################################### UPDATE ##############################
 # remove algebraic variable
algf  = ca.Function('algf', [model['dae']['x'], model['dae']['p'], model['dae']['z']], [model['dae']['alg']])
A = ca.jacobian(model['dae']['alg'], model['dae']['z'])
b = - algf(model['dae']['x'], model['dae']['p'], model['dae']['z'](0.0))
rootfinder = ca.Function('rootfinder', [model['dae']['z'], model['dae']['x'], model['dae']['p']], [ca.solve(A,b)])
z_0 = rootfinder(0.1, x_0, u_0)
z = rootfinder(z_0, xh_awe, u_awe)    

rm_indeces = []
z = ca.MX.sym('z', model['dae']['z']['z'].shape[0])
indeces = [k for k in range(x_awe.shape[0]+z.shape[0]) if k not in rm_indeces]
alg = model['dae']['alg'][indeces]
alg_energy = ca.vertcat(alg[:-1], model['dae']['z']['xdot'][-1] - model['dae']['quad']/model['t_f'])
alg_fun = ca.Function('alg_fun',[model['dae']['x'],model['dae']['p'],model['dae']['z']],[alg_energy])
#########################################################

# alg = model['dae']['alg'] 
# alg_fun = ca.Function('alg_fun',[model['dae']['x'],model['dae']['p'],model['dae']['z']],[alg])

dyn = ca.Function(
    'dae',
    [xdot,x,u,z],
    [alg_fun(x, u_awe, ct.vertcat(xdot, z))],
    ['xdot','x','u','z'],
    ['dyn'])

pickle_filename = f"kitepower_user_input_{windings*intervals_per_winding}_w_{windings}_tpw_{time_per_winding}_final_test_new.pkl"

with open(pickle_filename,'wb') as outfile:
        pickle.dump({
            'f': sys['f'],
            'l': cost,
            'h': h,
            'p': N,
            'w0': w0,
            'dyn': dyn,
            'ts': model['t_f']/N,
            'x_val': x_val_np,
            'u_val': u_val_np
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

            # 'jit': True #   #error Code generation not supported for Collocation
        }


integrator_casados, f_casados, l_casados = acados_simulator.create_awe_casados_integrator(dyn, model['t_f']/N,collocation_opts=collocation_opts, use_cython=False)  
x_sim_cas, l_sim_cas, timings_cas = run_simulation(f_casados, l_casados, x0, controls, N)
x_sim_casados, l_sim_casados, timings_casados = run_simulation(sys['f'], cost, x0, controls, N,diff_integrator=True)
x_ref_array = np.array(x_val_np)
x_sim_casados= np.array(x_sim_casados)
x_sim_cas=np.array(x_sim_cas)


len_ref = x_ref_array.shape[0]
len_sim = x_sim_cas.shape[0]
max_len = max(len_ref, len_sim)
x_sim_padded = np.pad(x_sim_cas, ((0, max_len - len_sim), (0, 0)), mode='constant')


data = {
    'x_ref': x_ref_array[:, 0],
    'y_ref': x_ref_array[:, 1],
    'z_ref': x_ref_array[:, 2],
    'x_sim_casadi': x_sim_padded[:, 0],
    'y_sim_casadi': x_sim_padded[:, 1],
    'z_sim_casadi': x_sim_padded[:, 2],
    'x_sim_casados': x_sim_cas[:, 0],
    'y_sim_casados': x_sim_cas[:, 1],
    'z_sim_casados': x_sim_cas[:, 2]
}      

df_padded = pd.DataFrame(data)

# Save to txt file
padded_file_path = "trajectories_comparision.txt"
df_padded.to_csv(padded_file_path, index=False, sep='\t')


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# plt.plot([xx[0] for xx in model_x], label='AWEBox')

ax.plot(x_ref_array[:,0], x_ref_array[:,1], x_ref_array[:,2], label='Reference Trajectory', linestyle='--')
# ax.plot(x_integrator_array[0, :], x_integrator_array[1, :], x_integrator_array[2, :], label='Casadi Trajectory')
ax.plot(x_sim_cas[:,0], x_sim_cas[:,1], x_sim_cas[:,2], label='Casados Trajectory')
ax.plot(x_sim_casados[:,0], x_sim_casados[:,1], x_sim_casados[:,2], label='Casadi IRK Trajectory')
# ax.plot(x_sim_rk[:,0], x_sim_rk[:,1], x_sim_rk[:,2], label='Casados Trajectory')

ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.legend()
ax.set_title('Kite Trajectories')
plt.show()