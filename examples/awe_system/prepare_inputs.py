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
import awebox.tools.integrator_routines as awe_integrators
import matplotlib.pyplot as plt
import numpy as np
import casadi as ca
import casadi.tools as ct
import pickle
import casados_integrator as casados
import acados_simulator
# from casados_integrator import acados_simulators
# import acados as acd
# from acd import acados_simulators

def generate_kite_model_and_orbit(N):

    import point_mass_model

    options ={}
    # make default options object
    # options = awe.Options()

    # single kite with point-mass model
    options['user_options.system_model.architecture'] = {1: 0}
    options['user_options.system_model.kite_dof'] = 3
    options['user_options.kite_standard'] = point_mass_model.data_dict()

    # trajectory should be a single pumping cycle with initial number of five windings
    options['user_options.trajectory.type'] = 'power_cycle'
    options['user_options.trajectory.system_type'] = 'drag_mode'
    options['user_options.trajectory.lift_mode.windings'] = 1

    # don't include induction effects, use simple tether drag
    options['user_options.induction_model'] = 'not_in_use'
    options['user_options.tether_drag_model'] = 'kite_only'
    options['model.system_bounds.theta.t_f'] = [5., 30.]
    # options['user_options.'] = 'trivial'
    options['nlp.n_k'] = N

    # get point mass model data
    options = point_mass_model.set_options(options)

    # initialize and optimize trial
    trial = awe.Trial(options, 'single_kite_drag_mode')
    trial.build()
    trial.optimize(final_homotopy_step='final')
    # trial.plot(['states','controls', 'constraints'])
    # plt.show()
    # extract model data
    sol = {}
    sol['model'] = trial.generate_optimal_model()
    print(f"Available keys in V_opt: {trial.optimization.V_opt.keys()}")
    # for main_keys in trial.optimization.V_opt.keys() :
    #      print("DEBUG: Type of main_keys:", type(main_keys))
    #      print("DEBUG: Value of main_keys:", main_keys)
    #      print("DEBUG: Available keys in {main_keys}:", list(trial.optimization.V_opt[main_keys, 0].keys()))

    #      if isinstance(main_keys, dict): 
    #         if  main_keys.keys() is not(None) :
    #             for sub_keys in main_keys.keys():
    #                 print(f"Available sub-keys in {main_keys}: {trial.optimization.V_opt[main_keys].keys()}")

    # print(f"Available sub-keys in x_dot: {trial.optimization.V_opt['theta', 'l_t']}")
    sol['l_t']   = trial.optimization.V_opt['theta', 'l_t']

    # initial guess
    w_init = []
    for k in range(N):
        w_init.append(trial.optimization.V_opt['x',k])
        w_init.append(trial.optimization.V_opt['u', k][3:6])
    sol['w0'] = ca.vertcat(*w_init)

    return sol


# discrete period of interest
# N = 40
N = 100
awe_sol = generate_kite_model_and_orbit(N)

# remove tether variables
model = awe_sol['model']
l_t = awe_sol['l_t']
x_shape = model['dae']['x'].shape
x_shape = (x_shape[0], x_shape[1])
x = ca.MX.sym('x',*x_shape)
x_awe = x

# remove fictitious forces, tether jerk...
u_shape = model['dae']['p'].shape
u_shape = (u_shape[0]-3, u_shape[1])
u = ca.MX.sym('u',*u_shape)
u_awe = ct.vertcat(0.0,0.0,0.0,u)

# remove algebraic variable
z = model['rootfinder'](0.1, x_awe, u_awe)
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

# create integrator
# integrator = awe_integrators.rk4root(
#         'F',
#         model['dae'],
#         model['rootfinder'],
#         {'tf': 1/N, 'number_of_finite_elements':10})

# _, f, l = acados_simulator.create_awe_casados_integrator(dyn, model['t_f']/N)
# xf = integrator(x0=x_awe, p=u_awe, z0 = 0.1)['xf']
# qf = integrator(x0=x_awe, p=u_awe, z0 = 0.1)['qf']

# sys = {
#     'f' : ca.Function('F',[x,u],[xf,qf],['x0','p'],['xf','qf']),
#     'h' : ca.Function('h', [x,u], [ca.vertcat(*constraints_new)])
# }

# # cost function
# power_output = -sys['f'](x0=x, p=u)['qf'][0]/model['t_f']/1e3
# regularization = 1/2*1e-4*ct.mtimes(u.T,u)

# cost = ca.Function(
#     'cost',
#     [x,u],
#     [power_output + regularization] #+ extra_regularization
# )

# initial guess
w0 = awe_sol['w0']

# save time-continuous dynamics
xdot = ca.MX.sym('xdot', x.shape[0])
xdot_awe = xdot
z = ca.MX.sym('z', model['dae']['z']['z'].shape[0])
# indeces = [*range(2,10)]+[*range(11,nx+3+z.shape[0])] # remove ldot, lddot, ldddot
alg = model['dae']['alg'] 
alg_fun = ca.Function('alg_fun',[model['dae']['x'],model['dae']['p'],model['dae']['z']],[alg])
dyn = ca.Function(
    'dae',
    [xdot,x,u,z],
    [alg_fun(x_awe, u_awe, ct.vertcat(xdot_awe, z))],
    ['xdot','x','u','z'],
    ['dyn'])

# integrator, f, l = acados_simulator.create_awe_casados_integrator(dyn, model['t_f']/N,use_cython=False)  
h = ca.Function('h', [x,u], [ca.vertcat(*constraints_new)])

# save user input info
with open('user_input.pkl','wb') as outfile:
        pickle.dump({
            # 'f': f,
            # 'l': l,
            'h': h,
            'p': N,
            'w0': w0,
            'dyn': dyn,
            'ts': model['t_f']/N
        },outfile)

# xf = integrator(x0=x_awe, p=u_awe, z0 = 0.1)['xf']      # final state
# qf = integrator(x0=x_awe, p=u_awe, z0 = 0.1)['qf']      # final power

# sys = {
#     'f' : ca.Function('F',[x,u],[xf,qf],['x0','p'],['xf','qf']),
#     'h' : ca.Function('h', [x,u], [ca.vertcat(*constraints_new)])
# }

# # cost function
# power_output = -sys['f'](x0=x, p=u)['qf'][0]/model['t_f']/1e3
# regularization = 1/2*1e-4*ct.mtimes(u.T,u)

# cost = ca.Function(
#     'cost',
#     [x,u],
#     [power_output + regularization] #+ extra_regularization
# )  

# # save user input info
# with open('user_input.pkl','wb') as f:
#         pickle.dump({
#             'f': sys['f'],
#             'l': cost,
#             'h': sys['h'],
#             'p': N,
#             'w0': w0,
#             'dyn': dyn,
#             'ts': model['t_f']/N
#         },f)
