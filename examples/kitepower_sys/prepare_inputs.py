import awebox as awe
import awebox.opts.kite_data.ampyx_ap2_settings as ampyx_ap2_settings
import matplotlib.pyplot as plt
import numpy as np
import awebox.tools.print_operations as print_op
import awebox.opts.kite_data.kitepower_lei_data as kitepower_lei_data
import casadi as ca
import casadi.tools as ct
import pickle


# indicate desired system architecture

def generate_kite_model_and_orbit(N):
    options = {}
    options['user_options.system_model.architecture'] = {1: 0}
    options['user_options.kite_standard'] = kitepower_lei_data.data_dict()
    options['user_options.system_model.wing_type'] = 'LEI'
    options['user_options.system_model.kite_dof'] = 3

    # indicate desired operation mode
    options['user_options.trajectory.type'] = 'power_cycle'
    options['user_options.trajectory.system_type'] = 'lift_mode'
    windings = 1
    options['user_options.trajectory.lift_mode.windings'] = windings

    # indicate desired environment
    options['params.wind.z_ref'] = 100.0
    options['params.wind.power_wind.exp_ref'] = 0.15
    options['user_options.wind.model'] = 'power'
    options['user_options.wind.u_ref'] = 10.

    # coefficient boundaries
    options['model.system_bounds.x.coeff'] =  [np.array([-1., 0.]), np.array([1., 1.])]

    # indicate numerical nlp details
    # here: nlp discretization, with a zero-order-hold control parametrization, and
    # a simple phase-fixing routine. also, specify a linear solver to perform the Newton-steps
    # within ipopt.
    options['nlp.n_k'] = N
    options['nlp.collocation.u_param'] = 'zoh'
    options['user_options.trajectory.lift_mode.phase_fix'] = 'simple' # 'single_reelout'
    options['solver.linear_solver'] = 'mumps'  # if HSL is installed, otherwise 'mumps'
    options['model.system_bounds.x.ddl_t'] = [-2.0, 2.0]
    # options['model.system_bounds.theta.t_f'] = [0.0, windings*30.0]

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
    options['solver.initialization.inclination_deg'] = 45.
    options['solver.initialization.groundspeed'] = 20.
    options['solver.initialization.theta.diam_t'] = 5e-3
    options['solver.initialization.l_t'] = 300.0
    options['solver.max_iter_hippo'] = 1000
    options['solver.max_iter'] = 1000
    # options['visualization.cosmetics.plot_ref'] = False

    # build and optimize the NLP (trial)
    trial = awe.Trial(options, 'Kitepower_LEI')
    trial.build()
    trial.optimize(final_homotopy_step = 'final')

    # extract model data
    sol = {}
    sol['model'] = trial.generate_optimal_model()
    print(f"Available keys in V_opt: {trial.optimization.V_opt.keys()}")
    # sol['l_t']   = trial.optimization.V_opt['theta', 'l_t']

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
N = 40
awe_sol,awe_x_val,awe_u_val,awe_xdot_val = generate_kite_model_and_orbit(N)

x_val_np = [ca.DM(x).full().flatten() for x in awe_x_val]
u_val_np = [ca.DM(u).full().flatten() for u in awe_u_val]
xdot_val_np = [ca.DM(xdot).full().flatten() for u in awe_xdot_val]

# remove tether variables
model = awe_sol['model']
# l_t = awe_sol['l_t']
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
with open('kitepower_user_input_40.pkl','wb') as outfile:
        pickle.dump({
            # 'f': f,
            # 'l': l,
            'h': h,
            'p': N,
            'w0': w0,
            'dyn': dyn,
            'ts': model['t_f']/N,
            'x_val': x_val_np,
            'u_val': u_val_np,
            'xdot_val':xdot_val_np
        },outfile)
     