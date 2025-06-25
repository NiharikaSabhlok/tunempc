import tunempc
import pickle
import ipdb
import casadi as ca
import awebox as awe
import awebox.tools.integrator_routines as awe_integrators
import casados_integrator as casados
import acados_simulator
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from awebox.logger.logger import Logger as awelogger
from tunempc.logger import Logger
Logger.logger.setLevel('DEBUG')
awelogger.logger.setLevel('DEBUG')

user_input_file = 'kitepower_user_input_54_w_1_tpw_27_beta0_0.1_acc_reg_1.0.pkl'
# 'kitepower_user_input_120_w_2_tpw_28_final.pkl'
convex_ref_file = 'convex_referencefile_54_w_1_tpw_27.pkl'

# load user input
with open(user_input_file,'rb') as outfile:
    user_input = pickle.load(outfile)

# Rebuild integrator + CasADi functions
dyn = user_input['dyn']
ts = user_input['ts']
N = user_input['p']
# model_xdot = user_input['xdot_val']
model_x = user_input['x_val']
model_u = user_input['u_val']
model_x = ca.horzcat(*model_x)  # (11, N)
model_u = ca.horzcat(*model_u)  # (3, N)
# model_xdot = ca.horzcat(*model_xdot)

print(f"shape of model_x: {model_x.shape}")
print(f"shape of model_u: {model_u.shape}")
# integrator, f, l = acados_simulator.create_awe_casados_integrator(dyn, ts, use_cython=False)

###################################################

N_reps = 1
def get_time_casadi_fun(fun):
    return fun.stats()['t_wall_total']

def run_simulation(f_fun, l_fun, x0, controls, N,diff_integrator=False):
    x_sim = [x0.full().squeeze()]
    l_sim = [0.0]
    timings=[]

    for k in range(N-1):
        print(f"sim_test {k=}")

        x_k = x_sim[-1]
        u_k = controls[:, k]

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
        
        

        x_sim.append(x_next)
        l_sim.append(l_sim[-1] + l_next)
        timings.append(elapsed_time)
    
    # timings_min = timings
        # else:
        #     timings_min = [min(timings[i], timings_min[i]) for i in range(len(timings))]

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
            timings.append(get_time_casadi_fun(jac_fun))
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



############################    CASADOS INTEGRATOR TEST    ###############################

TOL = 1e-10
x0 = model_x[:, 0]  # initial state
# x_ref = model_x      # full reference trajectory
# u_seq = model_u      # full reference input sequence

# print(f"shape of x0 : {model_x[:,0].shape}")  # should print (11,1)

# x_sim = [x0]
# x_curr = x0

# controls = u_seq

# CASADI SIMULATOR
# dyn = user_input['dyn']

# # Create correct symbolic variables
# x = ca.MX.sym('x', 11)
# xdot = ca.MX.sym('xdot', 11)
# u = ca.MX.sym('u', 3)
# z = ca.MX.sym('z', 1)

# # Create initial guess
# z0_default = ca.DM.zeros(12, 1)

# dae = {
#     'x': x,
#     'z': ca.vertcat(xdot, z),  # BOTH xdot and z treated as algebraic variables
#     'p': u,                   # controls
#     'ode': ca.MX.zeros(x.shape[0], 1),   # xdot is hidden inside z
#     'alg': dyn(xdot, x, u, z),       # full residual f(xdot, x, u, z)
#     'quad': ca.vertcat(0)
# }

# x_ref_array = model_x
controls = model_u

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
f=user_input['f']
l=user_input['l']
# CASADOS
# integrator, f_cas, l_cas = acados_simulator.create_awe_casados_integrator(user_input['dyn'], user_input['ts'], use_cython=False)
# x_sim_cas, l_sim_cas, timings_cas = run_simulation(f_cas, l_cas, x0, controls, N)
x_sim_casados, l_sim_casados, timings_casados = run_simulation(f, l, x0, controls, N,True)

x = ca.MX.sym('x', 11)
x_awe=x
u = ca.MX.sym('u',3)
# u_awe = ca.vertcat(0.0,0.0,0.0,u)
u_awe =u

# xf = integrator(x0=x_awe, p=u_awe)['xf']
# qf = integrator(x0=x_awe, p=u_awe)['qf']

# f_int=ca.Function('F',[x,u],[xf,qf],['x0','p'],['xf','qf'])


# # cost function
# power_output = -f_int(x0=x, p=u)['qf']/(user_input['ts']*N)/1e3
# regularization = 1/2*1e-4*ca.mtimes(u.T,u)

# cost = ca.Function(
#     'cost',
#     [x,u],
#     [power_output + regularization] #+ extra_regularization
# )

# x_sim_casados, l_sim_casados, timings_casados = run_simulation(f_int,cost, x0, controls, N)

x_ref_array = np.array(model_x.T).squeeze()
# x_ref_array = model_x
x_sim_casados= np.array(x_sim_casados)
# x_sim_cas= np.array(x_sim_cas)


# len_ref = x_ref_array.shape[0]
# len_sim = x_sim_casados.shape[0]
# max_len = max(len_ref, len_sim)
# x_sim_padded = np.pad(x_sim_casados, ((0, max_len - len_sim), (0, 0)), mode='constant')



# data = {
#     'x_ref': x_ref_array[:, 0],
#     'y_ref': x_ref_array[:, 1],
#     'z_ref': x_ref_array[:, 2],
#     'x_sim_casados': x_sim_cas[:, 0],
#     'y_sim_casados': x_sim_cas[:, 1],
#     'z_sim_casados': x_sim_cas[:, 2],
#     'x_sim_irk': x_sim_casados[:, 0],
#     'y_sim_irk': x_sim_casados[:, 1],
#     'z_sim_irk': x_sim_casados[:, 2],
# }      

# df_padded = pd.DataFrame(data)

# # Save to txt file
# padded_file_path = "trajectories_comparision.txt"
# df_padded.to_csv(padded_file_path, index=False, sep='\t')


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# plt.plot([xx[0] for xx in model_x], label='AWEBox')


# # ax.plot(x_integrator_array[0, :], x_integrator_array[1, :], x_integrator_array[2, :], label='Casadi Trajectory')
ax.plot(x_sim_casados[:,0], x_sim_casados[:,1], x_sim_casados[:,2], label='IRK Trajectory', linewidth=3,color='g')
# ax.plot(x_sim_cas[:,0], x_sim_cas[:,1], x_sim_cas[:,2], label='Casados Trajectory', linewidth=2, color='b')
ax.plot(x_ref_array[:,0], x_ref_array[:,1], x_ref_array[:,2], label='Reference Trajectory', linestyle='--', linewidth=1,color='r')
# # ax.plot(x_sim_rk[:,0], x_sim_rk[:,1], x_sim_rk[:,2], label='Casados Trajectory')

ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_zlabel('Z [m]')
ax.legend()
ax.set_title('Kite Trajectories')
plt.show()



# set-up tuning problem
tuner = tunempc.Tuner(
    f,
    l,
    h = user_input['h'],
    p = user_input['p']
)

# tuner = tunempc.Tuner(
#     f_int,
#     cost,
#     h = user_input['h'],
#     p = user_input['p']
# )

# solve OCP
wsol = tuner.solve_ocp(w0 = user_input['w0'])

# convexify stage cost matrices
Hc   = tuner.convexify(rho=2, solver='mosek',force=True)
# Hc   = tuner.convexify(solver='cvxopt')
S    = tuner.S

sys = tuner.sys
sys['vars'] = {
    'x': sys['vars']['x'].shape,
    'u': sys['vars']['u'].shape,
    'us': sys['vars']['us'].shape
    }

sol = {
    'S': S,
    'wsol': wsol,
    'lam_g': tuner.pocp.lam_g,
    'indeces_As': tuner.pocp.indeces_As,
    'sys': sys,
}



SAVE = True
if SAVE:
    with open(convex_ref_file,'wb') as f:
        pickle.dump(sol,f)

    # with open(user_input_file, 'wb') as f:
    #     pickle.dump(user_input, f)
