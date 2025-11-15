# run_closed_loop_batch.py
# Closed-loop with disturbances; batch over (T, N, beta, acc_reg).
# Saves plots/data into folders: T_<T>_N_<N>_beta_<beta>_accreg_<acc_reg>
import os, json, csv, pickle, copy, collections, math
from pathlib import Path
import numpy as np
import casadi as ca
import casadi.tools as ct
import pandas as pd

# ---- Make matplotlib non-interactive
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# ---- TuneMPC
import tunempc
import tunempc.pmpc as pmpc
import tunempc.preprocessing as preprocessing
import tunempc.mtools as mtools
from tunempc.logger import Logger

# --------------------------------------------------------------------------------------
# 1) USER CONFIG
# --------------------------------------------------------------------------------------
ROOT=Path("/pfs/data6/home/fr/fr_fr/fr_ns591/code/examples/files")
CSV_PATH   = ROOT / "master_filtered_new.csv"         # CSV with columns: t, n, beta, accreg (names can vary; we auto-map)
# user_pickle_folder    = Path("F:/Thesis/Parameter_sweep_analysis/Analysis_results/files/test_input_files")  # folder with user_input pickles
# convex_pickle_folder  = Path("F:/Thesis/Parameter_sweep_analysis/Analysis_results/files/test_convexified_files")
user_pickle_folder    = ROOT / "input_files"  # folder with user_input pickles
convex_pickle_folder  = ROOT / "convexified_files"  # folder with convexified reference pickles
ROW_RANGE  = (4,5)              # inclusive (0-based): process rows 10..24 #299
# NSIM       = 66                    # closed-loop steps
NMPC       = 10                     # prediction horizon
WSTD       = 0.00                  # process-noise std (0 disables)
VSTD       = 0.00                  # measurement-noise std (0 disables)
SEED       = 0                   # RNG seed for noise
STATE_IDX_TO_PLOT  = 0                     # state index for deviation plot
STATE_IDX_TO_PLOT_2  = 2                    # state index for deviation plot
OUT_ROOT   = ROOT / "simulation_files"
index_lt = 8
index_dl_t = 9
index_lambda = -1
scaling_x = [13.5916, 13.5916, 13.5916, 6.88403, 6.88403, 6.88403, 0.6, 1, 500, 6.88403, 1]
scaling_u =[0.08, 1, 50]
scaling_lambda = 4.96665

# ============================================================

# if b_add_noise :
#                 if k in indices_for_adding_noise:
#                     x_meas[x_vel_index] = x_true[x_vel_index]*( 1 + noise_to_be_added[k])
#                     x_meas[y_vel_index] = x_true[y_vel_index]*( 1 + noise_to_be_added[k])
#                     x_meas[z_vel_index] = x_true[z_vel_index]*( 1 + noise_to_be_added[k])

b_add_noise = True
CONFIG=1
indices_for_adding_noise=[]
array_of_percentage_of_traj = [0.4,0.6]
noise_to_be_added=[0.5,0.8]
dz=5
DISTURB_VELOCITY = False
DISTURB_POSITION = True
x_pos_index=0
y_pos_index=1
z_pos_index=2
x_vel_index=3
y_vel_index=4
z_vel_index=5

# Your pickle file-naming convention; adapt to your filenames!
# Must yield two files per combo:
#   1) user_input (with keys f/h/l/p/ts/...) and
#   2) convexified reference 'sol' (wsol, S, lam_g, sys, indeces_As, ...)
FILE_PATTERN = {
  "user": "kitepower_user_input_{T}_w_1_tpw_{N}_beta0_{beta}_acc_reg_{accreg}_with_z.pkl",
  "ref" : "convex_referencefile_{T}_w_1_tpw_{N}_beta0_{beta}_acc_reg_{accreg}.pkl",
}

# Simulation length in MPC steps
NSIM = 66

# Disturbance settings
GAUSS_W_STD = 0.02  # process noise std (scalar or len-nx array). Set 0.0 for none

controller_colors = {"EMPC" : "#5726DD" ,
                     "TUNEMPC" : "#11BC83",
                     "TMPC_1" : "#ED9F17",
                     "TMPC_2" : "#E23838" 
}

controller_linestyle = {"EMPC" : "-" ,
                        "TUNEMPC" : "--",
                        "TMPC_1" : "-.",
                        "TMPC_2" : ":" 
}

controller_linewidth = {"EMPC" :  2,
                        "TUNEMPC" : 1.8,
                        "TMPC_1" : 1.6,
                        "TMPC_2" : 1.5 
}

COLOR_OF_REFERENCE = "#13161C"
LINESTYLE_OF_REFERENCE = "-"
LINEWIDTH_OF_REFERENCE = 2.2


# --------------------------------------------------------------------------------------
# 2) HELPERS
# --------------------------------------------------------------------------------------
def load_pickles(user_file: Path, ref_file: Path):
    with open(user_file, "rb") as f:
        user_input = pickle.load(f)
    with open(ref_file, "rb") as f:
        sol = pickle.load(f)
    # Rebuild CasADi symbols (needed after unpickle), per your open-loop script. :contentReference[oaicite:1]{index=1}
    vars_ord = collections.OrderedDict()
    for var in ["x","u","us"]:
        vars_ord[var] = ca.MX.sym(var, sol["sys"]["vars"][var])
    sol["sys"]["vars"] = vars_ord
    return user_input, sol

def build_controllers(user_input, sol, Nmpc):
    nx = int(sol["sys"]["vars"]["x"].shape[0])
    nu = int(sol["sys"]["vars"]["u"].shape[0])
    ns = int(sol["sys"]["vars"]["us"].shape[0])

    # Options (projection op & presolve) — same as your open-loop. :contentReference[oaicite:2]{index=2}
    opts = {}
    opts["p_operator"] = ca.Function(
        "p_operator",
        [sol["sys"]["vars"]["x"]],
        [ct.vertcat(sol["sys"]["vars"]["x"][1:3], sol["sys"]["vars"]["x"][4:])]
    )
    opts["ipopt_presolve"] = True
    opts["max_iter"] = 400

    # Add MPC slacks to active constraints. :contentReference[oaicite:3]{index=3}
    mpc_sys = preprocessing.add_mpc_slacks(
        sol["sys"], sol["lam_g"], sol["indeces_As"], slack_flag="active"
    )
    
    Logger.logger.info(20*'=')
    Logger.logger.info(10*' '+'Building Controllers...')
    Logger.logger.info(20*'=')

    ctrls = {}

    # EMPC — economic cost l(x,u). Uses Pmpc.step(...) online. :contentReference[oaicite:4]{index=4}
    ctrls["EMPC"] = pmpc.Pmpc(
        N=Nmpc, sys=mpc_sys, cost=user_input["l"],
        wref=sol["wsol"], lam_g_ref=sol["lam_g"],
        sensitivities=sol["S"], options=opts
    )

    # Tracking shell + dual resets
    tracking_cost = mtools.tracking_cost(nx + nu + ns)
    lam_g0 = copy.deepcopy(sol["lam_g"])
    lam_g0["dyn"] = 0.0
    lam_g0["g"]   = 0.0

    # TMPC_1 — vanilla tracking weights
    tuning_t1 = {"H":[np.diag((nx+nu)*[1.0] + ns*[1e-10])]*user_input["p"],
                 "q":sol["S"]["q"]}
    ctrls["TMPC_1"] = pmpc.Pmpc(
        N=Nmpc, sys=mpc_sys, cost=tracking_cost,
        wref=sol["wsol"], tuning=tuning_t1, lam_g_ref=lam_g0,
        sensitivities=sol["S"], options=opts
    )

    # TMPC_2 — hand-tuned (from your open-loop) :contentReference[oaicite:5]{index=5}
    Ht2 = [np.diag([0.1,0.1,0.1, 1,1,1, 1e3, 1,100, 1,1,1, 1,1] + [1e-10]*ns)]*user_input["p"]
    tuning_t2 = {"H":Ht2, "q":sol["S"]["q"]}
    ctrls["TMPC_2"] = pmpc.Pmpc(
        N=Nmpc, sys=mpc_sys, cost=tracking_cost,
        wref=sol["wsol"], tuning=tuning_t2, lam_g_ref=lam_g0,
        sensitivities=sol["S"], options=opts
    )

    # TUNEMPC — convexified Hessians (first-order equivalent tracker). 
    tuning_tuned = {"H":sol["S"]["Hc"], "q":sol["S"]["q"]}
    ctrls["TUNEMPC"] = pmpc.Pmpc(
        N=Nmpc, sys=mpc_sys, cost=tracking_cost,
        wref=sol["wsol"], tuning=tuning_tuned, lam_g_ref=lam_g0,
        sensitivities=sol["S"], options=opts
    )

    return ctrls, mpc_sys

def simulate_forward(F, x, u):
    """Advance plant with sys['f'] which has signature F(x0, p) -> {'xf':...}."""
    try:   return F(x, u)['xf']
    except: return F(x0=x, p=u)['xf']
    
def calculate_power(lagrange_multiplier, x_l_t, x_dl_t):
    # power = abs(lagrange_multiplier) * x_l_t * x_dl_t
    power = lagrange_multiplier * x_l_t * x_dl_t
    return power
        

def closed_loop_with_noise(ctrls, F, user_input, sol, Nsim, gauss_w_std=0.0, gauss_v_std=0.0, seed=0):
    nx = int(sol["sys"]["vars"]["x"].shape[0])
    nu = int(sol["sys"]["vars"]["u"].shape[0])
    p  = int(user_input["p"])
    
    indices_for_adding_noise = (p * array_of_percentage_of_traj).astype(int)
    
    l_opt, h_opt, x_ref, u_ref, z_ref, power_ref, lt_ref, dlt_ref = [], [], [], [], [], [],[],[]
    log_ref = {"x_ref": [], "u_ref": [], "l_ref": [], "h_ref": [], "power_ref": [], "avg_power":[]}
    
    for k in range(Nsim):
        xr = sol["wsol"]["x", k % p]
        ur = sol["wsol"]["u", k % p]
        x_ref.append(xr)
        u_ref.append(ur)
        l_opt.append(float(user_input["l"](xr, ur).full()[0, 0]))
        # z_ref.append(float(user_input["z"](xr*scaling_x, ur*scaling_u).full()[0, 0]))
        z_temp = user_input["z"](xr, ur)
        z_ref.append(user_input["z"](xr, ur).full()[-1])
        lt_ref.append(xr[index_lt]*scaling_x[index_lt])
        dlt_ref.append(xr[index_dl_t]*scaling_x[index_dl_t])
        power_ref.append(calculate_power(z_ref[k]*scaling_lambda, lt_ref[k], dlt_ref[k]))
        if "h" in user_input:
            h_opt.append(float(user_input["h"](xr, ur).full()[0, 0]))
        else:
            h_opt.append(0.0)

    # add terminal reference state so x_ref has length Nsim+1
    x_ref.append(sol["wsol"]["x", (Nsim % p)])
    power_array = np.array([float(p) for p in power_ref])
    avg_power = np.mean(power_array)
    print(avg_power)
    log_ref['avg_power']=avg_power
    
    log_ref['x_ref']=np.vstack([np.array(d.full()).ravel() for d in x_ref])
    log_ref['u_ref']=np.vstack([np.array(d.full()).ravel() for d in u_ref])
    log_ref['l_ref']=l_opt
    log_ref['power_ref']=power_array
    
    rng = np.random.default_rng(seed)
    w_std = (np.ones(nx)*gauss_w_std).reshape(nx,1) if np.isscalar(gauss_w_std) else np.array(gauss_w_std).reshape(nx,1)
    v_std = (np.ones(nx)*gauss_v_std).reshape(nx,1) if np.isscalar(gauss_v_std) else np.array(gauss_v_std).reshape(nx,1)

    x0 = sol["wsol"]["x",0]
    log = {key:{name:[] for name in ctrls.keys()} for key in ["x","u","l","h","power","lambda","avg_power"]}
    
    power_params = {key:{name:[] for name in ctrls.keys()} for key in ["z_k","lt_k","dl_t_k","power"]}

    for name, ctrl in ctrls.items():
        ctrl.reset()
        noise_index=0
        x_true = copy.deepcopy(x0)
        log["x"][name].append(x_true)

        for k in range(Nsim):
            # measurement noise (optional)
            x_meas = x_true
            if np.any(v_std):
                x_meas = ca.DM(np.array(x_true) + rng.normal(0.0, v_std).reshape(nx,1))
            if b_add_noise :
                if k in indices_for_adding_noise:
                    if DISTURB_VELOCITY :
                        x_meas[x_vel_index] = x_true[x_vel_index]*( 1 + noise_to_be_added[noise_index])
                        x_meas[y_vel_index] = x_true[y_vel_index]*( 1 + noise_to_be_added[noise_index])
                        x_meas[z_vel_index] = x_true[z_vel_index]*( 1 + noise_to_be_added[noise_index])
                        noise_index += 1
                    elif DISTURB_POSITION :
                        # x_meas[x_pos_index] = x_true[x_vel_index]*( 1 + noise_to_be_added[noise_index])
                        # x_meas[y_pos_index] = x_true[y_vel_index]*( 1 + noise_to_be_added[noise_index])
                        # x_meas[z_pos_index] = x_true[z_vel_index]*( 1 + noise_to_be_added[noise_index])
                        x_meas[z_pos_index] = x_true[z_vel_index] + dz*noise_to_be_added[noise_index]
                        x_meas[x_pos_index] = np.sqrt(-x_meas[z_pos_index]**2 - x_true[y_pos_index]**2 + x_true[index_lt]**2)
                        noise_index += 1
 
            Logger.logger.info(10*'=')
            Logger.logger.info(10*' '+f'Evaluating Step {k} of Controller {name}.')
            Logger.logger.info(10*'=')
            # MPC action
            u = ctrl.step(x_meas)                          # online control from Pmpc. :contentReference[oaicite:7]{index=7}

            # stage cost & constraint at *true* state (pre-process-noise)
            lk = float(user_input["l"](x_true, u).full()[0,0])
            hk = float(user_input["h"](x_true, u).full()[0,0]) if "h" in user_input else 0.0

            # plant propagation
            x_next = simulate_forward(F, x_true, u)
            # z_k= user_input["z"](x_true*scaling_x, u*scaling_u)
            z_k = user_input["z"](x_true, u).full()[-1]
            lt_k = x_true[index_lt]*scaling_x[index_lt]
            dl_t_k = x_true[index_dl_t]*scaling_x[index_dl_t]
            # "z_k","lt_k","dl_t_k","power"
            power_params["z_k"][name].append(z_k)
            power_params["lt_k"][name].append(float(lt_k))
            power_params["dl_t_k"][name].append(float(dl_t_k))
            power_k = calculate_power(z_k*scaling_lambda, lt_k, dl_t_k)
            power_params["power"][name].append(float(power_k))
            # process noise after propagation
            if np.any(w_std):
                x_next = ca.DM(np.array(x_next) + rng.normal(0.0, w_std).reshape(nx,1))

            # log
            log["u"][name].append(np.array(u.full()).ravel())
            log["l"][name].append(lk)
            log["h"][name].append(hk)
            log["lambda"][name].append(z_k)
            log["power"][name].append(float(power_k))
            x_true = x_next
            log["x"][name].append(x_true)
            log["avg_power"][name]=np.mean(np.array(log["power"][name]))

        ctrl.reset()
        indices_for_adding_noise.clear()
    return log, log_ref

def summarize(log):
    out = {}
    for name in log["u"].keys():
        ls = np.array(log["l"][name], dtype=float).ravel()
        hs = np.array(log["h"][name], dtype=float).ravel()
        out[name] = {
            "J_sum": float(ls.sum()),
            "J_avg": float(ls.mean()),
            "vio_sum": float(np.sum(np.maximum(0.0, -hs))),
            "avg_power": float(log["avg_power"][name])
        }
    return out

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def save_controller_csvs(outdir: Path, name: str, log, sol, user_input):
    """Save x,u,l,h for a controller as CSVs."""
    nx = int(sol["sys"]["vars"]["x"].shape[0])
    nu = int(sol["sys"]["vars"]["u"].shape[0])
    p  = int(user_input["p"])

    # x: (N+1, nx)
    X = np.vstack([np.array(xx).reshape(1,-1) for xx in log["x"][name]])
    # u: (N, nu)
    U = np.vstack([np.array(uu).reshape(1,-1) for uu in log["u"][name]])
    # l/h: (N,)
    L = np.array(log["l"][name]).reshape(-1,1)
    H = np.array(log["h"][name]).reshape(-1,1)

    np.savetxt(outdir/f"{name}_traj_x.csv", X, delimiter=",")
    np.savetxt(outdir/f"{name}_traj_u.csv", U, delimiter=",")
    np.savetxt(outdir/f"{name}_stage_cost.csv", L, delimiter=",")
    np.savetxt(outdir/f"{name}_constraint.csv", H, delimiter=",")
    
    
def _to_float_array(seq):
    return np.array([float(v) for v in seq]).reshape(-1, 1)

def _stack_dm_rows(seq):
    # seq: list of DM/arrays shaped (nx,1) -> 2D (T, nx)
    return np.vstack([np.array(v).reshape(1, -1) for v in seq])

def save_all_logs_as_csv(outdir: Path, log, log_ref, sol, user_input):
    outdir.mkdir(parents=True, exist_ok=True)
    names = list(log["u"].keys())
    Nsim  = len(log["u"][names[0]])
    nx    = int(sol["sys"]["vars"]["x"].shape[0])
    nu    = int(sol["sys"]["vars"]["u"].shape[0])
    p     = int(user_input["p"])
    t     = np.arange(Nsim) * (1.0 / p)
    tx    = np.arange(Nsim+1) * (1.0 / p)

    # --- Reference CSVs ---
    # x_ref (Nsim+1, nx)
    # Xr = _stack_dm_rows(log_ref["x_ref"])
    Xr = log_ref["x_ref"]
    pd.DataFrame(np.column_stack([tx, Xr]),
                 columns=["t_cycle"] + [f"x{i}" for i in range(nx)]
                 ).to_csv(outdir/"ref_x.csv", index=False)

    # u_ref (Nsim, nu
    Ur = log_ref["u_ref"]
    pd.DataFrame(np.column_stack([t, Ur]),
                 columns=["t_cycle"] + [f"u{i}" for i in range(nu)]
                 ).to_csv(outdir/"ref_u.csv", index=False)

    # l_ref / h_ref / power_ref (Nsim, 1) — some may be numpy already
    if "l_ref" in log_ref and len(log_ref["l_ref"]) == Nsim:
        pd.DataFrame({"t_cycle": t, "l_ref": [float(v) for v in log_ref["l_ref"]]}
                     ).to_csv(outdir/"ref_l.csv", index=False)
    if "h_ref" in log_ref and len(log_ref["h_ref"]) == Nsim:
        pd.DataFrame({"t_cycle": t, "h_ref": [float(v) for v in log_ref["h_ref"]]}
                     ).to_csv(outdir/"ref_h.csv", index=False)
    if "power_ref" in log_ref and len(log_ref["power_ref"]) >= Nsim:
        Pref = np.array(log_ref["power_ref"]).ravel()[:Nsim]
        pd.DataFrame({"t_cycle": t, "power_ref": Pref}
                     ).to_csv(outdir/"ref_power.csv", index=False)

    # --- Per-controller CSVs ---
        # --- Per-controller CSVs ---
    for nm in names:
        # x: list of DM -> (Nsim+1, nx)
        X_list = log["x"][nm]          # list of length Nsim+1
        X = np.vstack([np.array(xx).reshape(1, -1) for xx in X_list])

        pd.DataFrame(
            np.column_stack([tx, X]),
            columns=["t_cycle"] + [f"x{i}" for i in range(nx)]
        ).to_csv(outdir / f"{nm}_x.csv", index=False)

        # u: list of DM -> (Nsim, nu)
        U_list = log["u"][nm]          # list of length Nsim
        U = np.vstack([np.array(uu).reshape(1, -1) for uu in U_list])

        pd.DataFrame(
            np.column_stack([t, U]),
            columns=["t_cycle"] + [f"u{i}" for i in range(nu)]
        ).to_csv(outdir / f"{nm}_u.csv", index=False)

        # l, h, power: 1D -> (Nsim,)
        L = np.array([float(v) for v in log["l"][nm]]).ravel()
        H = np.array([float(v) for v in log["h"][nm]]).ravel()
        P = np.array([float(v) for v in log["power"][nm]]).ravel()

        pd.DataFrame(
            {"t_cycle": t, "l": L, "h": H, "power": P}
        ).to_csv(outdir / f"{nm}_lhp.csv", index=False)

        # lambda: could be scalar or vector per step
        lam_list = log.get("lambda", {}).get(nm, [])
        if lam_list:
            lam_list = [
                np.atleast_1d(np.array(v).astype(float).ravel())
                for v in lam_list
            ]
            maxlen = max(len(v) for v in lam_list)
            LAM = np.zeros((len(lam_list), maxlen))
            for i, v in enumerate(lam_list):
                LAM[i, :len(v)] = v

            cols = ["t_cycle"] + [f"lambda{i}" for i in range(maxlen)]
            pd.DataFrame(
                np.column_stack([t, LAM]),
                columns=cols
            ).to_csv(outdir / f"{nm}_lambda.csv", index=False)
            
def latexify():
    import matplotlib
    params_MPL_Tex = {
                'text.usetex': True,
                'font.family': 'serif',
                # Use 10pt font in plots, to match 10pt font in document
                "axes.labelsize": 10,
                "font.size": 10,
                # Make the legend/label fonts a little smaller
                "legend.fontsize": 8,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8
              }
    matplotlib.rcParams.update(params_MPL_Tex)

def draw_tethers_3d(ax, X, Y, Z, x_g=0.0, y_g=0.0, z_g=0.0, step=5,color_line="gray"):
    """
    Plot very light dotted lines from ground point (x_g,y_g,z_g)
    to each (X[k], Y[k], Z[k]) every `step` samples.
    """
    for k in range(0, len(X), step):
        ax.plot(
            [x_g, X[k]],
            [y_g, Y[k]],
            [z_g, Z[k]],
            linestyle=":",
            linewidth=0.7,
            color=color_line,
            alpha=0.35,
            zorder=0,
        )


def plot_and_save(outdir: Path, log, sol, user_input, state_idx=0, state_idx_2=2):
    """Save 3 figures: stage cost; state deviation; first input. PNG + PDF."""
    names = list(log["u"].keys())
    Nsim  = len(log["u"][names[0]])
    p     = int(user_input["p"])

    t  = np.arange(Nsim)     * (1.0/p)
    tx = np.arange(Nsim + 1) * (1.0/p)

    # Stage cost
    plt.figure()
    for nm in names:
        plt.step(t, np.array(log["l"][nm]).ravel(), where="post", label=nm,color=controller_colors[nm], linestyle=controller_linestyle[nm],linewidth=controller_linewidth[nm])
    plt.grid(True); plt.legend(); plt.title("Stage cost l(x,u)"); plt.xlabel(r"time [cycles]"); plt.ylabel(r"Stage cost $l(x,u)$")
    plt.savefig(outdir/"stage_cost.png", dpi=200, bbox_inches="tight")
    plt.savefig(outdir/"stage_cost.pdf", bbox_inches="tight")
    plt.close()

    # State deviation at index
    plt.figure()
    for nm in names:
        xs   = [float(log["x"][nm][k][state_idx]) for k in range(Nsim+1)]
        xref = [float(sol["wsol"]["x", (k % p)][state_idx]) for k in range(Nsim+1)]
        plt.plot(tx, np.array(xs)-np.array(xref), label=nm,color=controller_colors[nm], linestyle=controller_linestyle[nm],linewidth=controller_linewidth[nm])
    plt.plot(tx, np.zeros_like(tx), "k--", linewidth=1)
    plt.grid(True); plt.legend(); plt.title(fr"State deviation $x[{state_idx}] - x_{{\text{{ref}}}}[{state_idx}]$ [m]")
    plt.xlabel(r"time [cycles]")
    plt.ylabel(fr"$x[{state_idx}] - x_{{\text{{ref}}}}[{state_idx}]$ [m]")
    plt.savefig(outdir/"state_deviation.png", dpi=200, bbox_inches="tight")
    plt.savefig(outdir/"state_deviation.pdf", bbox_inches="tight")
    plt.close()
    
    # State deviation at index
    plt.figure()
    for nm in names:
        xs   = [float(log["x"][nm][k][state_idx_2]) for k in range(Nsim+1)]
        xref = [float(sol["wsol"]["x", (k % p)][state_idx_2]) for k in range(Nsim+1)]
        plt.plot(tx, np.array(xs)-np.array(xref), label=nm, color=controller_colors[nm], linestyle=controller_linestyle[nm],linewidth=controller_linewidth[nm])
    plt.plot(tx, np.zeros_like(tx), "k--", linewidth=1)
    plt.grid(True); plt.legend(); plt.title(fr"State deviation $x[{state_idx_2}] - x_{{\text{{ref}}}}[{state_idx_2}]$ [m]")
    plt.xlabel(r"time [cycles]")
    plt.ylabel(fr"$x[{state_idx_2}] - x_{{\text{{ref}}}}[{state_idx_2}]$ [m]")
    plt.savefig(outdir/"state_deviation_2.png", dpi=200, bbox_inches="tight")
    plt.savefig(outdir/"state_deviation_2.pdf", bbox_inches="tight")
    plt.close()

    # First input
    plt.figure()
    for nm in names:
        u0 = [float(log["u"][nm][k][0]) for k in range(Nsim)]
        plt.step(t, u0, where="post", label=nm,color=controller_colors[nm], linestyle=controller_linestyle[nm],linewidth=controller_linewidth[nm])
    plt.grid(True); plt.legend(); plt.title("Input u[0]"); plt.xlabel(r"time [cycles]"); plt.ylabel(r"Input $u$")
    plt.savefig(outdir/"input_u0.png", dpi=200, bbox_inches="tight")
    plt.savefig(outdir/"input_u0.pdf", bbox_inches="tight")
    plt.close()
    
def plot_power_tracking(outdir: Path, log, log_ref, user_input):
    """Plot power of all controllers per step vs reference power."""
    names = list(log["u"].keys())
    Nsim  = len(log["u"][names[0]])
    p     = int(user_input["p"])
    t     = np.arange(Nsim) * (1.0 / p)

    plt.figure()
    # reference power
    if "power_ref" in log_ref and len(log_ref["power_ref"]) >= Nsim:
        Pref = np.array(log_ref["power_ref"]).ravel()[:Nsim]
        plt.step(t, Pref/1000, where="post", label="Reference", color=COLOR_OF_REFERENCE, linestyle=LINESTYLE_OF_REFERENCE, linewidth=LINEWIDTH_OF_REFERENCE)

    # controllers
    for nm in names:
        Pk = np.array([float(v) for v in log["power"][nm]]).ravel()
        plt.step(t, Pk/1000, where="post", label=nm, alpha=0.9,color=controller_colors[nm], linestyle=controller_linestyle[nm],linewidth=controller_linewidth[nm])

    plt.grid(True); plt.legend()
    plt.title("Instantaneous power vs reference")
    plt.xlabel(r"time [cycles]")
    plt.ylabel(r"Power $P [kW]$")
    plt.savefig(outdir/"power_tracking.png", dpi=200, bbox_inches="tight")
    plt.savefig(outdir/"power_tracking.pdf", bbox_inches="tight")
    plt.close()

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)
X_IDX, Y_IDX, Z_IDX = 0, 1, 2


def plot_trajectory_all(outdir: Path, log, log_ref, sol, user_input,
                        x_idx=0, y_idx=1, z_idx=2):
    """Single 3D figure: reference + all controllers."""
    names = list(log["u"].keys())
    Nsim  = len(log["u"][names[0]])
    p     = int(user_input["p"])

    # Reference (Nsim+1 points)
    Xr = [float(log_ref["x_ref"][k][x_idx] * scaling_x[x_idx]) for k in range(Nsim+1)]
    Yr = [float(log_ref["x_ref"][k][y_idx] * scaling_x[y_idx]) for k in range(Nsim+1)]
    Zr = [float(log_ref["x_ref"][k][z_idx] * scaling_x[z_idx]) for k in range(Nsim+1)]

    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    ax.scatter(0, 0, 0, color='black', marker='o', s=10 ) #label='origin'
    ax.plot(Xr, Yr, Zr, label="Reference",color=COLOR_OF_REFERENCE, linestyle=LINESTYLE_OF_REFERENCE, linewidth=LINEWIDTH_OF_REFERENCE)
    draw_tethers_3d(ax, Xr, Yr, Zr, x_g=0.0, y_g=0.0, z_g=0.0, step=2,color_line="gray")
    
    for nm in names:
        Xc = [float(log["x"][nm][k][x_idx] * scaling_x[x_idx]) for k in range(Nsim+1)]
        Yc = [float(log["x"][nm][k][y_idx] * scaling_x[y_idx]) for k in range(Nsim+1)]
        Zc = [float(log["x"][nm][k][z_idx] * scaling_x[z_idx]) for k in range(Nsim+1)]
        ax.plot(Xc, Yc, Zc, label=nm, alpha=0.9,color=controller_colors[nm], linestyle=controller_linestyle[nm],linewidth=controller_linewidth[nm])

    ax.set_title("3D Trajectory: controllers vs reference")
    ax.set_xlabel(r"x [m]"); ax.set_ylabel(r"y[m]"); ax.set_zlabel(r"z[m]")
    ax.grid(True, which="both", alpha=0.25)
    ax.view_init(elev=28, azim=45) #14,38
    ax.set_box_aspect((1, 1, 1))  # equal-ish proportions
    ax.legend()
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_edgecolor("none")
        axis.pane.set_alpha(0.05)
    plt.tight_layout()
    plt.savefig(outdir/"trajectory_all_3d.png", dpi=200, bbox_inches="tight")
    plt.savefig(outdir/"trajectory_all_3d.pdf", bbox_inches="tight")
    plt.close()

def plot_trajectory_per_controller(outdir: Path, log, log_ref, sol, user_input,
                                   x_idx=0, y_idx=1, z_idx=2):
    """One 3D figure per controller with the reference trajectory."""
    names = list(log["u"].keys())
    Nsim  = len(log["u"][names[0]])

    # Reference
    Xr = [float(log_ref["x_ref"][k][x_idx] * scaling_x[x_idx]) for k in range(Nsim+1)]
    Yr = [float(log_ref["x_ref"][k][y_idx] * scaling_x[y_idx]) for k in range(Nsim+1)]
    Zr = [float(log_ref["x_ref"][k][z_idx] * scaling_x[z_idx]) for k in range(Nsim+1)]

    for nm in names:
        Xc = [float(log["x"][nm][k][x_idx] * scaling_x[x_idx]) for k in range(Nsim+1)]
        Yc = [float(log["x"][nm][k][y_idx] * scaling_x[y_idx]) for k in range(Nsim+1)]
        Zc = [float(log["x"][nm][k][z_idx] * scaling_x[z_idx]) for k in range(Nsim+1)]

        fig = plt.figure()
        ax  = fig.add_subplot(111, projection='3d')
        ax.scatter(0, 0, 0, color='black', marker='o', s=10 ) #label='origin'
        ax.plot(Xr, Yr, Zr, label="Reference",color=COLOR_OF_REFERENCE, linestyle=LINESTYLE_OF_REFERENCE, linewidth=LINEWIDTH_OF_REFERENCE)
        ax.plot(Xc, Yc, Zc, label=nm, alpha=0.95,color=controller_colors[nm], linestyle=controller_linestyle[nm],linewidth=controller_linewidth[nm])
        draw_tethers_3d(ax, Xc, Yc, Zc, x_g=0.0, y_g=0.0, z_g=0.0, step=2,color_line=controller_colors[nm])
        ax.set_title(f"3D Trajectory: {nm}")
        ax.set_xlabel(r"x [m]"); ax.set_ylabel(r"y[m]"); ax.set_zlabel(r"z[m]")
        ax.grid(True, which="both", alpha=0.25)
        ax.view_init(elev=28, azim=45)
        ax.set_box_aspect((1, 1, 1))  # equal-ish proportions
        ax.legend()
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_edgecolor("none")
            axis.pane.set_alpha(0.05)
        plt.tight_layout()
        plt.savefig(outdir/f"trajectory_{nm}_3d.png", dpi=200, bbox_inches="tight")
        plt.savefig(outdir/f"trajectory_{nm}_3d.pdf", bbox_inches="tight")
        plt.close()

# --------------------------------------------------------------------------------------
# 3) MAIN LOOP
# --------------------------------------------------------------------------------------

def main():
    ensure_dir(OUT_ROOT)

    # Load CSV and normalize headers
    import pandas as pd
    df = pd.read_csv(CSV_PATH)
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Column resolver (accepts common variants)
    def pick(*cands):
        for c in cands:
            if c in df.columns:
                return c
        raise KeyError(f"CSV missing column; looked for any of: {cands}")

    col_t    = pick("T","t")
    col_n    = pick("N","n")
    col_beta = pick("beta")
    col_acc  = pick("acc_reg")

    lo, hi = ROW_RANGE
    if not (0 <= lo <= hi < len(df)):
        raise ValueError(f"Row range {ROW_RANGE} invalid for CSV with {len(df)} rows.")

    for ridx in range(lo, hi + 1):
        t     = df.loc[ridx, col_t]
        n     = df.loc[ridx, col_n]
        beta  = df.loc[ridx, col_beta]
        accrg = df.loc[ridx, col_acc]

        # Build filenames exactly as requested
        if accrg in [0,1] :
            user_pkl_name = Path(f"kitepower_user_input_{n}_w_1_tpw_{t}_beta0_{beta}_acc_reg_{int(accrg)}.pkl")
        else:
            user_pkl_name = Path(f"kitepower_user_input_{n}_w_1_tpw_{t}_beta0_{beta}_acc_reg_{format(accrg, '.1f')}_with_z.pkl")
        conv_pkl_name = Path(f"looped_wo_warmstarting_T_{t}N_{n}_beta_{beta}_acc_reg_{format(accrg, '.1f')}")  # note: 'conevxified_' prefix

        user_pkl = Path(os.path.join(user_pickle_folder, user_pkl_name))
        conv_pkl = Path(os.path.join(convex_pickle_folder, conv_pkl_name))
        if not user_pkl.exists() or not conv_pkl.exists():
            raise FileNotFoundError(f"Missing pickle(s) for row {ridx}:\n  {user_pkl}\n  {conv_pkl}")

        # Load pickles and build controllers/system
        user_input, sol = load_pickles(user_pkl, conv_pkl)
        ctrls, mpc_sys = build_controllers(user_input, sol, NMPC)
        F = mpc_sys["f"]

        # Run closed loop with disturbances
        log, log_ref = closed_loop_with_noise(
            ctrls, F, user_input, sol,
            Nsim=NSIM
            # , wstd=WSTD, vstd=VSTD, seed=SEED
        )

        # Output folder for this combo
        outdir = OUT_ROOT / f"T_{t}_N_{n}_beta_{beta}_accreg_{accrg}_NMPC_{NSIM}_noise_config{CONFIG}"
        ensure_dir(outdir)

        # Save raw log
        with open(outdir / "log.pkl", "wb") as f:
            pickle.dump(log, f)

        # Save per-controller CSVs
        for name in ctrls.keys():
            save_controller_csvs(outdir, name, log, sol, user_input)

        # KPIs
        summ = summarize(log)
        import csv as _csv
        with open(outdir / "summary.csv", "w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["controller", "J_sum", "J_avg", "vio_sum","avg_power","ref_avg_power"])
            for k, v in summ.items():
                w.writerow([k, v["J_sum"], v["J_avg"], v["vio_sum"], v["avg_power"],log_ref['avg_power']])

        # Headless plots
        latexify()
        plot_and_save(outdir, log, sol, user_input,  state_idx=STATE_IDX_TO_PLOT, state_idx_2=STATE_IDX_TO_PLOT_2)
        plot_power_tracking(outdir, log, log_ref, user_input)
        plot_trajectory_all(outdir, log, log_ref, sol, user_input,X_IDX, Y_IDX, Z_IDX)
        plot_trajectory_per_controller(outdir, log, log_ref, sol, user_input, X_IDX, Y_IDX, Z_IDX)

        save_all_logs_as_csv(outdir, log, log_ref, sol, user_input)
        # README metadata
        # meta = {
        #     "row_index": int(ridx),
        #     "t": t, "n": n, "beta": beta, "acc_reg": accrg,
        #     "Nsim": NSIM, "Nmpc": NMPC, "wstd": WSTD, "vstd": VSTD, "seed": SEED,
        #     "user_pickle": str(user_pkl), "convexified_pickle": str(conv_pkl),
        #     "notes": "Closed-loop with optional disturbances; non-interactive plots saved."
        # }
        # with open(outdir / "README.txt", "w") as f:
        #     f.write(json.dumps(meta, indent=2))

        print(f"✔ Row {ridx}: saved -> {outdir}")

if __name__ == "__main__":
    main()
