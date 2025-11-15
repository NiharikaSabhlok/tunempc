# run_closed_loop_batch.py
# Closed-loop with disturbances; batch over (T, N, beta, acc_reg).
# Saves plots/data into folders: T_<T>_N_<N>_beta_<beta>_accreg_<acc_reg>
import os, json, csv, pickle, copy, collections, math
from pathlib import Path
import numpy as np
import casadi as ca
import casadi.tools as ct

# ---- Make matplotlib non-interactive
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
CSV_PATH   = ROOT / "master_filtered.csv"         # CSV with columns: t, n, beta, accreg (names can vary; we auto-map)
# user_pickle_folder    = Path("F:/Thesis/Parameter_sweep_analysis/Analysis_results/files/test_input_files")  # folder with user_input pickles
# convex_pickle_folder  = Path("F:/Thesis/Parameter_sweep_analysis/Analysis_results/files/test_convexified_files")
user_pickle_folder    = ROOT / "input_files"  # folder with user_input pickles
convex_pickle_folder  = ROOT / "convexified_files"  # folder with convexified reference pickles
ROW_RANGE  = (2, 3)              # inclusive (0-based): process rows 10..24
NSIM       = 60                    # closed-loop steps
NMPC       = 10                     # prediction horizon
WSTD       = 0.00                  # process-noise std (0 disables)
VSTD       = 0.00                  # measurement-noise std (0 disables)
SEED       = 0                   # RNG seed for noise
STATE_IDX  = 2                     # state index for deviation plot
OUT_ROOT   = ROOT / "simulation_files"
# ============================================================

# Your pickle file-naming convention; adapt to your filenames!
# Must yield two files per combo:
#   1) user_input (with keys f/h/l/p/ts/...) and
#   2) convexified reference 'sol' (wsol, S, lam_g, sys, indeces_As, ...)
FILE_PATTERN = {
  "user": "kitepower_user_input_{T}_w_1_tpw_{N}_beta0_{beta}_acc_reg_{accreg}.pkl",
  "ref" : "convex_referencefile_{T}_w_1_tpw_{N}_beta0_{beta}_acc_reg_{accreg}.pkl",
}

# Simulation length in MPC steps
NSIM = 60

# Disturbance settings
GAUSS_W_STD = 0.02  # process noise std (scalar or len-nx array). Set 0.0 for none


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
    opts["max_iter"] = 250

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
    power = lagrange_multiplier * x_l_t * x_dl_t
    return power
        

def closed_loop_with_noise(ctrls, F, user_input, sol, Nsim, gauss_w_std=0.0, gauss_v_std=0.0, seed=0):
    nx = int(sol["sys"]["vars"]["x"].shape[0])
    nu = int(sol["sys"]["vars"]["u"].shape[0])
    p  = int(user_input["p"])
    
    l_opt, h_opt, x_ref, u_ref, z_ref, power_ref = [], [], [], [], [], []
    
    for k in range(Nsim):
        xr = sol["wsol"]["x", k % p]
        ur = sol["wsol"]["u", k % p]
        x_ref.append(xr)
        u_ref.append(ur)
        l_opt.append(float(user_input["l"](xr, ur).full()[0, 0]))
        z_ref.append(float(user_input["z"](xr, ur).full()[0, 0]))
        power_ref.append(calculate_power(z_ref,xr['l_t'],xr['dl_t']))
        if "h" in user_input:
            h_opt.append(float(user_input["h"](xr, ur).full()[0, 0]))
        else:
            h_opt.append(0.0)
    # add terminal reference state so x_ref has length Nsim+1
    x_ref.append(sol["wsol"]["x", (Nsim % p)])

    rng = np.random.default_rng(seed)
    w_std = (np.ones(nx)*gauss_w_std).reshape(nx,1) if np.isscalar(gauss_w_std) else np.array(gauss_w_std).reshape(nx,1)
    v_std = (np.ones(nx)*gauss_v_std).reshape(nx,1) if np.isscalar(gauss_v_std) else np.array(gauss_v_std).reshape(nx,1)

    x0 = sol["wsol"]["x",0]
    log = {key:{name:[] for name in ctrls.keys()} for key in ["x","u","l","h","power","lambda"]}

    for name, ctrl in ctrls.items():
        ctrl.reset()
        x_true = copy.deepcopy(x0)
        log["x"][name].append(x_true)

        for k in range(Nsim):
            # measurement noise (optional)
            x_meas = x_true
            if np.any(v_std):
                x_meas = ca.DM(np.array(x_true) + rng.normal(0.0, v_std).reshape(nx,1))

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

            # process noise after propagation
            if np.any(w_std):
                x_next = ca.DM(np.array(x_next) + rng.normal(0.0, w_std).reshape(nx,1))

            # log
            log["u"][name].append(u)
            log["l"][name].append(lk)
            log["h"][name].append(hk)
            x_true = x_next
            log["x"][name].append(x_true)

        ctrl.reset()
    return log

def summarize(log):
    out = {}
    for name in log["u"].keys():
        ls = np.array(log["l"][name], dtype=float).ravel()
        hs = np.array(log["h"][name], dtype=float).ravel()
        out[name] = {
            "J_sum": float(ls.sum()),
            "J_avg": float(ls.mean()),
            "vio_sum": float(np.sum(np.maximum(0.0, -hs)))
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

def plot_and_save(outdir: Path, log, sol, user_input, state_idx=0):
    """Save 3 figures: stage cost; state deviation; first input. PNG + PDF."""
    names = list(log["u"].keys())
    Nsim  = len(log["u"][names[0]])
    p     = int(user_input["p"])

    t  = np.arange(Nsim)     * (1.0/p)
    tx = np.arange(Nsim + 1) * (1.0/p)

    # Stage cost
    plt.figure()
    for nm in names:
        plt.step(t, np.array(log["l"][nm]).ravel(), where="post", label=nm)
    plt.grid(True); plt.legend(); plt.title("Stage cost l(x,u)"); plt.xlabel("time [cycles]")
    plt.savefig(outdir/"stage_cost.png", dpi=200, bbox_inches="tight")
    plt.savefig(outdir/"stage_cost.pdf", bbox_inches="tight")
    plt.close()

    # State deviation at index
    plt.figure()
    for nm in names:
        xs   = [float(log["x"][nm][k][state_idx]) for k in range(Nsim+1)]
        xref = [float(sol["wsol"]["x", (k % p)][state_idx]) for k in range(Nsim+1)]
        plt.plot(tx, np.array(xs)-np.array(xref), label=nm)
    plt.plot(tx, np.zeros_like(tx), "k--", linewidth=1)
    plt.grid(True); plt.legend(); plt.title(f"State deviation x[{state_idx}] - x_ref"); plt.xlabel("time [cycles]")
    plt.savefig(outdir/"state_deviation.png", dpi=200, bbox_inches="tight")
    plt.savefig(outdir/"state_deviation.pdf", bbox_inches="tight")
    plt.close()

    # First input
    plt.figure()
    for nm in names:
        u0 = [float(log["u"][nm][k][0]) for k in range(Nsim)]
        plt.step(t, u0, where="post", label=nm)
    plt.grid(True); plt.legend(); plt.title("Input u[0]"); plt.xlabel("time [cycles]")
    plt.savefig(outdir/"input_u0.png", dpi=200, bbox_inches="tight")
    plt.savefig(outdir/"input_u0.pdf", bbox_inches="tight")
    plt.close()

# --------------------------------------------------------------------------------------
# 3) MAIN LOOP
# --------------------------------------------------------------------------------------
def run_one_combo(T, Nmpc, beta, accreg, seed):
    outdir = OUT_ROOT / f"T_{T}_N_{Nmpc}_beta_{beta}_accreg_{accreg}"
    ensure_dir(outdir)

    # Resolve pickle paths from pattern
    user_file = Path(FILE_PATTERN["user"].format(T=T, N=Nmpc, beta=beta, accreg=accreg))
    ref_file  = Path(FILE_PATTERN["ref" ].format(T=T, N=Nmpc, beta=beta, accreg=accreg))

    if not user_file.exists() or not ref_file.exists():
        raise FileNotFoundError(f"Missing files:\n  {user_file}\n  {ref_file}")

    user_input, sol = load_pickles(user_file, ref_file)
    ctrls, mpc_sys = build_controllers(user_input, sol, Nmpc)
    F = mpc_sys["f"]    # discrete plant map used by Pmpc internally. :contentReference[oaicite:8]{index=8}

    # Closed loop with disturbances
    log = closed_loop_with_noise(
        ctrls, F, user_input, sol, Nsim=NSIM,
        gauss_w_std=GAUSS_W_STD, gauss_v_std=GAUSS_V_STD, seed=seed
    )

    # Save raw log
    with open(outdir/"log.pkl", "wb") as f:
        pickle.dump(log, f)

    # Save per-controller CSVs and plots
    for name in ctrls.keys():
        save_controller_csvs(outdir, name, log, sol, user_input)

    plot_and_save(outdir, log, sol, user_input, state_idx=STATE_IDX_TO_PLOT)

    # KPIs
    summ = summarize(log)
    with open(outdir/"summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["controller","J_sum","J_avg","vio_sum"])
        for k,v in summ.items():
            w.writerow([k, v["J_sum"], v["J_avg"], v["vio_sum"]])

    # Small README
    readme = {
        "T":T, "Nmpc":Nmpc, "beta":beta, "acc_reg":accreg, "seed":seed,
        "NSIM":NSIM, "GAUSS_W_STD":GAUSS_W_STD, "GAUSS_V_STD":GAUSS_V_STD,
        "user_file": str(user_file), "ref_file": str(ref_file),
        "notes":"Closed-loop with additive process noise after each step; data saved as CSV/PNG/PDF."
    }
    with open(outdir/"README.txt", "w") as f:
        f.write(json.dumps(readme, indent=2))

    return outdir

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
            user_pkl_name = Path(f"kitepower_user_input_{n}_w_1_tpw_{t}_beta0_{beta}_acc_reg_{format(accrg, '.1f')}.pkl")
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
        log = closed_loop_with_noise(
            ctrls, F, user_input, sol,
            Nsim=n
            # , wstd=WSTD, vstd=VSTD, seed=SEED
        )

        # Output folder for this combo
        outdir = OUT_ROOT / f"T_{t}_N_{n}_beta_{beta}_accreg_{accrg}"
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
            w.writerow(["controller", "J_sum", "J_avg", "vio_sum"])
            for k, v in summ.items():
                w.writerow([k, v["J_sum"], v["J_avg"], v["vio_sum"]])

        # Headless plots
        plot_and_save(outdir, log, sol, user_input, state_idx=STATE_IDX)

        # README metadata
        meta = {
            "row_index": int(ridx),
            "t": t, "n": n, "beta": beta, "acc_reg": accrg,
            "Nsim": NSIM, "Nmpc": NMPC, "wstd": WSTD, "vstd": VSTD, "seed": SEED,
            "user_pickle": str(user_pkl), "convexified_pickle": str(conv_pkl),
            "notes": "Closed-loop with optional disturbances; non-interactive plots saved."
        }
        with open(outdir / "README.txt", "w") as f:
            f.write(json.dumps(meta, indent=2))

        print(f"✔ Row {ridx}: saved -> {outdir}")

if __name__ == "__main__":
    main()
