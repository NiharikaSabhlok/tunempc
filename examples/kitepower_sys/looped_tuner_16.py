import tunempc
import pickle
import pandas as pd
from tunempc.logger import Logger
import csv
import os
import re
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


T=34
N=68
part=2
directory_path= "/pfs/data6/home/fr/fr_fr/fr_ns591/data/loop_T_34_N_68_2"
pickle_prefix="kitepower_user_input"
csv_file_prefix="looped_wo_warmstarting_avg_power_and_twall_for_beta_and_Acc_N"


FOLDER_RE = re.compile(r"^loop_T_(?P<T>\d+)_N_(?P<N>\d+)$")
PICKLE_FILE_RE = re.compile(
    r"^kitepower_user_input_(?P<N>\d+)_w_1_tpw_(?P<T>\d+)_beta0_"
    r"(?P<beta>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)_acc_reg_"
    r"(?P<acc_reg>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\.pkl$"
)

def parse_T_N_from_folder(folder: Path):
    m = FOLDER_RE.match(folder.name)
    if not m:
        return None
    try:
        T = int(m.group("T"))
        N = int(m.group("N"))
        return T, N
    except Exception:
        return None
    
def safe_load_pickle(pickle_path: Path) -> Any:
    with open(pickle_path, "rb") as f:
        return pickle.load(f)
    
def parse_from_filename(fname: str) -> Optional[Dict[str, Any]]:
    m = PICKLE_FILE_RE.match(fname)
    if not m:
        return None
    return {
        "T_file": int(m.group("T")),
        "N_file": int(m.group("N")),
        "beta": float(m.group("beta")),
        "acc_reg": float(m.group("acc_reg")),
    }
    
def tuner_convexification(pickle_file, beta, acc_reg, T, N):
    """ Convexify MPC Hessian 
    """
    Logger.logger.info(20*'=')
    Logger.logger.info(10*' '+'tuner_convexification...')
    Logger.logger.info(20*'=')
    
    with open(pickle_file,'rb') as outfile:
        user_input = pickle.load(outfile)

    tuner = tunempc.Tuner(
        f= user_input['f'],
        l= user_input['l'],
        h = user_input['h'],
        p = user_input['p']
    )
    # solve OCP
    wsol = tuner.solve_ocp(w0 = user_input['w0'])

    # convexify stage cost matrices
    Hc,solver_status, equivalence_status = tuner.convexify(rho=2, solver='mosek',force=True)
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
    
    with open(f'looped_wo_warmstarting_T_{T}N_{N}_beta_{beta}_acc_reg_{acc_reg}','wb') as convex_output:
        pickle.dump(sol,convex_output)
        
    return T,N, beta, acc_reg, solver_status, equivalence_status


def run(root: Path, out_csv: Path):
    rows = []
    for dirpath, dirnames, filenames in os.walk(root):
        # print(filenames)
        folder = Path(root)
        Logger.logger.info(f"folder path {folder }...")
        Logger.logger.info(f"root path {root }...")
        # print(f"folder path {folder }...")
        # print(f"folder path {root }...")
        T_folder = T
        N_folder = N
        print(f"Processing folder: {folder}, T={T_folder}, N={N_folder}")
        ## Open the csv file and read all the data 
        sim_data_df = pd.read_csv(os.path.join(folder,f'{csv_file_prefix}_{N_folder}_T_{T_folder}.txt'))
        sim_data_df.columns = sim_data_df.columns.str.strip()
        
        
        for fname in filenames:
            if not fname. endswith(".pkl"):
                continue
            mfile = PICKLE_FILE_RE.match(fname)
            if not mfile:
                # skip non-conforming pickles silently or log as needed
                continue

            T_file = int(mfile.group("T"))
            N_file = int(mfile.group("N"))
            beta = float(mfile.group("beta"))
            acc_reg = float(mfile.group("acc_reg"))
            tn_mismatch = (T_file != T_folder) or (N_file != N_folder)
            # Check with the data
            matched_row = sim_data_df[(sim_data_df['Beta'].astype(float) == float(beta)) & (sim_data_df['acc_reg'].astype(float) == float(acc_reg))]
            print("Matched rows found:")
            print(matched_row)
            if matched_row.empty or not matched_row.iloc[0]['b_optimal']:
                print(f"Skipping {fname} as it does not match optimal criteria.")
                continue
            
            try:
                d_T,d_N, d_beta, d_acc_reg, d_solver_status, d_equivalence_status = tuner_convexification(os.path.join(folder, fname),beta, acc_reg, T_folder, N_folder)
                rows.append({
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "folder": str(folder),
                    "pickle": str(folder / fname),
                    "T": T_folder, "N": N_folder,
                    "beta": d_beta, "acc_reg": d_acc_reg,
                    "solver_status": d_solver_status,
                    "equivalence_status": d_equivalence_status,
                    "avg_power": matched_row["avg_power"],
                    "twall": matched_row["twall"],
                    "final_time": matched_row["final_time"],
                    "b_optimal": matched_row["b_optimal"],
                    "error_type": None,
                    "error_message": None,
                    "traceback": None,
                    "tn_mismatch": tn_mismatch
                })
            except Exception as e:
                rows.append({
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "folder": str(folder),
                    "pickle": str(folder / fname),
                    "T": T_folder, "N": N_folder,
                    "beta": beta, "acc_reg": acc_reg,
                    "solver_status": "solver_error",
                    "equivalence_status": None,
                    "avg_power": matched_row["avg_power"],
                    "twall": matched_row["twall"],
                    "final_time": matched_row["final_time"],
                    "b_optimal": matched_row["b_optimal"],
                    "error_type": type(e).__name__,
                    "error_message": str(e),
                    "traceback": traceback.format_exc().strip(),
                    "tn_mismatch": tn_mismatch
                })

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "timestamp","folder","pickle","T","N","beta","acc_reg",
            "solver_status","equivalence_status","avg_power", "twall", 
            "final_time", "b_optimal", "error_type","error_message",
            "traceback","tn_mismatch"
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    
    # edit these two lines (or make them CLI args if you like)
    ROOT = Path(directory_path)                  # root folder containing looped_T_*_N_* dirs
    print(f"root path : {ROOT}")
    OUT  = Path(f"tuner_convexification_results_wo_warmstarting_T_{T}_N_{N}_{part}.csv")
    run(ROOT, OUT)
    print(f"Done. Wrote {OUT.resolve()}")
   