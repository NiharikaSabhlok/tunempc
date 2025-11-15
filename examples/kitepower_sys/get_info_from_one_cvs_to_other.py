import pandas as pd
from pathlib import Path

# --- paths (change if needed)
ROOT = Path("F:/Thesis/Parameter_sweep_analysis/Analysis_results/files")
EQUIV_CSV = ROOT / "filtered_equivalence_types.csv"
MASTER_CSV = ROOT / "master_prep_input.csv"
OUT_CSV = ROOT / "master_filtered.csv"


# --- read
eq = pd.read_csv(EQUIV_CSV)
ms = pd.read_csv(MASTER_CSV)

# --- make sure keys exist and types match
keys = ["T", "N", "beta", "acc_reg" ]
for c in keys:
    if c not in eq.columns or c not in ms.columns:
        raise ValueError(f"Missing key column '{c}' in one of the files.")
    # coerce to numeric for reliable matching
    eq[c] = pd.to_numeric(eq[c], errors="coerce")
    ms[c] = pd.to_numeric(ms[c], errors="coerce")

# --- keep only needed columns from master
need_from_master = ["avg_power", "twall", "final_time"]
missing = [c for c in need_from_master if c not in ms.columns]
if missing:
    raise ValueError(f"master file missing columns: {missing}")

ms_small = ms[keys + need_from_master]

# --- merge (row-by-row equivalently)
out = eq.merge(ms_small, on=keys, how="left")

# --- save
out.to_csv(OUT_CSV, index=False)
print(f"Saved: {OUT_CSV.resolve()}  (rows: {len(out)})")
