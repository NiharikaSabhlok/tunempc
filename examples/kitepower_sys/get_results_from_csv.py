from pathlib import Path
import pandas as pd
import numpy as np
import re

TARGET_VALUES = {"EQUIVALENCE_TYPE_A", "EQUIVALENCE_TYPE_B", "EQUIVALENCE_TYPE_FORCED"}
input_dir = Path("F:/Thesis/Parameter_sweep_analysis/Analysis_results")
output_csv =  Path("F:/Thesis/Parameter_sweep_analysis/Analysis_results/filtered_equivalence_types.csv")
match_col = "equivalence_status"
keep_cols = ["T","N","beta","acc_reg"] 
def read_csv_any_delim(path: Path, dtype=str) -> pd.DataFrame:
    """Auto-detect delimiter, try a few encodings, and normalize column names."""
    encodings = ("utf-8-sig", "utf-8", "cp1252", "latin1", "iso-8859-1", "utf-16-le", "utf-16-be")
    last_err = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, sep=None, engine="python", dtype=dtype,
                             encoding=enc, encoding_errors="replace")
            df.columns = [re.sub(r"\s+", " ", str(c).replace("\ufeff","").replace("\u00A0"," ")).strip()
                          for c in df.columns]
            return df
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Failed to read {path}: {last_err}")



if not input_dir.is_dir():
    raise FileNotFoundError(f"Input dir not found: {input_dir}")

csv_paths = list(input_dir.rglob("*.csv")) + list(input_dir.rglob("*.csv.gz"))
print(csv_paths)
if not csv_paths:
    raise FileNotFoundError("No CSV files found.")

targets = TARGET_VALUES
out_parts = []

for p in csv_paths:
    print(f"Processing {p}...")
    try:
        df = read_csv_any_delim(p)
    except Exception as e:
        print(f"[WARN] Skipping {p}: {e}")
        continue

    if match_col not in df.columns:
        print(f"[WARN] '{match_col}' not in {p.name}; skipping.")
        continue

    col = df[match_col].astype(str)
    vals = col
    mask = np.isin(vals, list(targets))
    if not mask.any():
        continue

    df_f = df.loc[mask].copy()

    if keep_cols:
        # ensure match_col is included
        keep = list(dict.fromkeys(list(keep_cols) + [match_col]))
        missing = [c for c in keep if c not in df_f.columns]
        if missing:
            print(f"[WARN] {p.name} missing columns {missing}; filling with NaN.")
        df_f = df_f.reindex(columns=keep)

    df_f.insert(0, "_source_file", p.name)
    out_parts.append(df_f)

if not out_parts:
    raise RuntimeError("No matching rows found in any file.")

out_df = pd.concat(out_parts, ignore_index=True)
output_csv.parent.mkdir(parents=True, exist_ok=True)
out_df.to_csv(output_csv, index=False)
print(f"[OK] Wrote combined CSV: {output_csv} ({len(out_df)} rows).")

# if per_type:
#     present_vals = sorted(out_df[match_col].astype(str).unique().tolist())
#     for val in present_vals:
#         safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", val)
#         per_path = output_csv.with_name(f"{output_csv.stem}__{safe}{output_csv.suffix}")
#         out_df[out_df[match_col].astype(str) == val].to_csv(per_path, index=False)
#         print(f"[OK] Wrote {val}: {per_path}")

# -----------------------
# Example usage:
# run(
#     input_dir=r"/path/to/folder",
#     output_csv=r"/path/to/out/filtered.csv",
#     match_col="EquivalenceType",
#     keep_cols=["ID","Timestamp","EquivalenceType","ValueA","ValueB"],  # or None
#     case_insensitive=True,
#     per_type=True
# )

