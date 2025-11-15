import pandas as pd
from pathlib import Path
import re

FOLDER = Path("F:/Thesis/Parameter_sweep_analysis/Analysis_results/data_folder_to_generate_graphs_for_param_analysis_chapter/Time_discretization/CSVs")        # <-- change to your folder
OUT_CSV = FOLDER / "time_discretization_master.csv" # output file

# Try multiple encodings; fall back to ignoring bad bytes
ENCODINGS = ("utf-8-sig", "cp1252", "latin1", "iso-8859-1", "utf-16-le", "utf-16-be")

def read_csv_any_delim(path: Path) -> pd.DataFrame:
    last_err = None
    for enc in ENCODINGS:
        try:
            df = pd.read_csv(
                path,
                sep=None, engine="python",  # auto-detect delimiter
                dtype=str,                  # keep everything as string to avoid type mismatches
                encoding=enc,
                encoding_errors="replace"
            )
            # normalize column names (strip BOM/NBSP/whitespace)
            df.columns = [
                re.sub(r"\s+", " ", str(c).replace("\ufeff", "").replace("\u00A0", " ")).strip()
                for c in df.columns
            ]
            return df
        except UnicodeDecodeError as e:
            last_err = e
            continue
        except Exception as e:
            # If parsing fails for other reasons, try next encoding
            last_err = e
            continue

    # last resort: read as bytes and ignore undecodable chars
    with open(path, "rb") as f:
        text = f.read().decode("utf-8", errors="ignore")
    from io import StringIO
    df = pd.read_csv(StringIO(text), sep=None, engine="python", dtype=str)
    df.columns = [re.sub(r"\s+", " ", str(c).replace("\ufeff", "").replace("\u00A0", " ")).strip()
                  for c in df.columns]
    return df

def main():
    csv_paths = sorted(p for p in FOLDER.glob("*.csv") if p.is_file())
    if not csv_paths:
        print(f"No CSV files found in {FOLDER.resolve()}")
        return

    frames = []
    for p in csv_paths:
        try:
            df = read_csv_any_delim(p)
            df.insert(0, "source_file", p.name)
            frames.append(df)
            print(f"Loaded: {p.name}  rows={len(df)}  cols={len(df.columns)}")
        except Exception as e:
            print(f"Skipping {p.name}: {e}")

    if not frames:
        print("No CSVs could be read.")
        return

    # Outer-concat to preserve all columns; align missing with NaN
    master = pd.concat(frames, axis=0, ignore_index=True, sort=False)

    # optional: sort columns alphabetically but keep 'source_file' first
    cols = master.columns.tolist()
    if "source_file" in cols:
        other = sorted([c for c in cols if c != "source_file"])
        master = master[["source_file"] + other]

    master.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"Saved master CSV: {OUT_CSV.resolve()}  rows={len(master)}  cols={len(master.columns)}")

if __name__ == "__main__":
    main()
