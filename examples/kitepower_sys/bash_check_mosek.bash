#!/usr/bin/env bash
#SBATCH -J tunempc_mosek
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=16G

module purge; module load devel/python/3.11.7-gnu-14.2

cd examples/kitepower_sys

set -euo pipefail

# --- (A) Load modules (adapt to your cluster) ---
# module purge
# module load python/3.10
# module load mosek/10.2        # if your cluster provides a MOSEK module

# --- (B) Activate your venv (or use module-provided Python) ---
source "../../venv/bin/activate"

# --- (C) Point to the MOSEK license ---
# If you have a license FILE:
# export MOSEKLM_LICENSE_FILE="$HOME/mosek/mosek.lic"
# Or if you use a LICENSE SERVER:
# export MOSEKLM_LICENSE_FILE="27000@license.myuni.edu"

# (Optional) If your cluster doesn’t add MOSEK’s libs to the runtime path:
# export MOSEK_HOME="$HOME/mosek/mosek"   # adjust if needed
# export LD_LIBRARY_PATH="${MOSEK_HOME}/tools/platform/linux64x86/bin:${LD_LIBRARY_PATH:-}"

# --- (D) Prove MOSEK is importable & licensed (prints version) ---
python - <<'PY'
import sys, logging
logging.basicConfig(level=logging.INFO)
try:
    import mosek
    with mosek.Env() as env:
        ver = env.getversion()
    if len(ver) == 3:
        major, minor, rev = ver
        print(f"[CHECK] MOSEK import OK. Version: {major}.{minor}.{rev}")
    else:
        major, minor, rev, build = ver
        print(f"[CHECK] MOSEK import OK. Version: {major}.{minor}.{rev} build {build}")
except Exception as e:
    print(f"[CHECK] MOSEK NOT OK: {e}", file=sys.stderr)
    sys.exit(2)
PY

# --- (E) Run your job with explicit solver='mosek' and verbose logging ---
# Tip: ensure your Python code logs the solver and redirects solver stdout/stderr into your logger
# (use the solveSDP wrapper & log_mosek_info() from my previous message).
# LOGDIR="logs"
# mkdir -p "$LOGDIR"
# python -u run_tuner.py --solver mosek --verbose 2>&1 | tee -a "${LOGDIR}/tunempc_${SLURM_JOB_ID}.log"
for s in looped_tuner_test_mosek_2.py 
        #  looped_tuner_2.py looped_tuner_3.py looped_tuner_4.py looped_tuner_5.py \
        #  looped_tuner_6.py looped_tuner_7.py looped_tuner_8.py looped_tuner_9.py looped_tuner_10.py \
        #  looped_tuner_11.py looped_tuner_12.py looped_tuner_13.py looped_tuner_14.py looped_tuner_15.py \
        #  looped_tuner_16.py looped_tuner_17.py looped_tuner_18.py \
        #  looped_tuner_19.py looped_tuner_20.py looped_tuner_21.py looped_tuner_22.py \
        #  looped_tuner_23.py looped_tuner_24.py looped_tuner_25.py looped_tuner_26.py;
do
  sbatch -J "${s%.py}" -p cpu -t 40:00:00 -c 1 --mem=20G -o logs/%x_%j_mosek.out \
    --export=ALL \
    --wrap "
    set -euo pipefail;
    module purge; 
    module load devel/python/3.11.7-gnu-14.2; 
    source ../../venv/bin/activate;
    export OMP_NUM_THREADS=1;
    export MKL_NUM_THREADS=1;

    python - <<'PY'
import sys, logging
logging.basicConfig(level=logging.INFO)
try:
    import mosek
    with mosek.Env() as env:
        ver = env.getversion()
    print('[CHECK-CHILD] MOSEK import OK. Version:', '.'.join(map(str, ver)))
except Exception as e:
    print('[CHECK-CHILD] MOSEK NOT OK:', e, file=sys.stderr); sys.exit(2)
PY

      # Run your script; ensure it forces solver='mosek' and prints it
      # If your script accepts an arg, pass it; otherwise make sure inside it you call solver='mosek'.
      python $s"
done