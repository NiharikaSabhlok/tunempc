module purge; module load devel/python/3.11.7-gnu-14.2

cd examples/kitepower_sys
mkdir -p param_sweep_logs
set -euo pipefail
source "../../venv/bin/activate"

# cluster_looped_parameter_sweep_analysis_beta_acc_reg.py

for s in cluster_looped_parameter_sweep_analysis_beta_acc_reg_T_20.py cluster_looped_parameter_sweep_analysis_beta_acc_reg_T_21.py \
         cluster_looped_parameter_sweep_analysis_beta_acc_reg_T_22.py cluster_looped_parameter_sweep_analysis_beta_acc_reg_T_23.py \
         cluster_looped_parameter_sweep_analysis_beta_acc_reg_T_24.py cluster_looped_parameter_sweep_analysis_beta_acc_reg_T_25.py \
         cluster_looped_parameter_sweep_analysis_beta_acc_reg_T_26.py cluster_looped_parameter_sweep_analysis_beta_acc_reg_T_27.py \
         cluster_looped_parameter_sweep_analysis_beta_acc_reg_T_28.py cluster_looped_parameter_sweep_analysis_beta_acc_reg_T_29.py \
         cluster_looped_parameter_sweep_analysis_beta_acc_reg_T_30.py cluster_looped_parameter_sweep_analysis_beta_acc_reg_T_31.py \
         cluster_looped_parameter_sweep_analysis_beta_acc_reg_T_32.py cluster_looped_parameter_sweep_analysis_beta_acc_reg_T_33.py \
         cluster_looped_parameter_sweep_analysis_beta_acc_reg_T_34.py cluster_looped_parameter_sweep_analysis_beta_acc_reg_T_35.py \
         cluster_looped_parameter_sweep_analysis_beta_acc_reg_T_36.py cluster_looped_parameter_sweep_analysis_beta_acc_reg_T_37.py \
         cluster_looped_parameter_sweep_analysis_beta_acc_reg_T_38.py cluster_looped_parameter_sweep_analysis_beta_acc_reg_T_39.py \
         cluster_looped_parameter_sweep_analysis_beta_acc_reg_T_40.py ;
do
  sbatch -J "${s%.py}" -p cpu -t 20:00:00 -c 1 --mem=10G -o param_sweep_logs/%x_%j.out \
    --export=ALL \
    --wrap "
    set -euo pipefail;
    module purge; 
    module load devel/python/3.11.7-gnu-14.2; 
    source ../../venv/bin/activate;
    export=ALL,OMP_NUM_THREADS=1,MKL_NUM_THREADS=1,PYTHONUNBUFFERED=1;
    srun -c ${SLURM_CPUS_PER_TASK:-1} python -u $s"
done
