module purge; module load devel/python/3.11.7-gnu-14.2

cd examples/kitepower_sys
# mkdir -p param_sweep_logs
mkdir -p prep_input_logs
set -euo pipefail
source "../../venv/bin/activate"

# cluster_looped_parameter_sweep_analysis_beta_acc_reg.py

for s in prepare_input_cluster_1.py prepare_input_cluster_2.py prepare_input_cluster_3.py prepare_input_cluster_4.py prepare_input_cluster_5.py  \
         prepare_input_cluster_6.py prepare_input_cluster_7.py prepare_input_cluster_8.py prepare_input_cluster_9.py prepare_input_cluster_10.py  \
         prepare_input_cluster_11.py prepare_input_cluster_12.py prepare_input_cluster_13.py prepare_input_cluster_14.py prepare_input_cluster_15.py \
         prepare_input_cluster_16.py prepare_input_cluster_17.py prepare_input_cluster_18.py prepare_input_cluster_19.py prepare_input_cluster_20.py ;
          # cluster_looped_parameter_sweep_analysis_Time_discretization_3.py cluster_looped_parameter_sweep_analysis_Time_discretization_4.py \
          # cluster_looped_parameter_sweep_analysis_Time_discretization_5.py cluster_looped_parameter_sweep_analysis_Time_discretization_6.py;
do
  # sbatch -J "${s%.py}" -p cpu -t 10:00:00 -c 1 --mem=20G -o param_sweep_logs/%x_%j.out \
  sbatch -J "${s%.py}" -p cpu -t 1:00:00 -c 1 --mem=10G -o prep_input_logs/%x_%j.out \
    --export=ALL \
    --wrap "
    set -euo pipefail;
    module purge; 
    module load devel/python/3.11.7-gnu-14.2; 
    source ../../venv/bin/activate;
    export=ALL,OMP_NUM_THREADS=1,MKL_NUM_THREADS=1,PYTHONUNBUFFERED=1;
    srun -c ${SLURM_CPUS_PER_TASK:-1} python -u $s"
done
