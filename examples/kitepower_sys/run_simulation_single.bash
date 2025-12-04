module purge; module load devel/python/3.11.7-gnu-14.2

cd examples/kitepower_sys
mkdir -p sim_logs
set -euo pipefail
source "../../venv/bin/activate"


for s in closed_loop_simulation_cluster_with_noise_added_in_z_config_1_20.py ;
# \
#           closed_loop_simulation_cluster_with_noise_added_4.py closed_loop_simulation_cluster_with_noise_added_5.py \
#           closed_loop_simulation_cluster_with_noise_added_6.py ;
do
  sbatch -J "${s%.py}" -p cpu -t 20:00:00 -c 1 --mem=6G -o sim_logs/%x_%j.out \
    --export=ALL \
    --wrap "
    set -euo pipefail;
    module purge; 
    module load devel/python/3.11.7-gnu-14.2; 
    source ../../venv/bin/activate;
    export=ALL,OMP_NUM_THREADS=1,MKL_NUM_THREADS=1,PYTHONUNBUFFERED=1;
    srun -c ${SLURM_CPUS_PER_TASK:-1} python -u $s"
done
