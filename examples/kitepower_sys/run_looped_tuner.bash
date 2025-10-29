module purge; module load devel/python/3.11.7-gnu-14.2

cd examples/kitepower_sys
mkdir -p logs
for s in looped_tuner_1.py looped_tuner_2.py looped_tuner_3.py looped_tuner_4.py looped_tuner_5.py \
         looped_tuner_6.py looped_tuner_7.py looped_tuner_8.py looped_tuner_9.py looped_tuner_10.py \
         looped_tuner_11.py looped_tuner_12.py looped_tuner_13.py looped_tuner_14.py looped_tuner_15.py \
         looped_tuner_16.py looped_tuner_17.py looped_tuner_18.py \
         looped_tuner_19.py looped_tuner_20.py looped_tuner_21.py looped_tuner_22.py \
         looped_tuner_23.py looped_tuner_24.py looped_tuner_25.py looped_tuner_26.py;
do
  sbatch -J "${s%.py}" -p cpu -t 40:00:00 -c 1 --mem=4G -o logs/%j.out \
    --wrap "
    set -euo pipefail;
    module purge; 
    module load devel/python/3.11.7-gnu-14.2; 
    source ../../venv/bin/activate;
    export OMP_NUM_THREADS=1;
    export MKL_NUM_THREADS=1;
    python $s"
done
