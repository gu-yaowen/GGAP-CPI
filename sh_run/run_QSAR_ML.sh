#!/bin/bash

# singularity exec --nv --overlay chem.ext3:ro /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash
# source /ext3/env.sh
# conda activate chem_gyw
# cd Activity-cliff-prediction

model="$1"
filename=sh_run/Ours_${model}_miss.txt
data_path=data/Ours/
n=1
while read file; do
    csv_file=$data_path$file
    echo Running on $csv_file
    python main.py --mode baseline_QSAR \
                   --data_path $csv_file \
                   --dataset_type regression \
                   --seed 0 \
                   --baseline_model $model \
                   --print \
                   --no_cuda
    n=$((n+1))
done < $filename
