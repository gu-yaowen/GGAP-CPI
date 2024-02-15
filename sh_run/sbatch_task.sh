srun --cpus-per-task=2 --mem=32GB --gres=gpu:v100:1 --time=4:00:00 --pty /bin/bash
srun --cpus-per-task=2 --mem=32GB --time=3:00:00 --pty /bin/bash
sbatch --time=8:00:00 --cpus-per-task=4 --mem=32GB --gres=gpu:a100:1 --wrap "sleep infinity"

sbatch --time=12:00:00 --cpus-per-task=6 --mem=32GB --wrap "sleep infinity"

singularity exec --nv --overlay chem.ext3:ro /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash
source /ext3/env.sh
conda activate chem_gyw
cd Activity-cliff-prediction

MOLECULEACE_DATALIST=("CHEMBL2047_EC50"
                      "CHEMBL204_Ki" "CHEMBL2147_Ki" "CHEMBL214_Ki" "CHEMBL218_EC50"
                      "CHEMBL219_Ki" "CHEMBL228_Ki" "CHEMBL231_Ki" "CHEMBL233_Ki"
                      "CHEMBL234_Ki" "CHEMBL235_EC50" "CHEMBL236_Ki" "CHEMBL237_EC50"
                      "CHEMBL237_Ki" "CHEMBL238_Ki" "CHEMBL239_EC50" "CHEMBL244_Ki"
                      "CHEMBL262_Ki" "CHEMBL264_Ki" "CHEMBL2835_Ki" "CHEMBL287_Ki"
                      "CHEMBL2971_Ki" "CHEMBL3979_EC50" "CHEMBL4005_Ki" "CHEMBL4203_Ki"
                      "CHEMBL4616_EC50" "CHEMBL4792_Ki")
data_folder="data/MoleculeACE/"
file="CHEMBL2047_EC50"
data_path=$data_folder$file.csv
model=GCN
python main.py --gpu 0 \
                   --mode baseline_QSAR \
                   --data_path $data_path \
                   --dataset_type regression \
                   --seed 0 \
                   --baseline_model $model
for file in "${MOLECULEACE_DATALIST[@]}"; do
    data_path=$data_folder$file.csv
    echo $data_path
    model="GCN"
    python main.py --gpu 0 \
                   --mode baseline_QSAR \
                   --data_path $data_path \
                   --dataset_type regression \
                   --seed 0 \
                   --baseline_model $model
done
