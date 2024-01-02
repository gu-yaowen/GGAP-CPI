#!/bin/bash
# data_folder="data/MoleculeACE/"
# MOLECULEACE_DATALIST=(CHEMBL2147_Ki CHEMBL214_Ki
#                       CHEMBL228_Ki CHEMBL244_Ki
#                       CHEMBL2835_Ki CHEMBL4005_Ki
#                       CHEMBL4203_Ki CHEMBL4792_Ki)
# for model in MLP SVM RF GBM KNN; do
#     for file in "${MOLECULEACE_DATALIST[@]}"; do
#         data_path=$data_folder$file.csv
#         echo $data_path
#         python main.py --gpu 0 \
#                     --mode baseline_QSAR \
#                     --data_path $data_path \
#                     --dataset_type regression \
#                     --seed 0 \
#                     --baseline_model $model
#     done
# done

# MOLECULEACE_DATALIST=(CHEMBL214_Ki_Integrated CHEMBL2147_Ki_Integrated
#                       CHEMBL228_Ki_Integrated CHEMBL244_Ki_Integrated
#                       CHEMBL2835_Ki_Integrated CHEMBL4005_Ki_Integrated
#                       CHEMBL4203_Ki_Integrated CHEMBL4792_Ki_Integrated)
# data_folder="data/MoleculeACE/"                      
# model=AFP
# for file in "${MOLECULEACE_DATALIST[@]}"; do
#         data_path=$data_folder$file.csv
#         echo $data_path
#         python main.py --gpu 0 \
#                     --mode baseline_QSAR \
#                     --data_path $data_path \
#                     --dataset_type regression \
#                     --seed 0 \
#                     --baseline_model $model
# done

# for model in MLP SVM RF GBM KNN GAT GCN AFP MPNN CNN Transformer LSTM; do
#     for file in "${MOLECULEACE_DATALIST[@]}"; do
#         data_path=$data_folder$file.csv
#         echo $data_path
#         python main.py --gpu 0 \
#                     --mode baseline_QSAR \
#                     --data_path $data_path \
#                     --dataset_type regression \
#                     --seed 0 \
#                     --baseline_model $model
#     done
# done


data_path="data/Ours"
model="your_model"
for model in MLP SVM RF GBM KNN GAT GCN AFP MPNN CNN Transformer LSTM; do
for file in "$data_path"/*.csv; do
    if [[ "$(basename "$file")" != "CPI_Integrated.csv" ]]; then
        echo Running on $file
        python main.py --no_cuda \
                       --mode baseline_QSAR \
                       --data_path $file \
                       --dataset_type regression \
                       --seed 0 \
                       --baseline_model $model
    fi
done
done

singularity exec --nv --overlay chem.ext3:ro /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash
source /ext3/env.sh
conda activate chem_gyw
cd Activity-cliff-prediction
data_path="data/Ours"
model=AFP
for file in "$data_path"/*.csv; do
    if [[ "$(basename "$file")" != "CPI_Integrated.csv" ]]; then
        filename_without_ext="${file%.csv}"

        result_folder="exp_results/$(basename "$filename_without_ext")"
        pred_file="${result_folder}/0/${model}_test_pred.csv"

        if [[ ! -f "$pred_file" ]]; then
            echo "Running on $file because $pred_file does not exist."
            python main.py --gpu 0 \
                           --mode baseline_QSAR \
                           --data_path $file \
                           --dataset_type regression \
                           --seed 0 \
                           --baseline_model $model
        else
            echo "Skipping $file as $pred_file already exists."
        fi
    fi
done
