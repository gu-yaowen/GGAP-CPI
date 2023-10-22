# run all the datasets in MoleculeACE with KANO model
# data_folder="data/MoleculeACE/"
# while IFS= read -r file; do
#     if [[ "$file" == *.csv ]]; then
#         data_path=$dataset$file
#         echo $data_path
#         python main.py --gpu 0 \
#                        --data_path $data_path \
#                        --dataset_type regression \
#                        --seed 0 \
#                        --features_scaling \
#                        --metric rmse \
#                        --epochs 100 \
#                        --split_sizes 0.8 0.0 0.2
#     fi
# done < <(find "$data_folder" -maxdepth 1 -type f)

# run all the datasets in MoleculeACE with QSAR baseline
data_folder="data/MoleculeACE/"
while IFS= read -r file; do
    if [[ "$file" == *.csv ]]; then
        data_path=$dataset$file
        echo $data_path
        # for model in "MLP" "SVM" "RF" "GBM" "KNN" "GAT" "GCN" "AFP" "MPNN" "Transformer" "LSTM"; do
        for model in "LSTM" "GCN" "GAT" "MPNN"; do
            python main.py --gpu 0 \
                           --mode baseline_QSAR \
                           --data_path $data_path \
                           --dataset_type regression \
                           --seed 0 \
                           --baseline_model $model
        done
    fi
done < <(find "$data_folder" -maxdepth 1 -type f)

# python main.py --gpu 0 \
#                 --mode baseline_CPI \
#                 --dataset_type regression \
#                 --data_path test \
#                 --seed 0 \
#                 --baseline_model DeepDTA
