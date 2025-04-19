# Description: Run target-specific model for CPI experiments

# model in ['SVM', 'RF< 'GBM', 'KNN', 'MLP', 
#           'GCN', 'GAT', 'MPNN', 'AFP', 'KANO',
#           'CNN', 'Transformer']
model="$1"

# filename in ['ki', 'kd', 'ec50', 'ic50'] -> for CPI2M-main, 

#             ['MolACE_CPI_ki', 'MolACE_CPI_ec50'] -> for MoleculeACE,
#             feasible for mode in ['train', 'retrain', 'finetune', 'baseline_CPI']
data_path="$2"

# random seed setting
seed="$3"

echo "Running on $data_path"
python main.py --gpu 0 \
            --mode baseline_QSAR \
            --data_path data/"$data_path".csv \
            --dataset_type regression \
            --seed "$seed" \
            --baseline_model "$model" \
            --print
