# Description: Run CPI experiments
model=$1
echo $model
python main.py --gpu 0 \
               --data_path data/MoleculeACE/CPI_plus_Integrated.csv \
                --mode baseline_CPI \
                --dataset_type regression \
                --seed 0 \
                --baseline_model $model
