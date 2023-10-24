MOLECULEACE_DATALIST=("CHEMBL218_EC50"
                      "CHEMBL219_Ki" "CHEMBL228_Ki" "CHEMBL231_Ki" "CHEMBL233_Ki"
                      "CHEMBL234_Ki" "CHEMBL235_EC50" "CHEMBL236_Ki" "CHEMBL237_EC50"
                      "CHEMBL237_Ki" "CHEMBL238_Ki" "CHEMBL239_EC50" "CHEMBL244_Ki"
                      "CHEMBL262_Ki" "CHEMBL264_Ki" "CHEMBL2835_Ki" "CHEMBL287_Ki"
                      "CHEMBL2971_Ki" "CHEMBL3979_EC50" "CHEMBL4005_Ki" "CHEMBL4203_Ki"
                      "CHEMBL4616_EC50" "CHEMBL4792_Ki")
data_folder="data/MoleculeACE/"
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
MOLECULEACE_DATALIST=("CHEMBL1862_Ki" "CHEMBL1871_Ki" "CHEMBL2034_Ki" "CHEMBL2047_EC50"
                      "CHEMBL204_Ki" "CHEMBL2147_Ki" "CHEMBL214_Ki" "CHEMBL218_EC50"
                      "CHEMBL219_Ki" "CHEMBL228_Ki" "CHEMBL231_Ki" "CHEMBL233_Ki"
                      "CHEMBL234_Ki" "CHEMBL235_EC50" "CHEMBL236_Ki" "CHEMBL237_EC50"
                      "CHEMBL237_Ki" "CHEMBL238_Ki" "CHEMBL239_EC50" "CHEMBL244_Ki"
                      "CHEMBL262_Ki" "CHEMBL264_Ki" "CHEMBL2835_Ki" "CHEMBL287_Ki"
                      "CHEMBL2971_Ki" "CHEMBL3979_EC50" "CHEMBL4005_Ki" "CHEMBL4203_Ki"
                      "CHEMBL4616_EC50" "CHEMBL4792_Ki")
for file in "${MOLECULEACE_DATALIST[@]}"; do
    data_path=$data_folder$file.csv
    echo $data_path
    model="GAT"
    python main.py --gpu 0 \
                   --mode baseline_QSAR \
                   --data_path $data_path \
                   --dataset_type regression \
                   --seed 0 \
                   --baseline_model $model
done
MOLECULEACE_DATALIST=("CHEMBL1862_Ki" "CHEMBL1871_Ki" "CHEMBL2034_Ki" "CHEMBL2047_EC50"
                      "CHEMBL204_Ki" "CHEMBL2147_Ki" "CHEMBL214_Ki" "CHEMBL218_EC50"
                      "CHEMBL219_Ki" "CHEMBL228_Ki" "CHEMBL231_Ki" "CHEMBL233_Ki"
                      "CHEMBL234_Ki" "CHEMBL235_EC50" "CHEMBL236_Ki" "CHEMBL237_EC50"
                      "CHEMBL237_Ki" "CHEMBL238_Ki" "CHEMBL239_EC50" "CHEMBL244_Ki"
                      "CHEMBL262_Ki" "CHEMBL264_Ki" "CHEMBL2835_Ki" "CHEMBL287_Ki"
                      "CHEMBL2971_Ki" "CHEMBL3979_EC50" "CHEMBL4005_Ki" "CHEMBL4203_Ki"
                      "CHEMBL4616_EC50" "CHEMBL4792_Ki")
for file in "${MOLECULEACE_DATALIST[@]}"; do
    data_path=$data_folder$file.csv
    echo $data_path
    model="MPNN"
    python main.py --gpu 0 \
                   --mode baseline_QSAR \
                   --data_path $data_path \
                   --dataset_type regression \
                   --seed 0 \
                   --baseline_model $model
done
MOLECULEACE_DATALIST=("CHEMBL1862_Ki" "CHEMBL1871_Ki" "CHEMBL2034_Ki" "CHEMBL2047_EC50"
                      "CHEMBL204_Ki" "CHEMBL2147_Ki" "CHEMBL214_Ki" "CHEMBL218_EC50"
                      "CHEMBL219_Ki" "CHEMBL228_Ki" "CHEMBL231_Ki" "CHEMBL233_Ki"
                      "CHEMBL234_Ki" "CHEMBL235_EC50" "CHEMBL236_Ki" "CHEMBL237_EC50"
                      "CHEMBL237_Ki" "CHEMBL238_Ki" "CHEMBL239_EC50" "CHEMBL244_Ki"
                      "CHEMBL262_Ki" "CHEMBL264_Ki" "CHEMBL2835_Ki" "CHEMBL287_Ki"
                      "CHEMBL2971_Ki" "CHEMBL3979_EC50" "CHEMBL4005_Ki" "CHEMBL4203_Ki"
                      "CHEMBL4616_EC50" "CHEMBL4792_Ki")
for file in "${MOLECULEACE_DATALIST[@]}"; do
    data_path=$data_folder$file.csv
    echo $data_path
    model="LSTM"
    python main.py --gpu 0 \
                   --mode baseline_QSAR \
                   --data_path $data_path \
                   --dataset_type regression \
                   --seed 0 \
                   --baseline_model $model
done