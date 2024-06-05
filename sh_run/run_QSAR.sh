model="$1"
data_path="$2"
seed="$3"
n=1

for file in data/${data_path}/*; do
    echo "Running on $file"
    python main.py --gpu 0 \
                --mode baseline_QSAR \
                --data_path "$file" \
                --dataset_type regression \
                --seed "$seed" \
                --baseline_model "$model" \
                --print
done
