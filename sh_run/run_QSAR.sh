model="$1"
filename="$2"
data_path="data/Ours/"

if [ "$model" != "KANO_Siams" ]; then
    n=1
    while read -r file; do
        csv_file="${data_path}${file}"
        echo "Running on $csv_file"
        python main.py --gpu 0 \
                    --mode baseline_QSAR \
                    --data_path "$csv_file" \
                    --dataset_type regression \
                    --seed 0 \
                    --baseline_model "$model" \
                    --print
        n=$((n+1))
    done < "$filename"
else
    n=1
    while read -r file; do
        csv_file="${data_path}${file}"
        echo "Running on $csv_file"
        python main.py --gpu 0 \
                    --mode train \
                    --data_path "$csv_file" \
                    --dataset_type regression \
                    --seed 0 \
                    --train_model KANO_Siams \
                    --loss_weights "1 0 0" \
                    --siams_num 3 \
                    --batch_size 256 \
                    --epoch 1 \
                    --print
        n=$((n+1))
    done < "$filename"
fi
