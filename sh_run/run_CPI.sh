# Description: Run CPI experiments
# model in ['KANO_Prot', 'KANO_Siams', 'DeepDTA', 'GraphDTA', 'MolTrans']https://translate.google.cn/?sl=auto&tl=en&op=translate
model="$1"
# filename in ['CPI_ki', 'CPI_kd', 'CPI_ec50', 'CPI_ic50', 'MolACE_CPI_ki', 'MolACE_CPI_ec50']
filename="$2"
# mode in ['train', 'inference', 'retrain', 'finetune', 'baseline_CPI']
mode="$3"

seed="$4"

echo "Running on $filename, model: $model, mode: $mode"

if [ "$mode" = "baseline_CPI" ]; then
    python main.py --gpu 0 \
                   --mode $mode \
                   --data_path data/${filename}.csv \
                   --dataset_type regression \
                   --seed $seed \
                   --baseline_model $model
elif [ "$mode" = "train" ]; then
    python main.py --gpu 0 \
                    --data_path data/${filename}.csv \
                    --mode $mode \
                    --dataset_type regression \
                    --seed $seed \
                    --train_model $model \
                    --loss_weights "1 0 0" \
                    --batch_size 64 \
                    --dropout 0.0 \
                    --print
elif [ "$mode" = "retrain" ]; then
    python main.py --gpu 0 \
                    --data_path data/${filename}.csv \
                    --mode $mode \
                    --dataset_type regression \
                    --seed $seed \
                    --train_model $model \
                    --model_path exp_results/$model/$filename/$seed \
                    --loss_weights "1 0 0" \
                    --batch_size 64 \
                    --dropout 0.0 \
                    --print
elif [ "$mode" = "finetune" ]; then
    model_path="$5"
    python main.py --gpu 0 \
                    --data_path data/${filename}.csv \
                    --mode $mode \
                    --dataset_type regression \
                    --seed $seed \
                    --train_model $model \
                    --model_path exp_results/$model/$model_path/2 \
                    --loss_weights "1 0 0" \
                    --batch_size 64 \
                    --dropout 0.0 \
                    --print
else
    echo "Invalid mode: $mode"
fi
