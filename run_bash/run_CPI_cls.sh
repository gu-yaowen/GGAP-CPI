# Description: Run classification tasks, such as binder/nonbinder classification, drug-target interaction prediction.

# model in ['KANO_Prot', 'KANO_ESM', 'ECFP_ESM_RF', 'ECFP_ESM_GBM',
#           'DeepDTA', 'GraphDTA', 'HyperAttentionDTI', 'PerceiverCPI']
model="$1"


# filename for classification data, column 'y' should be 0 or 1.
filename="$2"

# mode in ['train', 'inference', 'retrain', 'finetune'], 
# finetune for activity-based transfer learning
# retrain for some devices with limited running time, then use "retrain" 
# to restart training from the previous stopped epoches.
mode="$3"

# random seed setting
seed="$4"

# pretrained model path for inference and finetune.
model_path="$5"

echo "Running on $filename, model: $model, mode: $mode"

if [ "$mode" = "train" ]; then
    python main.py --gpu 0 \
                    --data_path data/${filename}.csv \
                    --mode $mode \
                    --dataset_type classification \
                    --seed $seed \
                    --train_model $model \
                    --loss_weights "1 0 0" \
                    --batch_size 64 \
                    --dropout 0.0 \
                    --metric auc \
                    --print
elif [ "$mode" = "retrain" ]; then
    if [ "$ablation" = "none" ]; then
        python main.py --gpu 0 \
                        --data_path data/${filename}.csv \
                        --mode $mode \
                        --dataset_type classification \
                        --metric auc \
                        --seed $seed \
                        --train_model $model \
                        --model_path exp_results/$model/$filename/$seed \
                        --loss_weights "1 0 0" \
                        --batch_size 64 \
                        --dropout 0.0 \
                        --print
    else
        python main.py --gpu 0 \
                        --data_path data/${filename}.csv \
                        --mode $mode \
                        --dataset_type classification \
                        --metric auc \
                        --seed $seed \
                        --train_model $model \
                        --model_path exp_results/"$model"_"$ablation"/$filename/$seed \
                        --loss_weights "1 0 0" \
                        --batch_size 64 \
                        --dropout 0.0 \
                        --print
    fi
elif [ "$mode" = "finetune" ]; then
    python main.py --gpu 0 \
                    --data_path data/${filename}.csv \
                    --mode $mode \
                    --dataset_type classification \
                    --metric auc \
                    --seed $seed \
                    --train_model $model \
                    --model_path exp_results/$model/$model_path \
                    --loss_weights "1 0 0" \
                    --batch_size 64 \
                    --dropout 0.0 \
                    --print
elif [ "$mode" = "inference" ]; then
    python main.py --gpu 0 \
                    --data_path data/${filename}.csv \
                    --ref_path data/${model_path}.csv \
                    --mode $mode \
                    --dataset_type classification \
                    --metric auc \
                    --seed $seed \
                    --train_model $model \
                    --model_path exp_results/$model/$model_path \
                    --loss_weights "1 0 0" \
                    --batch_size 64 \
                    --dropout 0.0 \
                    --print
else
    echo "Invalid mode: $mode"
fi
