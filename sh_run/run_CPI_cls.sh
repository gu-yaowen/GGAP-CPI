# Description: Run CPI experiments
# model in ['KANO_Prot', 'KANO_Siams', 'DeepDTA', 'GraphDTA', 'MolTrans']
model="$1"
# filename in ['CPI_ki', 'CPI_kd', 'CPI_ec50', 'CPI_ic50', 'MolACE_CPI_ki', 'MolACE_CPI_ec50']
# cls dataset in ['binder_nonbinder_fgfr_pt']
filename="$2"
# mode in ['train', 'inference', 'retrain', 'finetune', 'baseline_CPI']
mode="$3"
seed="$4"

model_path="$5"
seed2="$6"
ablation="$7"
echo "Running on $filename, model: $model, mode: $mode"

if [ "$mode" = "baseline_CPI" ]; then
    if [ "$model" = 'PerceiverCPI' ]; then
        cd CPI_baseline/PerceiverCPI
        savepath=../../exp_results/PerceiverCPI/$filename/$seed/
        python train.py --data_path ./dataset/"$filename"_train.csv \
                --separate_test_path ./dataset/"$filename"_test.csv \
                --metric mse --dataset_type classification \
                --save_dir $savepath --target_columns label \
                --epochs 80 --ensemble_size 1 --num_folds 1 \
                --batch_size 512 --aggregation mean --dropout 0.1 --save_preds
    else
        python main.py --gpu 0 \
                    --mode $mode \
                    --data_path data/${filename}.csv \
                    --dataset_type classification \
                    --seed $seed \
                    --baseline_model $model \
                    --metric auc \
                    --print
    fi
elif [ "$mode" = "baseline_inference" ]; then
    if [ "$model" = 'PerceiverCPI' ]; then
        cd CPI_baseline/PerceiverCPI
        savepath=../../exp_results/PerceiverCPI/$model_path/$seed/
        python predict.py --test_path ./dataset/"$filename".csv \
                    --checkpoint_dir $savepath \
                    --metric auc \
                    --preds_path "$filename"_infer.csv
    else
        python main.py --gpu 0 \
                        --mode $mode \
                        --data_path data/${filename}.csv \
                        --model_path exp_results/$model/$model_path/$seed \
                        --dataset_type classification \
                        --seed $seed \
                        --baseline_model $model \
                        --metric auc \
                        --print
    fi
elif [ "$mode" = "train" ]; then
    python main.py --gpu 0 \
                    --data_path data/${filename}.csv \
                    --mode $mode \
                    --dataset_type classification \
                    --seed $seed \
                    --train_model $model \
                    --loss_weights "1 0 0" \
                    --batch_size 64 \
                    --dropout 0.0 \
                    --ablation $ablation \
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
                        --batch_size 256 \
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
                        --ablation $ablation \
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
                    --model_path exp_results/$model/$model_path/$seed2 \
                    --loss_weights "1 0 0" \
                    --batch_size 256 \
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
                    --model_path exp_results/$model/$model_path/$seed2 \
                    --loss_weights "1 0 0" \
                    --batch_size 64 \
                    --dropout 0.0 \
                    --print
else
    echo "Invalid mode: $mode"
fi
