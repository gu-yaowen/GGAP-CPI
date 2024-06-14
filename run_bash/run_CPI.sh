# Description: Run CPI training, finetuning, and virtual screening testing

# model in ['KANO_Prot', 'KANO_ESM', 'ECFP_ESM_RF', 'ECFP_ESM_GBM',
#           'DeepDTA', 'GraphDTA', 'HyperAttentionDTI', 'PerceiverCPI']
model="$1"

# filename in ['ki', 'kd', 'ec50', 'ic50', 'integrated'] -> for CPI2M-main, 
#             feasible for mode in ['train', 'retrain', 'finetune', 'baseline_CPI']

#             ['ki_last', 'kd_last', 'ec50_last', 'ic50_last'] -> for CPI2M-few,
#             feasible for mode in ['inference']

#             ['MolACE_CPI_ki', 'MolACE_CPI_ec50'] -> for MoleculeACE,
#             feasible for mode in ['train', 'retrain', 'finetune', 'baseline_CPI']

#             ['PDBbind_CASF', 'PDBbind_all'] -> for CASF-2016,
#             feasible for mode in ['inference', 'finetune']

#             {LIT-PCBA ID}_LITPCBA.csv -> for LIT-PCBA,
#             feasible for mode in ['inference']
filename="$2"

# mode in ['train', 'inference', 'retrain', 'finetune', 'baseline_CPI'], 
# finetune for activity-based transfer learning
# retrain for some devices with limited running time, then use "retrain" 
# to restart training from the previous stopped epoches.
mode="$3"

# random seed setting
seed="$4"

# pretrained model path for inference, retrain, and finetune.
# e.g., 'ki/2' to load pretrained model saved in exp_results/KANO_Prot/ki/2
model_path="$5"

# ablation setting for ablation study, in ['none', 'KANO', 'ESM', 'GCN', 'Attn']
# ablation="$6"
ablation="none"

echo "Running on $filename, model: $model, mode: $mode"

if [ "$mode" = "baseline_CPI" ]; then
    if [ "$model" = 'PerceiverCPI' ]; then
        cd CPI_baseline/PerceiverCPI
        savepath=../../exp_results/PerceiverCPI/$filename/$seed/
        python train.py --data_path ./dataset/"$filename"_train.csv \
                --separate_test_path ./dataset/"$filename"_test.csv \
                --metric mse --dataset_type regression \
                --save_dir $savepath --target_columns label \
                --epochs 80 --ensemble_size 1 --num_folds 1 \
                --batch_size 256 --aggregation mean --dropout 0.1 --save_preds
    else
        python main.py --gpu 0 \
                    --mode $mode \
                    --data_path data/${filename}.csv \
                    --dataset_type regression \
                    --seed $seed \
                    --baseline_model $model \
                    --print
    fi
elif [ "$mode" = "baseline_inference" ]; then
    if [ "$model" = 'PerceiverCPI' ]; then
        cd CPI_baseline/PerceiverCPI
        savepath=../../exp_results/PerceiverCPI/$model_path/$seed/
        python predict.py --test_path ./dataset/"$filename".csv \
                    --checkpoint_dir $savepath \
                    --preds_path "$filename"_infer.csv
    else
        python main.py --gpu 0 \
                        --mode $mode \
                        --data_path data/${filename}.csv \
                        --model_path exp_results/$model/$model_path \
                        --dataset_type regression \
                        --seed $seed \
                        --baseline_model $model \
                        --print
    fi
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
                    --ablation $ablation \
                    --print
elif [ "$mode" = "retrain" ]; then
    if [ "$ablation" = "none" ]; then
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
    else
        python main.py --gpu 0 \
                        --data_path data/${filename}.csv \
                        --mode $mode \
                        --dataset_type regression \
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
                    --dataset_type regression \
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
                    --dataset_type regression \
                    --seed $seed \
                    --train_model $model \
                    --model_path exp_results/"$model"/$model_path \
                    --loss_weights "1 0 0" \
                    --batch_size 64 \
                    --dropout 0.0 \
                    --ablation $ablation \
                    --print
else
    echo "Invalid mode: $mode"
fi
