singularity exec --nv --overlay chem_new.ext3:ro /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash
source /ext3/env.sh
conda activate bind
cd BIND
conda activate chem_gyw
cd Activity-cliff-prediction
srun --jobid=47498814 --pty /bin/bash


for data in ki kd ec50; do
sbatch run_CPI.sbatch KANO_ESM $data train 2 kd 2 none
done

sbatch run_CPI.sbatch DeepDTA kd baseline_CPI 2 kd 2 none

sbatch run_CPI.sbatch GraphDTA kd baseline_CPI 2 kd 2 none

sbatch run_CPI.sbatch KANO_Prot ic50 retrain 3 kd 2 none

# QSAR
sh sh_run/run_QSAR.sh KANO kd 3
for data in ki kd ec50; do
sbatch run_QSAR.sbatch KANO $data 3
done

data=ic50
sh sh_run/run_QSAR.sh Transformer $data 2
sbatch run_QSAR.sbatch Transformer ic50 2
data=ic50
sh sh_run/run_QSAR.sh CNN $data 2

for model in MLP SVM RF GBM KNN; do
for data in ki kd ec50; do
sbatch run_QSAR.sbatch $model $data 2
done
done

for model in MLP SVM RF GBM KNN; do
sbatch run_QSAR.sbatch $model ic50 2
done

for model in GAT GCN AFP MPNN CNN Transformer LSTM; do
sbatch run_QSAR.sbatch $model kd 2
done

for model in GAT GCN AFP MPNN; do
sh sh_run/run_QSAR.sh $model ic50 2
done

for model in AFP CNN Transformer LSTM; do
sh sh_run/run_QSAR.sh $model ki 2
done

model=KANO
sh sh_run/run_QSAR.sh $model ki 2

for ab in KANO GCN ESM Attn; do
sbatch run_CPI.sbatch KANO_Prot ki train 3 kd 2 $ab
done

ab=KANO
sbatch run_CPI.sbatch KANO_Prot kd train 3 kd 2 $ab

ab=ESM
sh sh_run/run_CPI.sh KANO_Prot ki retrain 3 kd 3 $ab

for model in KANO_ESM_RF KANO_ESM_GBM; do
for data in MolACE_CPI_ec50 MolACE_CPI_ki; do
sh sh_run/run_CPI.sh $model $data baseline_CPI 2 $data 2 none
done
done


# train
sbatch run_CPI.sbatch KANO_Prot ic50 retrain 2 ic50 2 none
sh sh_run/run_CPI.sh KANO_Prot ic50 retrain 2 ic50 2 none
sbatch run_CPI.sbatch KANO_Prot integrated train 2 integrated 2 none
sh sh_run/run_CPI.sh KANO_Prot integrated retrain 2 integrated 2 none

singularity exec --nv --overlay chem_new.ext3:ro /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash
source /ext3/env.sh
conda activate chem_gyw
cd Activity-cliff-prediction

for dataset in qi_fgfr_time_v2 qi_fgfr_scaffold_v2 qi_CPI2M_fgfr_time_v2 qi_CPI2M_fgfr_scaffold_v2; do 
dataset=qi_CPI2M_fgfr_time_v2
# sh sh_run/run_CPI_cls.sh KANO_Prot $dataset train 0 $dataset 0 none
# sh sh_run/run_CPI_cls.sh KANO_Prot $dataset retrain 0 $dataset 0 none
sh sh_run/run_CPI_cls.sh KANO_Prot $dataset retrain 1 $dataset 1 none
sh sh_run/run_CPI_cls.sh KANO_Prot $dataset finetune 1 binder_nonbinder_fgfr_pt 2 none
done
watch nvidia-smi

# finetune
sh sh_run/run_CPI.sh KANO_Prot MolACE_CPI_ki finetune 6 ic50 2 none
sh sh_run/run_CPI.sh KANO_Prot MolACE_CPI_ec50 finetune 6 ic50 2 none
sbatch run_CPI.sbatch KANO_Prot MolACE_CPI_ki finetune 6 ic50 2 none
sbatch run_CPI.sbatch KANO_Prot MolACE_CPI_ec50 finetune 6 ic50 2 none

sbatch run_CPI.sbatch KANO_Prot ki_ft finetune 6 ic50 2 none
sbatch run_CPI.sbatch KANO_Prot ec50_ft finetune 6 ic50 2 none
sh sh_run/run_CPI.sh KANO_Prot ec50_ft retrain 6 ec50_ft 6 none
sh sh_run/run_CPI.sh KANO_Prot ec50 finetune 6 ic50 2 none
sh sh_run/run_CPI.sh KANO_Prot ki retrain 6 ki 6 none
sh sh_run/run_CPI.sh KANO_Prot ec50 retrain 6 ec50 6 none
sh sh_run/run_CPI.sh KANO_Prot MolACE_CPI_ec50 finetune 2 ec50_ft 6 none

cd Activity-cliff-prediction/sh_run

data=PDBbind_KANO_CASF
sh sh_run/run_CPI.sh KANO_Prot $data finetune 6 ic50 2 none
sh sh_run/run_CPI.sh KANO_Prot $data finetune 2 ki 2 none
data=PDBbind_KANO_all
sh sh_run/run_CPI.sh KANO_Prot $data finetune 6 ic50 2 none
sh sh_run/run_CPI.sh KANO_Prot $data finetune 6 ki 2 none
sbatch run_CPI.sbatch KANO_Prot $data finetune 6 ic50 2 none
sbatch run_CPI.sbatch KANO_Prot $data finetune 3 ki 2 none
sbatch run_CPI.sbatch KANO_Prot $data train 3 ki 2 none

# inference
sh sh_run/run_CPI.sh KANO_Prot ki_last inference 2 ki 2 none
sh sh_run/run_CPI.sh KANO_Prot ec50_last inference 2 ec50 2 none
sh sh_run/run_CPI.sh KANO_Prot kd_last inference 2 kd 2 none
sh sh_run/run_CPI.sh KANO_Prot ic50_last inference 2 ic50 2 none

sh sh_run/run_CPI.sh KANO_Prot PDBbind_refine_all inference 2 ic50 2 none
sh sh_run/run_CPI.sh KANO_Prot PDBbind_refine_all inference 2 ki 2 none
sh sh_run/run_CPI.sh KANO_Prot PDBbind_refine_all inference 2 kd 2 none

sh sh_run/run_CPI.sh KANO_Prot PDBbind_all inference 2 ic50 2 none
sh sh_run/run_CPI.sh KANO_Prot PDBbind_all inference 2 ki 2 none
sh sh_run/run_CPI.sh KANO_Prot PDBbind_all inference 2 kd 2 none

data=LITPCBA
sh sh_run/run_CPI.sh KANO_Prot $data inference 2 ic50 2 none
sh sh_run/run_CPI.sh KANO_Prot $data inference 2 ki 2 none
sh sh_run/run_CPI.sh KANO_Prot $data inference 2 kd 2 none

for ep in ic50 ki ec50; do
sh sh_run/run_CPI.sh KANO_Prot MolACE_CPI_ki_alltest inference 2 $ep 2 none
sh sh_run/run_CPI.sh KANO_Prot MolACE_CPI_ec50_alltest inference 2 $ep 2 none
done

sh sh_run/run_CPI.sh KANO_ESM ki_last inference 2 ki 2 none
sh sh_run/run_CPI.sh KANO_ESM ec50_last inference 2 ec50 2 none
sh sh_run/run_CPI.sh KANO_ESM kd_last inference 2 kd 2 none
sh sh_run/run_CPI.sh KANO_ESM ic50_last inference 2 ic50 2 none

# baseline_inference
sh sh_run/run_CPI.sh DeepDTA ki_last baseline_inference 2 ki 2 none
sh sh_run/run_CPI.sh DeepDTA ec50_last baseline_inference 2 ec50 2 none
sh sh_run/run_CPI.sh DeepDTA kd_last baseline_inference 2 kd 2 none
sh sh_run/run_CPI.sh DeepDTA ic50_last baseline_inference 2 ic50 2 none

sh sh_run/run_CPI.sh GraphDTA ki_last baseline_inference 2 ki 2 none
sh sh_run/run_CPI.sh GraphDTA ec50_last baseline_inference 2 ec50 2 none
sh sh_run/run_CPI.sh GraphDTA kd_last baseline_inference 2 kd 2 none
sh sh_run/run_CPI.sh GraphDTA ic50_last baseline_inference 2 ic50 2 none

sh sh_run/run_CPI.sh HyperAttentionDTI ki_last baseline_inference 2 ki 2 none
sh sh_run/run_CPI.sh HyperAttentionDTI ec50_last baseline_inference 2 ec50 2 none
sh sh_run/run_CPI.sh HyperAttentionDTI kd_last baseline_inference 2 kd 2 none
sh sh_run/run_CPI.sh HyperAttentionDTI ic50_last baseline_inference 2 ic50 2 none

data=LITPCBA
for model in ECFP_ESM_GBM ECFP_ESM_RF; do
sh sh_run/run_CPI.sh $model $data baseline_inference 2 ki 2 none
sh sh_run/run_CPI.sh $model $data baseline_inference 2 ic50 2 none
done

data=PDBbind_seq_alltest_screen
# model=HyperAttentionDTI
sh sh_run/run_CPI.sh $model $data baseline_inference 2 ki 2 none
sh sh_run/run_CPI.sh $model $data baseline_inference 2 ic50 2 none
data=LITPCBA
model=GraphDTA
sh sh_run/run_CPI.sh $model $data baseline_inference 2 ki 2 none
sh sh_run/run_CPI.sh $model $data baseline_inference 2 ic50 2 none

for model in DeepDTA GraphDTA HyperAttentionDTI; do
sh sh_run/run_CPI.sh $model $data baseline_inference 2 ki 2 none
sh sh_run/run_CPI.sh $model $data baseline_inference 2 ic50 2 none
done

data=LITPCBA_ki
sh sh_run/run_CPI.sh PerceiverCPI $data baseline_inference 2 ki 2 none
data=LITPCBA_ic50
sh sh_run/run_CPI.sh PerceiverCPI $data baseline_inference 2 ic50 2 none


for model in ECFP_ESM_GBM ECFP_ESM_RF KANO_ESM_GBM KANO_ESM_RF DeepDTA GraphDTA HyperAttentionDTI; do
sh sh_run/run_CPI.sh $model $data baseline_inference 2 ki 2 none
sh sh_run/run_CPI.sh $model $data baseline_inference 2 ic50 2 none
sh sh_run/run_CPI.sh $model $data baseline_inference 2 kd 2 none
done

sh sh_run/run_CPI.sh $model MolACE_CPI_ec50_alltest baseline_inference 2 ec50 2 none

sh sh_run/run_CPI.sh KANO_Prot MolACE_CPI_ki_alltest inference 2 ic50 2 none
sh sh_run/run_CPI.sh KANO_Prot MolACE_CPI_ec50_alltest inference 2 ic50 2 none

for model in KANO_ESM_RF KANO_ESM_GBM; do
for data in ic50; do
sh sh_run/run_CPI.sh $model "$data"_last baseline_inference 2 $data 2 none
done
done
KANO_ESM_GBM KANO_ESM_RF

for model in ECFP_ESM_GBM ECFP_ESM_RF KANO_ESM_GBM KANO_ESM_RF; do
for data in ic50; do
sh sh_run/run_CPI.sh $model "$data"_last baseline_inference 2 $data 2 none
done
done

sh sh_run/run_CPI.sh PerceiverCPI ki_test baseline_inference 2 ki 2 none
sh sh_run/run_CPI.sh PerceiverCPI ec50_test baseline_inference 2 ec50 2 none
sh sh_run/run_CPI.sh PerceiverCPI kd_test baseline_inference 2 kd 2 none
sh sh_run/run_CPI.sh PerceiverCPI ic50_last   2 ic50 2 none

# run ablation
sh sh_run/run_CPI.sh KANO_Prot ic50 retrain 2 ic50 2 KANO
sh sh_run/run_CPI.sh KANO_Prot ic50 retrain 2 ic50 2 GCN
sh sh_run/run_CPI.sh KANO_Prot ic50 retrain 2 ic50 2 GCN
sh sh_run/run_CPI.sh KANO_Prot ic50 retrain 2 ic50 2 Attn
sbatch run_CPI.sbatch KANO_Prot ic50 retrain 2 ic50 2 GCN
sh sh_run/run_CPI.sh KANO_Prot kd retrain 2 kd 2 GCN
# run KANO_ESM_RF and KANO_ESM_GBM
for model in KANO_ESM_RF KANO_ESM_GBM; do
sh sh_run/run_CPI.sh $model ic50 baseline_CPI 2 ic50 2 none
done
sh sh_run/run_CPI.sh KANO_ESM_RF ic50 baseline_CPI 2 ic50 2 none

# run ECFP_ESM_RF and ECFP_ESM_GBM
for model in ECFP_ESM_RF ECFP_ESM_GBM; do
sh sh_run/run_CPI.sh $model ic50 baseline_CPI 2 ic50 2 none
done
sh sh_run/run_CPI.sh ECFP_ESM_RF ic50 baseline_CPI 2 ic50 2 none

# run KANO_ESM
sbatch run_CPI.sbatch KANO_ESM ic50 retrain 2 ic50 2 none
sbatch run_CPI.sbatch KANO_ESM ki retrain 2 ki 2 none
sbatch run_CPI.sbatch KANO_ESM kd train 2 kd 2 none
sh sh_run/run_CPI.sh KANO_ESM ic50 retrain 2 ic50 2 none
sh sh_run/run_CPI.sh KANO_ESM ec50 train 2 ec50 2 none
sh sh_run/run_CPI.sh KANO_ESM MolACE_CPI_ki train 2 MolACE_CPI_ki 2 none
sh sh_run/run_CPI.sh KANO_ESM MolACE_CPI_ec50 train 2 MolACE_CPI_ec50 2 none

# run HyperAttentionDTI
sh sh_run/run_CPI.sh HyperAttentionDTI ki baseline_CPI 2 kd 3 none
for data in ki MolACE_CPI_ec50; do
sbatch run_CPI.sbatch HyperAttentionDTI $data baseline_CPI 2 kd 3 none
done
sh sh_run/run_CPI.sh HyperAttentionDTI ic50 baseline_CPI 2 ic50 2 none
sbatch run_CPI.sbatch HyperAttentionDTI ic50 baseline_CPI 2 ic50 2 none

# run PerceiverCPI
cd Activity-cliff-prediction/CPI_baseline/PerceiverCPI
dataset=ki
savepath=../../exp_results/PerceiverCPI/$dataset/0/
python train.py --data_path ./dataset/"$dataset"_train.csv \
                --separate_val_path ./dataset/"$dataset"_val.csv \
                --separate_test_path ./dataset/"$dataset"_test.csv \
                --metric mse --dataset_type regression \
                --save_dir $savepath --target_columns label \
                --epochs 150 --ensemble_size 1 --num_folds 1 \
                --batch_size 50 --aggregation mean --dropout 0.1 --save_preds


singularity exec --nv --overlay chem_new.ext3:ro /scratch/work/public/singularity/cuda11.6.124-cudnn8.4.0.27-devel-ubuntu20.04.4.sif /bin/bash
source /ext3/env.sh
conda activate chem_gyw
cd Activity-cliff-prediction

sh sh_run/run_CPI.sh PerceiverCPI ki baseline_CPI 2 ki 2 none
sh sh_run/run_CPI.sh PerceiverCPI ic50 baseline_CPI 2 ic50 2 none
                --separate_val_path ./dataset/"$filename"_val.csv \




sbatch run_CPI.sbatch PerceiverCPI ki baseline_CPI 2 ki 2 none

# run GraphDTA 
sh sh_run/run_CPI.sh GraphDTA ic50 baseline_CPI 2 ic50 2 none

# run DeepDTA
sh sh_run/run_CPI.sh DeepDTA ic50 baseline_CPI 2 ic50 2 none

# Classification
sh sh_run/run_CPI_cls.sh KANO_Prot binder_nonbinder_fgfr_pt train 2 ki 2 none
cd Activity-cliff-prediction/sh_run

sbatch run_CPI_cls.sbatch KANO_Prot binder_nonbinder_fgfr_pt retrain 2 binder_nonbinder_fgfr_pt 2 none
sh sh_run/run_CPI_cls.sh KANO_Prot binder_nonbinder_fgfr_pt retrain 2 binder_nonbinder_fgfr_pt 2 none

sbatch run_CPI_cls.sbatch KANO_Prot fgfr_ft_test train 3 fgfr_ft_test 3 none

sh sh_run/run_CPI_cls.sh KANO_Prot binder_nonbinder_fgfr_pt_100k retrain 2 kd 3 none


sh sh_run/run_CPI_cls.sh KANO_Prot fgfr_ft_test retrain 2 fgfr_ft_test 2 none
sh sh_run/run_CPI_cls.sh KANO_Prot fgfr_test train 2 fgfr_test 2 none

sbatch run_CPI_cls.sbatch KANO_Prot qi_CPI2M_fgfr_time train 2 qi_CPI2M_fgfr_time 2 none
sbatch run_CPI_cls.sbatch KANO_Prot qi_fgfr_time train 2 qi_fgfr_time 2 none

sh sh_run/run_CPI_cls.sh KANO_Prot qi_CPI2M_fgfr_time train 2 qi_CPI2M_fgfr_time 2 none
sh sh_run/run_CPI_cls.sh KANO_Prot qi_fgfr_time train 2 qi_fgfr_time 2 none

sh sh_run/run_CPI_cls.sh KANO_Prot qi_CPI2M_fgfr_time finetune 2 binder_nonbinder_fgfr_pt 2 none
sh sh_run/run_CPI_cls.sh KANO_Prot qi_fgfr_time finetune 2 binder_nonbinder_fgfr_pt 2 none

# LIT-PCBA
model=GraphDTA
for dataset in P07550_LITPCBA P00352_LITPCBA P11473_LITPCBA \
    P03372_ago_LITPCBA P03372_ant_LITPCBA P04062_LITPCBA O75874_LITPCBA \
    Q92830_LITPCBA P28482_LITPCBA Q13451_LITPCBA P41145_LITPCBA P14618_LITPCBA \
    P37231_LITPCBA P04637_LITPCBA; do
sh sh_run/run_CPI.sh $model $dataset inference 2 ki 2 none
sh sh_run/run_CPI.sh $model $dataset inference 2 ic50 2 none
# sh sh_run/run_CPI.sh $model "$dataset"_ki baseline_inference 2 ki 2 none
# sh sh_run/run_CPI.sh $model "$datas t"_ic50 baseline_inference 2 ic50 2 none
done

model=ECFP_ESM_GBM
for dataset in P07550_LITPCBA P00352_LITPCBA P11473_LITPCBA \
    P03372_ago_LITPCBA P03372_ant_LITPCBA P04062_LITPCBA O75874_LITPCBA \
    Q92830_LITPCBA P28482_LITPCBA Q13451_LITPCBA P41145_LITPCBA P14618_LITPCBA \
    P37231_LITPCBA P04637_LITPCBA P39748_LITPCBA; do
    echo "Processing $model $dataset..."
    if [ -f "../exp_results/${model}/ic50/2/${dataset}_test_pred_infer.csv" ]; then
        echo "../exp_results/${model}/ic50/2/${dataset}_test_pred_infer.csv exists, skipping..."
        continue
    fi
    # if [ -f "../CPI_baseline/PerceiverCPI/${dataset}_ic50_infer.csv" ]; then
    #     echo "../CPI_baseline/PerceiverCPI/${dataset}_ic50_infer.csv exists, skipping..."
    #     continue
    # fi  
    sbatch run_CPI.sbatch $model $dataset baseline_inference 2 ic50 2 none
    # sbatch run_CPI.sbatch $model "$dataset"_ic50 baseline_inference 2 ic50 2 none
    # sh sh_run/run_CPI.sh $model $dataset inference 2 ic50 2 none

    if [ -f "../exp_results/${model}/ki/2/${dataset}_test_pred_infer.csv" ]; then
        echo "../exp_results/${model}/ki/2/${dataset}_test_pred_infer.csv exists, skipping..."
        continue
    fi
    # if [ -f "../CPI_baseline/PerceiverCPI/${dataset}_ki_infer.csv" ]; then
    #     echo "../CPI_baseline/PerceiverCPI/${dataset}_ki_infer.csv exists, skipping..."
    #     continue
    # fi
    sbatch run_CPI.sbatch $model $dataset baseline_inference 2 ki 2 none
    # sbatch run_CPI.sbatch $model "$dataset"_ki baseline_inference 2 ki 2 none
    # sh sh_run/run_CPI.sh $model $dataset inference 2 ki 2 none
    # sh sh_run/run_CPI.sh $model $dataset inference 2 ki 2 none
    # sh sh_run/run_CPI.sh $model $dataset inference 2 ic50 2 none
    # sh sh_run/run_CPI.sh $model "$dataset"_ki baseline_inference 2 ki 2 none
    # sh sh_run/run_CPI.sh $model "$dataset"_ic50 baseline_inference 2 ic50 2 none
done

