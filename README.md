# Overview
![Views](https://komarev.com/ghpvc/?username=gu-yaowen&label=GGAP-CPI%20views&color=0e75b6&style=flat)

This project is based on our paper "Mitigating Activity Cliff-induced Discrepancies in Deep Learning of Compound-Protein Interaction Affinity Prediction". GGAP-CPI stands for **protein Graph and ligand Graph network with Attention Pooling for Compound-Protein Interaction prediction**.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#Dataset)
- [Model Training](#Model-Training)
- [Model Inference](#model-inference)
- [Benchmark Results](#benchmark-results)
- [Use Your Own Data](#Use-Your-Own-Data)
- [Citation](#citation)

## Introduction

Machine learning-based protein-ligand binding affinity prediction is crucial for drug virtual screening. **Structure-free Compound-Protein Interaction (CPI) prediction methods** leverage millions of bioassay measurements, offering greater flexibility than crystal structure-dependent methods. However, **activity cliffs**—minor chemical modifications leading to substantial bioactivity changes—pose significant challenges that are not well-explored in CPI prediction.

In this study, we present **CPI2M**, a large-scale CPI benchmark dataset containing approximately **2 million endpoints** across four activity types (Ki, Kd, EC50, and IC50), specifically annotated for activity cliff data. Additionally, we developed **GGAP-CPI**, a deep learning model that mitigates the impact of activity cliffs through protein feature fusion and activity type-based transfer learning.

Our evaluation across three simulated scenarios (general, unknown proteins, and transfer learning) shows that **GGAP-CPI outperforms 12 target-specific and 7 general CPI methods**, excelling in distinguishing bioactivity differences among activity cliff molecules. It also demonstrates comparable performance on **CASF-2016 and LIT-PCBA benchmarks**, highlighting its potential for practical virtual screening.

This study proposes effective strategies for investigating activity cliffs in CPI predictions, enhancing our understanding of structure-activity relationship discontinuities and aiding future CPI methodology development.

## Dataset
![Dataset](https://github.com/gu-yaowen/Activity-cliff-prediction/blob/main/fig/dataset.jpg)

### Dataset Summary

| Dataset                | Activity Type | Num.     | Num. Mol. | Num. Prot. | Avg. Bioactivity | Std. Bioactivity | % AC   |
|------------------------|---------------|----------|-----------|------------|------------------|------------------|--------|
| **CPI2M-main** (train, internal validation) | Ki            | 341,244  | 124,345   | 418        | 6.50             | 1.42             | 25.39  |
|                        | Kd            | 4,337    | 3,212     | 21         | 6.90             | 1.60             | 34.03  |
|                        | EC50          | 88,302   | 61,095    | 178        | 5.80             | 1.56             | 25.08  |
|                        | IC50          | 751,941  | 419,985   | 1115       | 6.15             | 1.47             | 30.60  |
| **CPI2M-few** (external validation) | Ki            | 65,529   | 41,365    | 2373       | 6.20             | 1.60             | -      |
|                        | Kd            | 55,017   | 14,667    | 1564       | 5.79             | 1.35             | -      |
|                        | EC50          | 42,301   | 28,818    | 1506       | 6.00             | 1.48             | -      |
|                        | IC50          | 148,929  | 94,883    | 4562       | 5.69             | 1.43             | -      |

We also incorporate [**MoleculeACE**](https://github.com/molML/MoleculeACE) for activity cliff estimation, [**CASF-2016**](http://www.pdbbind.org.cn/casf.php) and [**LIT-PCBA**](https://drugdesign.unistra.fr/LIT-PCBA/) for virtual screening estimation. We have organized these datasets and all of them can be downloaded by: 

### Description of CPI2M

- **Source**: **EquiVS** (ChEMBL29, BindingDB, PubChem, Probe&Drugs, IUPHAR/BPS), and **Papyrus** (ChEMBL30, EXCAPE, literature)
- **Structure**: **CPI2M-main** for model training and internal evaluating, **CPI2M-few** for external evaluating.
- **Preprocessing**: Including multistep filtering and duplicate cleaning for activity value, unit, ligand, and protein data.
  

## Model
![Model Architecture](https://github.com/gu-yaowen/Activity-cliff-prediction/blob/main/fig/model.jpg)

## Model training
Available training dataset includes: **CPI2M-main** (noted as 'ki', 'kd', 'ec50', 'ic50', 'integrated'), **MoleculeACE** ('MolACE_CPI_ki', 'MolACE_CPI_ec50'), and **PDBbind** ('PDBbind_all', only for reproducing GGAP-CPI-ft on CASF-2016).

Please run the following command for model training: 

```
sh run_bash/run_CPI.sh KANO_Prot {DATA_NAME} train {SEED}
```

parameters include: 1. training dataset; 2. mode (e.g., train); 3. random seed.

## Model Inference
We provide pretrained GGAP-CPI model on **CPI2M-main** dataset with different activity type predictor(GGAP-CPI-pKi, -pKd, -pEC50, -pIC50, and -pAC). We generally recommand you to use **GGAP-CPI-pKi**, **GGAP-CPI-pIC50** or **GGAP-CPI-pAC** (AC means pretrained on "integrated" activity data) for inferencing on your own data. 

Available inference dataset includes: **CPI2M-few** ('ki_last', 'kd_last', 'ec50_last', 'ic50_last'), **CASF-2016** ('PDBbind_CASF'), and **LIT-PCBA** ('UNIPROT_ID_LITPCBA').

Taking GGAP-CPI-pIC50 for example, please run the following command for model inference on MoleculeACE-pEC50 dataset:

```
sh run_bash/run_CPI.sh KANO_Prot MolACE_CPI_ec50 inference {SEED} ic50/2
```

parameters include: 1. inference dataset; 2. mode (e.g., inference, finetune); 3. random seed; 4. pretrained model path (only for KANO_Prot);

## Benchmark Results
The performances of GGAP-CPI and 19 baseline methods are evaluated on CPI2M-main, CPI2M-few, MoleculeACE, CASF-2016, and LIT-PCBA datasets. For your convience, 
we add the benchmarking result files for each of them in "benchmark_result" folder. \
Also, to reproduce results for GGAP-CPI, you can run the code:

Please note that there should be mild performance differences with different devices and package versions.

## Use Your Own Data
To train GGAP-CPI and other baseline models on your own **.CSV** data, which should at least include columns ['smiles', 'Uniprot_id', 'label']. Please run the following commands:
```
# preprocess data
python process_data.py --dataset {DATA} --task {CPI or QSAR} --split {random or ac} --train_ratio {RATIO} --seed {SEED}
# train
sh run_bash/run_CPI.sh {MODEL} {DATA_NAME} train {SEED}
# optional: finetune, inference, ...
```

GGAP-CPI is also applicable for classification tasks such as binder/nonbinder classification and drug-target interaction prediction. Please replace ```run_CPI.sh``` by ```run_CPI_cls.sh``` for model training and testing.

## Citation
TBD
