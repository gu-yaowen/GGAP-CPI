# 🚀 Overview
![Views](https://komarev.com/ghpvc/?username=gu-yaowen&label=GGAP-CPI%20views&color=0e75b6&style=flat)

This project is based on our paper "Complex structure-free compound-protein interaction prediction for mitigating activity cliff-induced discrepancies and integrated bioactivity learning". GGAP-CPI stands for **protein Graph and ligand Graph network with Attention Pooling for Compound-Protein Interaction prediction**.

## 📖 Table of Contents

- [🎯 Introduction](#🎯-Introduction)  
- [🗄️ Dataset](#🗄️-Dataset)  
- [📦 Dependencies](#📦-Dependencies)  
- [🏋️‍♂️ Model Training](#🏋️‍♂️-Model-Training)  
- [🔄 Model Fine‑tuning](#🔄-Model-Finetuning)  
- [🔍 Model Inference](#🔍-Model-Inference)  
- [📊 Benchmark Results](#📊-Benchmark-Results)  
- [🛠️ Use Your Own Data](#🛠️-Use-Your-Own-Data)  
- [📢 Citation](#📢-Citation)


## 🎯 Introduction

Protein-ligand binding affinity assessment plays a pivotal role in virtual drug screening, yet conventional data-driven approaches rely heavily on limited protein-ligand crystal structures. Structure-free compound-protein interaction (CPI) methods have emerged as competitive alternatives, leveraging extensive bioactivity data to serve as more robust scoring functions. However, these methods often overlook two critical challenges that affect data efficiency and modeling accuracy: **the heterogeneity of bioactivity data** due to differences in bioassay measurements, and **the presence of activity cliffs (ACs)**—small chemical modifications that lead to significant changes in bioactivity, which have not been thoroughly investigated in CPI modeling. 

To address these challenges, we present **CPI2M**, a large-scale CPI benchmark dataset containing approximately 2 million bioactivity endpoints across four activity types (Ki, Kd, EC50, and IC50) with AC annotations. Moreover, we developed **GGAP-CPI-IntEns**, a complex structure-free deep learning model trained by integrated bioactivity learning and designed to mitigate the impact of ACs on CPI prediction through advanced protein representation modelling and integrated bioactivity learning. 

Our comprehensive evaluation demonstrates that GGAP-CPI-IntEns outperforms 12 target-specific and 7 general CPI baselines across four key scenarios (**general CPI prediction, rare protein prediction, transfer learning, and virtual screening**) on seven benchmarks (**CPI2M, MoleculeACE, CASF-2016, MerckFEP, DUD-E, DEKOIS-v2, and LIT-PCBA**). Furthermore, GGAPCPI-IntEns not only delivers stable predictions by distinguishing bioactivity differences between ACs and non-ACs, but also enriches binding pocket residues and interactions, underscoring its applicability to real-world binding affinity assessments and virtual drug screening.

## 🗄️ Dataset
![Dataset](https://github.com/gu-yaowen/Activity-cliff-prediction/blob/main/fig/dataset.jpg)

### Bioactivity Dataset Summary

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

We also incorporate [**MoleculeACE**](https://github.com/molML/MoleculeACE) for activity cliff estimation, [**CASF-2016**](http://www.pdbbind.org.cn/casf.php), [**MerckFEP**](https://pubs.acs.org/doi/full/10.1021/acs.jcim.0c00900), [**DUD-E**](https://dude.docking.org/), [**DEKOIS-2**](http://www.dekois.com), and [**LIT-PCBA**](https://drugdesign.unistra.fr/LIT-PCBA/) for virtual screening estimation.

### Description of CPI2M

- **Source**: **EquiVS** (ChEMBL29, BindingDB, PubChem, Probe&Drugs, IUPHAR/BPS), and **Papyrus** (ChEMBL30, EXCAPE, literature)
- **Structure**: **CPI2M-main** for model training and internal evaluating, **CPI2M-few** for external evaluating.
- **Preprocessing**: Including multistep filtering and duplicate cleaning for activity value, unit, ligand, and protein data.

The access of full CPI2M dataset is available at Zenodo: [CPI2M](https://zenodo.org/records/13738981).

## 📦 Dependencies

- `torch==2.4.1+cu121`  
- `torch-geometric==2.6.1`  
- `torch-scatter==2.1.2+pt24cu121`  
- `torch-sparse==0.6.18+pt24cu121`  
- `torch-spline-conv==1.2.2+pt24cu121`  
- `fair-esm==2.0.0`  
- `chemprop==1.6.1`  
- `DeepPurpose==0.1.5`  
- `MoleculeACE==2.0.1`  
- `graphein==1.7.5`  
- `rdkit==2023.9.1`  
- `MolVS==0.1.1`  
- `biopython==1.81`  
- `scikit-learn==1.3.2`  
- `networkx==3.1`  
- `numpy`  
- `pandas`  
- `yaml`


## Model
![Model Architecture](https://github.com/gu-yaowen/Activity-cliff-prediction/blob/main/fig/model.jpg)

## 🏋️‍♂️ Model Training
Please run the following command for model training: 

```
sh run_bash/run_CPI.sh GGAP_CPI {DATA_NAME} train {SEED}
```

parameters include: 1. training dataset; 2. mode (e.g., train); 3. random seed.

## 🔄 Model Finetuning
To use the pretrained GGAP-CPI-IntEns model (ensemble of 10 GGAP-CPI models) for finetuing on your specific dataset, please run the following command:

```
for seed in $(seq 0 9); do
model_path=GGAP_CPI_IntEns_${seed}
sh run_bash/run_CPI.sh GGAP_CPI {DATA_NAME} finetune {SEED} ${model_path}
done
```

## 🔍 Model Inference
Taking "kd.csv" in data folder for example, please run the following command for inferencing:

```
example_data=kd
example_model_path=GGAP_CPI_IntEns_0
example_seed=0
sh run_bash/run_CPI.sh GGAP_CPI ${example_data} inference ${example_seed} ${example_model_path}
```

## 📊 Benchmark Results
The performances of GGAP-CPI and 19 baseline methods are evaluated on CPI2M-main, CPI2M-few, MoleculeACE, CASF-2016, MerckFEP, DUD-E, DEKOIS-2, and LIT-PCBA datasets. For your convience, 
we add the benchmarking result files for each of them in "benchmark_result" folder.


## 🛠️ Use Your Own Data
To train GGAP-CPI from scratch on your own **.CSV** data, which should at least include columns ['smiles', 'Uniprot_id', 'label']. Note that the 'Uniprot_id' can be either the protein UniProt ID that can be accessed from Alphafold2 database or the protein PDB file name that have been stored in the "data/PDB" folder. For raw PDB file, we will automatically extract the first chain as the protein structure for training and testing.
Please run the following commands:
```
# preprocess data
python process_data.py --dataset {DATA} --task {CPI or QSAR} --split {random or ac} --train_ratio {RATIO} --seed {SEED}
# train
sh run_bash/run_CPI.sh {MODEL} {DATA_NAME} train {SEED}
# optional: finetune, inference, ...
```

## 📢 Citation

```
@article{GGAP_CPI,
   author = {Gu, Yaowen and Xia, Song and Ouyang, Qi and Zhang, Yingkai},
   title = {Complex structure-free compound-protein interaction prediction for mitigating activity cliff-induced discrepancies and integrated bioactivity learning},
   DOI = {10.26434/chemrxiv-2025-96p6b},
   year = {2025},
   type = {Journal Article}
}
```
