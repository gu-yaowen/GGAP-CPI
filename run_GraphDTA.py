import os
# import torch.nn as nn
import numpy as np
import pandas as pd
from main import set_up
from argparse import Namespace
args = Namespace(activation='ReLU', atom_messages=False, batch_size=256, bias=False,
                 checkpoint_dir=None,
                 checkpoint_path='KANO_model/dumped/pretrained_graph_encoder/original_CMPN_0623_1350_14000th_epoch.pkl', checkpoint_paths=['KANO_model/dumped/pretrained_graph_encoder/original_CMPN_0623_1350_14000th_epoch.pkl'],
                 config_path=None, crossval_index_dir=None, crossval_index_file=None, cuda=True,
                 data_path='data/MoleculeACE/CHEMBL214_Ki_Integrated.csv', dataset_type='regression', depth=3,
                 dropout=0.0, dump_path='dumped', encoder_name='CMPNN', ensemble_size=1, epochs=100,
                 exp_id='bbbp_test', exp_name='finetune', features_generator=None, features_only=False,
                 features_path=None, features_scaling=True, ffn_hidden_size=300, ffn_num_layers=2, final_lr=0.0001,
                 folds_file=None, gpu=0, hidden_size=300, init_lr=0.0001, log_frequency=10, max_data_size=None,
                 max_lr=0.001, metric='r2', minimize_score=True, multiclass_num_classes=3, no_cache=False,
                 num_lrs=1, num_runs=1, quiet=False, save_dir=None, save_smiles_splits=False, seed=0,
                 separate_test_features_path=None, separate_test_path=None, separate_val_features_path=None,
                 separate_val_path=None, show_individual_scores=False, split_sizes=[0.8, 0.1, 0.1],
                 split_type='scaffold_balanced', step='functional_prompt', temperature=0.1, test=False,
                 test_fold_index=None, undirected=False, use_compound_names=False, use_input_features=None,
                 val_fold_index=None, warmup_epochs=2.0)
args.save_path = 'exp_results/GraphDTA'
args.data_path = 'data/MoleculeACE/CPI_plus_Integrated.csv'
args.baseline_model = 'GraphDTA'
args.mode = 'baseline_CPI'
args.data_name = args.data_path.split('/')[-1].split('.')[0]
args, logger = set_up(args)

args.data_name = args.data_path.split('/')[-1].split('.')[0]

import random
from tqdm import tqdm
from data_prep import smiles_to_graph
args.smiles_columns = ['smiles']
args.target_columns = ['y']

# load data
df_data = pd.DataFrame()
chembl_list = []
if args.split_sizes:
    _, valid_ratio, test_ratio = args.split_sizes

df_data = pd.read_csv(args.data_path)
logger.info(f'Loading data from {args.data_path}')
chembl_list_2 = df_data['Chembl_id'].unique()
logger.info('{} are not included in the dataset'.format(set(chembl_list) - set(chembl_list_2)))

X_drug = df_data['smiles'].values
X_target = df_data['Sequence'].values
y = df_data['y'].values
train_idx, test_idx = list(df_data[df_data['split'].values == 'train'].index), \
                    list(df_data[df_data['split'].values == 'test'].index)
val_idx = random.sample(list(train_idx), int(len(df_data) * valid_ratio))
train_idx = list(set(train_idx) - set(val_idx))
train_data = df_data.iloc[train_idx].reset_index(drop=True)
val_data = df_data.iloc[val_idx].reset_index(drop=True)
test_data = df_data.iloc[test_idx].reset_index(drop=True)

if not os.path.exists(os.path.join(args.save_path, 'processed', f'{args.data_name}_train.pt')):
    train_graph = {}
    logger.info('Training set: converting SMILES to graph data...')
    for s in tqdm(train_data['smiles'].values):
        g = smiles_to_graph(s)
        train_graph[s] = g
    val_graph = {}
    logger.info('Validation set: converting SMILES to graph data...')
    for s in tqdm(val_data['smiles'].values):
        g = smiles_to_graph(s)
        val_graph[s] = g
    test_graph = {}
    logger.info('Test set: converting SMILES to graph data...')
    for s in tqdm(test_data['smiles'].values):
        g = smiles_to_graph(s)
        test_graph[s] = g

    from data_prep import seq_cat
    train_smiles, val_smiles, test_smiles = train_data['smiles'].values, \
                                            val_data['smiles'].values, \
                                            test_data['smiles'].values

    train_protein = [seq_cat(t) for t in train_data['Sequence'].values]
    val_protein = [seq_cat(t) for t in val_data['Sequence'].values]
    test_protein = [seq_cat(t) for t in test_data['Sequence'].values]
else:
    train_smiles, val_smiles, test_smiles = [], [], []
    train_protein, val_protein, test_protein = [], [], []
    train_graph, val_graph, test_graph = {}, {}, {}
    train_label, val_label, test_label = [], [], []

from CPI_baseline.utils import TestbedDataset
from torch_geometric.data import DataLoader
train_data = TestbedDataset(root=args.save_path, dataset=args.data_name+'_train',
        xd=train_smiles, xt=train_protein, y=train_label, smile_graph=train_graph)
train_data = DataLoader(train_data, batch_size=512, shuffle=True)
if len(val_data) > 0:
    val_data = TestbedDataset(root=args.save_path, dataset=args.data_name+'_val',
            xd=val_smiles, xt=val_protein, y=val_label, smile_graph=val_graph)
    val_data = DataLoader(val_data, batch_size=512, shuffle=False)
else:
    val_data = []
test_data = TestbedDataset(root=args.save_path, dataset=args.data_name+'_test',
        xd=test_smiles, xt=test_protein, y=test_label, smile_graph=test_graph)
test_data = DataLoader(test_data, batch_size=512, shuffle=False)

from CPI_baseline.GraphDTA import GraphDTA
model = GraphDTA(args, logger)
logger.info(f'load {args.baseline_model} model')
logger.info(f'training {args.baseline_model}...')
model.train(args, logger, train_data, val_data)
_, test_pred = model.predict(test_data)

# save prediction
test_data_all = df_all[df_all['split']=='test']

if 'Chembl_id' in test_data_all.columns:
    test_data_all['Chembl_id'] = test_data_all['Chembl_id'].values
    task = test_data_all['Chembl_id'].unique()
else:
    task = test_data_all['UniProt_id'].unique()

test_data_all['Prediction'] = test_pred[:len(test_data_all)] # some baseline may have padding, delete the exceeds
test_data_all = test_data_all.rename(columns={'Label': 'y'})
test_data_all.to_csv(os.path.join(args.save_path, f'{args.data_name}_test_pred.csv'), index=False)
rmse, rmse_cliff = [], []

for target in task:
    if 'Chembl_id' in test_data_all.columns:
        test_data_target = test_data_all[test_data_all['Chembl_id']==target]
    else:
        test_data_target = test_data_all[test_data_all['UniProt_id']==target]
    rmse.append(calc_rmse(test_data_target['y'].values, test_data_target['Prediction'].values))
    rmse_cliff.append(calc_cliff_rmse(y_test_pred=test_data_target['Prediction'].values,
                                        y_test=test_data_target['y'].values,
                                    cliff_mols_test=test_data_target['cliff_mol'].values))
                                    
logger.info('Prediction saved, RMSE: {:.4f}±{:.4f}, '
            'RMSE_cliff: {:.4f}±{:.4f}'.format(np.mean(rmse), np.std(rmse),
                                                np.mean(rmse_cliff), np.std(rmse_cliff)))
