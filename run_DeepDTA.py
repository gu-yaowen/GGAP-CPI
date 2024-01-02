# from MoleculeACE import MPNN, Data, calc_rmse, calc_cliff_rmse, get_benchmark_config
# from Feature.const import Descriptors
# import torch
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
args.save_path = 'exp_results/DeepDTA'
args.data_path = 'data/Ours/CPI_plus_Integrated.csv'
args.baseline_model = 'DeepDTA'
args.mode = 'baseline_CPI'
args.data_name = args.data_path.split('/')[-1].split('.')[0]
args, logger = set_up(args)

from main import process_data_CPI
df_all, test_idx, train_data, val_data, test_data = process_data_CPI(args, logger)

import DeepPurpose.DTI as models
from DeepPurpose.utils import generate_config
drug_encoding = 'CNN'
target_encoding = 'CNN'
# Note: the hyperparameters are reported as the best performing ones in DeepPurpose
# for the KIBA and DAVIS dataset
config = generate_config(drug_encoding = drug_encoding,
                    target_encoding = target_encoding,
                    cls_hidden_dims = [1024,1024,512],
                    train_epoch = 50,
                    LR = 0.001,
                    batch_size = 512,
                    cnn_drug_filters = [32,64,96],
                    cnn_target_filters = [32,64,96],
                    cnn_drug_kernels = [4,6,8],
                    cnn_target_kernels = [4,8,12]
                    )

model = models.model_initialize(**config)
logger.info(f'load {args.baseline_model} model from DeepPurpose')
model = models.model_initialize(**config)

logger.info(f'training {args.baseline_model}...')
if len(val_data) > 0:
    model.train(train=train_data, val=val_data, test=test_data)
else:
    model.train(train=train_data, val=None, test=test_data)
# get predictions
test_pred = model.predict(test_data)
model.save_model(os.path.join(args.save_path,f'{args.baseline_model}')) 

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
