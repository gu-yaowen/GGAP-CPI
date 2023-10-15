import os
import numpy as np
import pandas as pd
import molvs
from rdkit import Chem
from chemprop.data.utils import get_class_sizes, get_data, get_task_names
from chemprop.data import MoleculeDataset, StandardScaler
from KANO_model.model import build_model, add_functional_prompt
from KANO_model.utils import build_optimizer, build_lr_scheduler, build_loss_func
from data_prep import split_data
from warnings import simplefilter
import random
import torch
import logging
from chemprop.train.evaluate import evaluate_predictions
import importlib
import train_val
importlib.reload(train_val)
from train_val import predict_epoch, train_epoch, evaluate_epoch
from chemprop.train.evaluate import evaluate_predictions
from torch.optim.lr_scheduler import ExponentialLR
from args import add_args
from utils import set_save_path, set_seed, check_molecule, set_collect_metric, \
                  collect_metric_epoch, save_checkpoint
import pickle

def define_logging(args, logger):
    handler = logging.FileHandler(os.path.join(args.save_path, 'logs.log'))
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    return

def predict_main(args):
    return

def train_main(args):
    set_save_path(args)
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.INFO)
    define_logging(args, logger)

    simplefilter(action='ignore', category=Warning)
    logger.info(f'current task: {args.data_name}')
    logger.info(f'arguments: {args}')
    
    # check the validity of SMILES
    df = pd.read_csv(args.data_path)
    df[args.smiles_columns] = df[args.smiles_columns].applymap(check_molecule)
    df = df.dropna(subset=args.smiles_columns)

    set_seed(args.seed)
    logger.info(f'random seed: {args.seed}')
    logger.info(f'save path: {args.save_path}')

    if args.split_sizes:
        _, valid_ratio, test_ratio = args.split_sizes
    # get splitting index and calculate the activity cliff based on MoleculeACE
    if args.split_type == 'moleculeACE':
        if 'split' not in df.columns and 'cliff_mol' not in df.columns:
            df = split_data(df[args.smiles_columns].values,
                            bioactivity=df[args.target_columns].values,
                            in_log10=True, similarity=0.9, test_size=test_ratio, random_state=args.seed)
            df.to_csv(args.data_path, index=False)
            args.ignore_columns = ['exp_mean [nM]', 'split', 'cliff_mol']
        else:
            args.ignore_columns = None
        pos_num, neg_num = len(df[df['cliff_mol']==1]), len(df[df['cliff_mol']==0])
        logger.info(f'ACs: {pos_num}, non-ACs: {neg_num}')

    # get data from csv file
    args.task_names = get_task_names(args.data_path, args.smiles_columns,
                                    args.target_columns, args.ignore_columns)
    data = get_data(path=args.data_path, 
                    smiles_columns=args.smiles_columns,
                    target_columns=args.target_columns,
                    ignore_columns=args.ignore_columns)
    
    # split data by MoleculeACE
    if args.split_sizes:
        train_idx, test_idx = df[df['split']=='train'].index, df[df['split']=='test'].index
        val_idx = random.sample(list(train_idx), int(len(df) * valid_ratio))
    train_data, val_data, test_data = tuple([[data[i] for i in train_idx],
                                        [data[i] for i in val_idx],
                                        [data[i] for i in test_idx]])
    train_data, val_data, test_data = MoleculeDataset(train_data), \
                                    MoleculeDataset(val_data), \
                                    MoleculeDataset(test_data)
    logger.info(f'total size: {len(data)}, train size: {len(train_data)}, '
                f'val size: {len(val_data)}, test size: {len(test_data)}')

    if args.features_scaling:
        features_scaler = train_data.normalize_features(replace_nan_token=0)
        val_data.normalize_features(features_scaler)
        test_data.normalize_features(features_scaler)
    else:
        features_scaler = None
    
    if args.dataset_type == 'regression':
        _, train_targets = train_data.smiles(), train_data.targets()
        scaler = StandardScaler().fit(train_targets)
        scaled_targets = scaler.transform(train_targets).tolist()
        train_data.set_targets(scaled_targets)
    else:
        # get class sizes for classification
        get_class_sizes(data)
        scaler = None

    # load KANO model
    model = build_model(args, encoder_name=args.encoder_name)
    if args.checkpoint_path is not None:
        model.encoder.load_state_dict(torch.load(args.checkpoint_path, map_location='cpu'), strict=False)
    if args.step == 'functional_prompt':
        add_functional_prompt(model, args)
    if args.cuda:
        model = model.cuda()
    logger.info('load KANO model')
    logger.info(f'model: {model}')

    # Optimizers
    optimizer = build_optimizer(model, args)
    logger.info(f'optimizer: {optimizer}')

    # Learning rate schedulers
    args.train_data_size = len(train_data)
    scheduler = build_lr_scheduler(optimizer, args)
    logger.info(f'scheduler: {scheduler}')

    # Loss function
    loss_func = build_loss_func(args)
    logger.info(f'loss function: {loss_func}')

    if args.dataset_type == 'regression':
        args.metric_func = ['rmse', 'r2', 'mse']
    elif args.dataset_type == 'classification':
        args.metric_func = ['auc', 'pr-auc', 'accuracy']
    logger.info(f'metric function: {args.metric_func}')

    n_iter = 0
    args.prompt = False
    metric_dict = set_collect_metric(args)
    best_score = float('inf') if args.minimize_score else -float('inf')
    
    # training
    logger.info(f'training...')
    for epoch in range(args.epochs):
        n_iter, loss = train_epoch(args, model, train_data, loss_func, optimizer, scheduler, n_iter)

        if isinstance(scheduler, ExponentialLR):
            scheduler.step()
        val_scores = evaluate_epoch(args, model, val_data, scaler)

        test_pred = predict_epoch(args, model, test_data, scaler)
        test_scores = evaluate_predictions(test_pred, test_data.targets(),
                                        args.num_tasks, args.metric_func, args.dataset_type)
        
        logger.info('Epoch : {:02d}, Training Loss : {:.4f}, ' \
                    'Validation score : {:.4f}, Test score : {:.4f}'.format(epoch, loss,
                    list(val_scores.values())[0][0], list(test_scores.values())[0][0]))
        metric_dict = collect_metric_epoch(args, metric_dict, loss, val_scores, test_scores)
        
        if args.minimize_score and list(val_scores.values())[0][0] < best_score or \
                not args.minimize_score and list(val_scores.values())[0][0] > best_score:
            best_score, best_epoch = list(val_scores.values())[0][0], epoch
            best_test_score = list(test_scores.values())[0][0]
            save_checkpoint(os.path.join(args.save_path, 'model.pt'), model, scaler, features_scaler, args) 
            # logger.info('Best model saved at epoch : {:02d}, Validation score : {:.4f}'.format(best_epoch, best_score))
    logger.info('Final best performed model in {} epoch, val score: {:.4f}, '
                'test score: {:.4f}'.format(best_epoch, best_score, best_test_score))

    # save results
    pickle.dump(metric_dict, open(os.path.join(args.save_path, 'metric_dict.pkl'), 'wb'))
    df['Prediction'] = None
    df.loc[test_idx, 'Prediction'] = test_pred
    df[df['split']=='test'].to_csv(os.path.join(args.save_path, 'test_pred.csv'), index=False)
    logger.info('Prediction saved')

    logger.handlers.clear()
    return

if __name__ == '__main__':
    args = add_args()

    if args.mode == 'train':
        train_main(args)
    else:
        predict_main(args)