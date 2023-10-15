import os
import random
import molvs
import torch
import numpy as np
from rdkit import Chem
from argparse import Namespace
from chemprop.data import StandardScaler

def set_save_path(args):
    args.save_path = os.path.join('exp_results', args.data_name, str(args.seed))
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    return args

def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)

def check_molecule(smiles):
    mol = molvs.Standardizer().standardize(Chem.MolFromSmiles(smiles))
    if mol is None:
        return False
    else:
        return Chem.MolToSmiles(mol)
    
def set_collect_metric(args):
    metric_dict = {'loss':[]}
    for metric in args.metric_func:
        metric_dict[f'val_{metric}'] = []
        metric_dict[f'test_{metric}'] = []
    return metric_dict

def collect_metric_epoch(args: Namespace, collect_metric: dict, loss: float,
                         val_scores: dict, test_scores: dict):
    collect_metric['loss'].append(loss)
    for metric in args.metric_func:
        collect_metric[f'val_{metric}'].append(val_scores[metric])
        collect_metric[f'test_{metric}'].append(test_scores[metric])
    return collect_metric

def save_checkpoint(path: str,
                    model,
                    scaler: StandardScaler = None,
                    features_scaler: StandardScaler = None,
                    args: Namespace = None):
    """
    Saves a model checkpoint.

    :param model: A MoleculeModel.
    :param scaler: A StandardScaler fitted on the data.
    :param features_scaler: A StandardScaler fitted on the features.
    :param args: Arguments namespace.
    :param path: Path where checkpoint will be saved.
    """
    state = {
        'args': args,
        'state_dict': model.state_dict(),
        'data_scaler': {
            'means': scaler.means,
            'stds': scaler.stds
        } if scaler is not None else None,
        'features_scaler': {
            'means': features_scaler.means,
            'stds': features_scaler.stds
        } if features_scaler is not None else None
    }
    torch.save(state, path)