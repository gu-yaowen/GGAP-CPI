import os
import random
import json
import molvs
import requests
import torch
import numpy as np
from rdkit import Chem
from yaml import load, Loader
from argparse import Namespace
from chemprop.data import StandardScaler
from chembl_webresource_client.new_client import new_client

def set_save_path(args):
    if args.mode == 'baseline_CPI':
        args.save_path = os.path.join('exp_results', args.baseline_model)
    else:
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
    

def chembl_to_uniprot(chembl_id):
    target = new_client.target
    res = target.filter(target_chembl_id=chembl_id)
    if res:
        components = res[0]['target_components']
        for component in components:
            for xref in component['target_component_xrefs']:
                if xref['xref_src_db'] == 'UniProt':
                    return xref['xref_id']
    return None


def get_protein_sequence(uniprot_id):
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
    response = requests.get(url)
    if response.status_code == 200:
        fasta_data = response.text
        # The first line in FASTA format is the description, so we skip it
        sequence = "".join(fasta_data.split("\n")[1:])
        return sequence
    else:
        print(f"Error {response.status_code}: Unable to fetch data for {uniprot_id}")
        return None
    

def get_config(file: str):
    """ Load a yml config file"""
    if file.endswith('.yml') or file.endswith('.yaml'):
        with open(file, "r", encoding="utf-8") as read_file:
            config = load(read_file, Loader=Loader)
    if file.endswith('.json'):
        with open(file, 'r') as f:
            config = json.load(f)
    return config


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


def get_metric_func(args):
    if args.dataset_type == 'regression':
        metric_func = ['rmse', 'mae', 'r2']
        metric_func.remove(args.metric) 
        args.metric_func = [args.metric] + metric_func
    elif args.dataset_type == 'classification':
        metric_func = ['auc', 'prc-auc', 'accuracy', 'cross_entropy']
        metric_func.remove(args.metric)
        args.metric_func = [args.metric]  + metric_func
    return args.metric_func


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