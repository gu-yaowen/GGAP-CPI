import os
import random
import json
import logging
import molvs
import requests
import torch
import numpy as np
from rdkit import Chem
from yaml import load, Loader
from argparse import Namespace
from warnings import simplefilter
from chemprop.data import StandardScaler
from chembl_webresource_client.new_client import new_client
from MoleculeACE.benchmark.cliffs import ActivityCliffs


def define_logging(args, logger):
    """ Define logging handler.

    :param args: Namespace object
    :param logger: logger object
    """
    handler = logging.FileHandler(os.path.join(args.save_path, 'logs.log'))
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    return


def set_up(args):
    """ Set up arguments, logger, seed, save path.

    :param args: Namespace object
    :return: args, logger
    """
    set_save_path(args)
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.INFO)
    define_logging(args, logger)

    simplefilter(action='ignore', category=Warning)

    logger.info(f'current task: {args.data_name}') if args.print else None
    logger.info(f'arguments: {args}') if args.print else None

    set_seed(args.seed)

    logger.info(f'random seed: {args.seed}') if args.print else None
    logger.info(f'save path: {args.save_path}') if args.print else None

    args.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() 
                    and not args.no_cuda else 'cpu')
    logger.info(f'device: {args.device}') if args.print else None
    
    return args, logger


def set_save_path(args):
    args.save_path = os.path.join('exp_results', args.train_model, 
                                    args.data_name, str(args.seed))
    if args.mode in ['train', 'retrain']:
        args.save_model_path = os.path.join(args.save_path, f'{args.train_model}_model.pt')
        args.save_best_model_path = os.path.join(args.save_path, f'{args.train_model}_best_model.pt')
        args.save_pred_path = os.path.join(args.save_path, f'{args.train_model}_test_pred.csv')
        args.save_metric_path = os.path.join(args.save_path, f'{args.baseline_model}_metrics.pkl')
    elif args.mode in ['finetune']:
        args.save_model_path = os.path.join(args.save_path, f'{args.train_model}_model_ft.pt')
        args.save_best_model_path = os.path.join(args.save_path, f'{args.train_model}_best_model_ft.pt')
        args.save_pred_path = os.path.join(args.save_path, f'{args.baseline_model}_test_pred_ft.csv')
        args.save_metric_path = os.path.join(args.save_path, f'{args.baseline_model}_metrics_ft.pkl')
    elif args.mode in ['inference']:
        args.save_pred_path = os.path.join(args.save_path, f'{args.baseline_model}_test_pred_infer.csv')
    elif args.mode in ['baseline_CPI', 'baselin_QSAR']:
        args.save_path = os.path.join('exp_results', args.baseline_model, 
                                      args.data_name, str(args.seed))
        args.save_pred_path = os.path.join(args.save_path, f'{args.baseline_model}_test_pred.csv')
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    return args


def get_config(file: str):
    """ Load a yml config file"""
    if file.endswith('.yml') or file.endswith('.yaml'):
        with open(file, "r", encoding="utf-8") as read_file:
            config = load(read_file, Loader=Loader)
    if file.endswith('.json'):
        with open(file, 'r') as f:
            config = json.load(f)
    return config


def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)


def check_molecule(smiles):
    try:
        mol = molvs.Standardizer().standardize(Chem.MolFromSmiles(smiles))
        if mol is None:
            return None
        else:
            if mol.GetNumAtoms() <= 1:
                print(f'Error: {smiles} is invalid')
                return None
            else:
                return Chem.MolToSmiles(mol, isomericSmiles=True)
    except:
        print(f'Error: {smiles} is invalid')
        return None

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


def uniprot_to_pdb(uniprot_id):
    url = f"https://www.uniprot.org/uniprot/{uniprot_id}.txt"
    response = requests.get(url)

    if response.status_code == 200:
        content = response.text
        for line in content.split('\n'):
            if line.startswith("DR   PDB;"):
                pdb_id = line.split(";")[1].strip()
                return pdb_id
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
    

def set_collect_metric(args):
    metric_dict = {}
    if args.mode in ['train', 'retrain', 'finetune']:
        metric_dict['Total'] = []
        for key in args.loss_func_wt.keys():
            metric_dict[key] = []
    else: 
        metric_dict['loss'] = []
    for metric in args.metric_func:
        metric_dict[f'val_{metric}'] = []
        metric_dict[f'test_{metric}'] = []
    return metric_dict


def collect_metric_epoch(args: Namespace, collect_metric: dict, loss: float or dict,
                         val_scores: dict, test_scores: dict):
    if isinstance(loss, dict):
        for key in loss.keys():
            collect_metric[key].append(loss[key])
    else:
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
                    epoch: int = None,
                    optimizer=None,
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
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer is not None else None,
        'data_scaler': {
            'means': scaler.means,
            'stds': scaler.stds
        } if scaler is not None else None,
        'features_scaler': {
            'means': features_scaler.means,
            'stds': features_scaler.stds
        } if features_scaler is not None else None,
        
    }
    torch.save(state, path)


# def calc_cliff_metrics(y_test_pred: Union[List[float], np.array], y_test: Union[List[float], np.array],
#                     cliff_mols_test: List[int] = None, smiles_test: List[str] = None,
#                     y_train: Union[List[float], np.array] = None, smiles_train: List[str] = None,
#                     metrics = 'RMSE', **kwargs):
#     """ Calculate the RMSE of activity cliff compounds

#     :param y_test_pred: (lst/array) predicted test values
#     :param y_test: (lst/array) true test values
#     :param cliff_mols_test: (lst) binary list denoting if a molecule is an activity cliff compound
#     :param smiles_test: (lst) list of SMILES strings of the test molecules
#     :param y_train: (lst/array) train labels
#     :param smiles_train: (lst) list of SMILES strings of the train molecules
#     :param kwargs: arguments for ActivityCliffs()
#     :return: float RMSE on activity cliff compounds
#     """

#     # Check if we can compute activity cliffs when pre-computed ones are not provided.
#     if cliff_mols_test is None:
#         if smiles_test is None or y_train is None or smiles_train is None:
#             raise ValueError('if cliff_mols_test is None, smiles_test, y_train, and smiles_train should be provided '
#                              'to compute activity cliffs')

#     # Convert to numpy array if it is none
#     y_test_pred = np.array(y_test_pred) if type(y_test_pred) is not np.array else y_test_pred
#     y_test = np.array(y_test) if type(y_test) is not np.array else y_test

#     if cliff_mols_test is None:
#         y_train = np.array(y_train) if type(y_train) is not np.array else y_train
#         # Calculate cliffs and
#         cliffs = ActivityCliffs(smiles_train + smiles_test, np.append(y_train, y_test))
#         cliff_mols = cliffs.get_cliff_molecules(return_smiles=False, **kwargs)
#         # Take only the test cliffs
#         cliff_mols_test = cliff_mols[len(smiles_train):]

#     # Get the index of the activity cliff molecules
#     cliff_test_idx = [i for i, cliff in enumerate(cliff_mols_test) if cliff == 1]

#     # Filter out only the predicted and true values of the activity cliff molecules
#     y_pred_cliff_mols = y_test_pred[cliff_test_idx]
#     y_test_cliff_mols = y_test[cliff_test_idx]

#     if metric == 'RMSE':
#         return calc_rmse(y_pred_cliff_mols, y_test_cliff_mols)
#     elif metric == 'R2':
#         return calc_r2(y_pred_cliff_mols, y_test_cliff_mols)
#     elif metric == 'PCC':
#         return calc_pcc(y_pred_cliff_mols, y_test_cliff_mols)