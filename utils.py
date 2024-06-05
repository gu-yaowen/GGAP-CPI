import os
import random
import json
import logging
import molvs
import requests
import torch
import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from yaml import load, Loader
from argparse import Namespace
from warnings import simplefilter
from chemprop.data import StandardScaler
from chembl_webresource_client.new_client import new_client
from MoleculeACE.benchmark.cliffs import ActivityCliffs
from KANO_model.model import MoleculeModel, prompt_generator_output


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
    if args.ablation == 'none':
        args.save_path = os.path.join('exp_results', args.train_model, 
                                        args.data_name, str(args.seed))
    else:
        args.save_path = os.path.join('exp_results', args.train_model + '_' + args.ablation, 
                                        args.data_name, str(args.seed))
    if args.mode in ['train', 'retrain']:
        args.save_model_path = os.path.join(args.save_path, f'{args.train_model}_model.pt')
        args.save_best_model_path = os.path.join(args.save_path, f'{args.train_model}_best_model.pt')
        args.save_pred_path = os.path.join(args.save_path, f'{args.train_model}_test_pred.csv')
        args.save_metric_path = os.path.join(args.save_path, f'{args.train_model}_metrics.pkl')
    elif args.mode in ['finetune']:
        args.save_model_path = os.path.join(args.save_path, f'{args.train_model}_model_ft.pt')
        args.save_best_model_path = os.path.join(args.save_path, f'{args.train_model}_best_model_ft.pt')
        args.save_pred_path = os.path.join(args.save_path, f'{args.data_name}_test_pred_ft.csv')
        args.save_metric_path = os.path.join(args.save_path, f'{args.train_model}_metrics_ft.pkl')
    elif args.mode in ['inference']:
        args.save_path = args.model_path
        args.save_pred_path = os.path.join(args.save_path, f'{args.data_name}_test_pred_infer.csv')
        args.save_best_model_path = os.path.join(args.save_path, f'{args.train_model}_best_model.pt')
    elif args.mode in ['baseline_CPI', 'baseline_QSAR']:
        if args.mode == 'baseline_CPI':
            args.save_path = os.path.join('exp_results', args.baseline_model, 
                                        args.data_name, str(args.seed))
        else:
            args.save_path = os.path.join('exp_results', args.baseline_model, args.endpoint_type,
                                        args.data_name, str(args.seed))
        args.save_pred_path = os.path.join(args.save_path, f'{args.data_name}_test_pred.csv')
    elif args.mode in ['baseline_inference']:
        args.save_path = args.model_path
        args.save_pred_path = os.path.join(args.save_path, f'{args.data_name}_test_pred_infer.csv')
        if args.train_model == 'KANO_ESM':
            args.save_best_model_path = os.path.join(args.save_path, f'{args.baseline_model}_best_model.pt')
        elif args.baseline_model == 'DeepDTA':
            args.save_best_model_path = os.path.join(args.save_path, 'model.pt')
        elif args.baseline_model == 'GraphDTA':
            args.save_best_model_path = os.path.join(args.save_path, 'GraphDTA.pt')
        elif args.baseline_model == 'HyperAttentionDTI':
            args.save_best_model_path = os.path.join(args.save_path, 'HyperAttentionDTI.pt')
        elif args.baseline_model == 'PerceiverCPI':
            args.save_best_model_path = os.path.join(args.save_path, 'fold_0', 'model_0', 'model.pt')
        elif args.baseline_model in ['ECFP_ESM_GBM', 'ECFP_ESM_RF', 'KANO_ESM_GBM', 'KANO_ESM_RF']:
            args.save_best_model_path = os.path.join(args.save_path, f'{args.baseline_model}_model.pkl')
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
    

def get_molecule_feature(args, logger, smiles):
    logger.info(f'loading molecule features...') if args.print else None
    args.atom_output = False
    molecule_encoder = MoleculeModel(classification=args.dataset_type == 'classification',
                                    multiclass=args.dataset_type == 'multiclass',
                                    pretrain=False)
    molecule_encoder.create_encoder(args, 'CMPNN')
    molecule_encoder.encoder.encoder.W_i_atom = prompt_generator_output(args)(
                                        molecule_encoder.encoder.encoder.W_i_atom)
    molecule_encoder.load_state_dict(torch.load(args.checkpoint_path, map_location='cpu'), strict=False)
    molecule_encoder.to(args.device)
    molecule_encoder.eval()
    feat = []
    if len(smiles) > 0:
        for i in range(0, len(smiles), args.batch_size):
            mol_feat, _ = molecule_encoder.encoder('finetune', False, 
                                smiles[i: i + args.batch_size if i + args.batch_size < len(smiles) else len(smiles)])
            mol_feat = mol_feat.detach().cpu().numpy()
            for j in mol_feat:
                feat.append(j)
    return feat


def get_protein_feature(args, logger, df_all):
    logger.info('loading protein features...') if args.print else None
    prot_list = df_all['Uniprot_id'].unique()

    prot_graph_dict = {}
    for prot_id in prot_list:
        with open(f'data/Protein_pretrained_feat/{prot_id}.pkl', 'rb') as f:
            prot_feat = pickle.load(f)
        prot_feat_values = list(prot_feat.values())[0]
        feat, graph = prot_feat_values[1], prot_feat_values[-1]

        try:
            # x = torch.tensor(feat[:graph.num_nodes], device=args.device)
            x = torch.tensor(feat[:graph.num_nodes])
            if x.shape[0] < graph.num_nodes:
                # x = torch.cat([x, torch.zeros(graph.num_nodes - x.shape[0], x.shape[1], device=args.device)], dim=0)
                x = torch.cat([x, torch.zeros(graph.num_nodes - x.shape[0], x.shape[1])], dim=0)
            graph.x = x
        except Exception as e:
            logger.error(f'Error processing {prot_id}: {e}')
            continue

        prot_graph_dict[prot_id] = graph

    return prot_graph_dict


def set_collect_metric(args):
    metric_dict = {}
    if args.mode in ['train', 'retrain', 'finetune']:
        metric_dict['Total'] = []
        if args.dataset_type == 'regression':
            keys = ['MSE', 'CLS', 'CL']
        elif args.dataset_type == 'classification':
            keys = ['AUC', 'AUPR', 'CrossEntropy']
        for key in keys:
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


def get_fingerprint(smiles_list):
    mols = [Chem.MolFromSmiles(smile) for smile in smiles_list]
    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) for mol in mols]
    return np.array(fps)


def get_residue_onehot_encoding(args, batch_prot):
    residue = batch_prot.node_id
    res_feat = []
    for idx in range(len(residue)):
        res_list = [res.split(':')[1].upper() for res in residue[idx]]
        res_feat.extend(generate_onehot_features(res_list))
    batch_prot.x = torch.tensor(res_feat).float().to(args.device)
    return batch_prot


def generate_onehot_features(residue_sequence):
    amino_acids = ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS',
                   'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP',
                   'TYR', 'VAL']
    one_code = {'A': 'ALA', 'R': 'ARG', 'N': 'ASN', 'D': 'ASP', 'C': 'CYS',
                'Q': 'GLN', 'E': 'GLU', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
                'L': 'LEU', 'K': 'LYS', 'M': 'MET', 'F': 'PHE', 'P': 'PRO',
                'S': 'SER', 'T': 'THR', 'W': 'TRP', 'Y': 'TYR', 'V': 'VAL'}
    aa_to_index = {aa: idx for idx, aa in enumerate(amino_acids)}
    one_hot_features = []
    for aa in residue_sequence:
        one_hot = np.zeros(len(amino_acids))
        if aa in one_code:
            aa = one_code[aa]
        one_hot[aa_to_index[aa]] = 1
        one_hot_features.append(one_hot)
    
    return one_hot_features
