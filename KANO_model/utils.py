import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from rdkit import Chem
from chemprop.nn_utils import NoamLR
from argparse import Namespace
from typing import List, Union, Tuple
from chemprop.features.featurization import atom_features, bond_features

# Atom feature sizes
MAX_ATOMIC_NUM = 100
ATOM_FEATURES = {
    'atomic_num': list(range(MAX_ATOMIC_NUM)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}

# Distance feature sizes
PATH_DISTANCE_BINS = list(range(10))
THREE_D_DISTANCE_MAX = 20
THREE_D_DISTANCE_STEP = 1
THREE_D_DISTANCE_BINS = list(range(0, THREE_D_DISTANCE_MAX + 1, THREE_D_DISTANCE_STEP))

# len(choices) + 1 to include room for uncommon values; + 2 at end for IsAromatic and mass
ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2
BOND_FDIM = 14

# Memoization
SMILES_TO_GRAPH = {}

eletype_list = [i for i in range(118)]

rel2emb = pickle.load(open('./KANO_model/initial/rel2emb.pkl','rb'))
fg2emb = pickle.load(open('./KANO_model/initial/fg2emb.pkl', 'rb'))
ele2emb = pickle.load(open('./KANO_model/initial/ele2emb.pkl','rb'))

with open('./KANO_model/initial/funcgroup.txt', "r") as f:
    funcgroups = f.read().strip().split('\n')
    name = [i.split()[0] for i in funcgroups]
    smart = [Chem.MolFromSmarts(i.split()[1]) for i in funcgroups]
    smart2name = dict(zip(smart, name))
    
hrc2emb = {}
for eletype in eletype_list:
    hrc_emb = np.random.rand(14)
    hrc2emb[eletype] = hrc_emb

def hrc_features(ele):
    fhrc = hrc2emb[ele]
    return fhrc.tolist()

def ele_features(ele):
    fele = ele2emb[ele]
    return fele.tolist()

def relation_features(e1,e2):
    frel = rel2emb[(e1,e2)]
    return frel.tolist()

def match_fg(mol):
    fg_emb = [[1] * 133]
    pad_fg = [[0] * 133]
    for sm in smart:
        if mol.HasSubstructMatch(sm):
            fg_emb.append(fg2emb[smart2name[sm]].tolist())
    if len(fg_emb) > 13:
        fg_emb = fg_emb[:13]
    else:
        fg_emb.extend(pad_fg * (13 - len(fg_emb)))
    return fg_emb

def get_atom_fdim(args: Namespace) -> int:
    return ATOM_FDIM

def get_bond_fdim(args: Namespace) -> int:
    return BOND_FDIM

def build_optimizer(model: nn.Module, args: Namespace) -> Optimizer:
    """
    Builds an Optimizer.

    :param model: The model to optimize.
    :param args: Arguments.
    :return: An initialized Optimizer.
    """
    increase = ['prompt_generator']
    no_increase_param = [param for name, param in model.named_parameters() if not any(inc in name for inc in increase)]
    increase_param = [param for name, param in model.named_parameters() if any(inc in name for inc in increase)]
    params = [{'params': no_increase_param, 'lr': args.init_lr, 'weight_decay': 0}, {'params': increase_param, 'lr': args.init_lr*5, 'weight_decay': 0}]
    # params = [{'params': model.parameters(),'lr': args.init_lr, 'weight_decay': 0}]
    return Adam(params)


def build_lr_scheduler(optimizer: Optimizer, args: Namespace, total_epochs = None) -> _LRScheduler:
    """
    Builds a learning rate scheduler.

    :param optimizer: The Optimizer whose learning rate will be scheduled.
    :param args: Arguments.
    :param total_epochs: The total number of epochs for which the model will be run.
    :return: An initialized learning rate scheduler.
    """

    total_epochs=total_epochs or [args.epochs] * args.num_lrs
    return NoamLR(
    optimizer=optimizer,
    warmup_epochs=[args.warmup_epochs] *2,
    total_epochs=total_epochs *2,
    steps_per_epoch=args.train_data_size // args.batch_size,
    init_lr=[args.init_lr, args.init_lr*5],
    max_lr=[args.max_lr, args.max_lr*5],
    final_lr=[args.final_lr, args.final_lr*5]
    )

def build_loss_func(args: Namespace) -> nn.Module:
    """
    Gets the loss function corresponding to a given dataset type.

    :param args: Namespace containing the dataset type ("classification" or "regression").
    :return: A PyTorch loss function.
    """
    if args.dataset_type == 'classification':
        return nn.BCEWithLogitsLoss(reduction='none')

    if args.dataset_type == 'regression':
        return nn.MSELoss(reduction='none')
    
    if args.dataset_type == 'multiclass':
        return nn.CrossEntropyLoss(reduction='none')
    
    raise ValueError(f'Dataset type "{args.dataset_type}" not supported.')


class MolGraph:
    """
    A MolGraph represents the graph structure and featurization of a single molecule.

    A MolGraph computes the following attributes:
    - smiles: Smiles string.
    - n_atoms: The number of atoms in the molecule.
    - n_bonds: The number of bonds in the molecule.
    - f_atoms: A mapping from an atom index to a list atom features.
    - f_bonds: A mapping from a bond index to a list of bond features.
    - a2b: A mapping from an atom index to a list of incoming bond indices.
    - b2a: A mapping from a bond index to the index of the atom the bond originates from.
    - b2revb: A mapping from a bond index to the index of the reverse bond.
    """

    def __init__(self, smiles: str, args: Namespace, prompt: bool):
        """
        Computes the graph structure and featurization of a molecule.

        :param smiles: A smiles string.
        :param args: Arguments.
        """
        self.smiles = smiles
        self.n_atoms = 0  # number of atoms
        self.n_bonds = 0  # number of bonds
        self.f_atoms = []  # mapping from atom index to atom features
        self.f_bonds = []  # mapping from bond index to concat(in_atom, bond) features
        
        self.n_real_atoms = 0
        self.n_eles = 0

        self.a2b = []  # mapping from atom index to incoming bond indices
        self.b2a = []  # mapping from bond index to the index of the atom the bond is coming from
        self.b2revb = []  # mapping from bond index to the index of the reverse bond
        self.bonds = []
        # Convert smiles to molecule
        mol = Chem.MolFromSmiles(smiles)
        self.f_fgs = match_fg(mol)
        self.n_fgs = len(self.f_fgs)
        self.prompt = prompt


        if not self.prompt:
            # fake the number of "atoms" if we are collapsing substructures
            self.n_atoms = mol.GetNumAtoms()
            # Get atom features
            for i, atom in enumerate(mol.GetAtoms()):
                self.f_atoms.append(atom_features(atom))
            self.f_atoms = [self.f_atoms[i] for i in range(self.n_atoms)]
            for _ in range(self.n_atoms):
                self.a2b.append([])

            # Get bond features
            for a1 in range(self.n_atoms):
                for a2 in range(a1 + 1, self.n_atoms):
                    bond = mol.GetBondBetweenAtoms(a1, a2)

                    if bond is None:
                        continue

                    f_bond = bond_features(bond)

                    if args.atom_messages:
                        self.f_bonds.append(f_bond)
                        self.f_bonds.append(f_bond)
                    else:
                        self.f_bonds.append(self.f_atoms[a1] + f_bond)
                        self.f_bonds.append(self.f_atoms[a2] + f_bond)

                    # Update index mappings
                    b1 = self.n_bonds
                    b2 = b1 + 1
                    self.a2b[a2].append(b1)  # b1 = a1 --> a2
                    self.b2a.append(a1)
                    self.a2b[a1].append(b2)  # b2 = a2 --> a1
                    self.b2a.append(a2)
                    self.b2revb.append(b2)
                    self.b2revb.append(b1)
                    self.n_bonds += 2
                    self.bonds.append(np.array([a1, a2]))
                    
        else:
            # fake the number of "atoms" if we are collapsing substructures
            self.n_real_atoms = mol.GetNumAtoms()
            # Get atom features
            self.atomic_nums = []
            for i, atom in enumerate(mol.GetAtoms()):
                self.f_atoms.append(atom_features(atom))
                
                atomicnum = atom.GetAtomicNum()
                self.atomic_nums.append(atomicnum)
            
            self.eles = list(set(self.atomic_nums))
            self.eles.sort()
            self.n_eles = len(self.eles)
            self.n_atoms += len(self.eles)+self.n_real_atoms
            
            self.f_eles = [ele_features(self.eles[i]) for i in range(self.n_eles)]
            self.f_atoms += self.f_eles
                        
            self.f_atoms = [self.f_atoms[i] for i in range(self.n_atoms)]
            
            self.atomic_nums += self.eles

            for _ in range(self.n_atoms):
                self.a2b.append([])

            # Get bond features
            for a1 in range(self.n_atoms):
                for a2 in range(a1 + 1, self.n_atoms):
                    if a2 < self.n_real_atoms:
                        bond = mol.GetBondBetweenAtoms(a1, a2)

                        if bond is None:
                            continue

                        # f_bond = self.f_atoms[a1] + bond_features(bond)
                        f_bond = bond_features(bond)
                        
                    
                    elif a1 < self.n_real_atoms and a2 >= self.n_real_atoms:
                        if self.atomic_nums[a1] == self.atomic_nums[a2]:
                            ele = self.atomic_nums[a1]
                            f_bond = hrc_features(ele)
                        else:
                            continue
                            
                    elif a1 >= self.n_real_atoms:
                        if (self.atomic_nums[a1],self.atomic_nums[a2]) in rel2emb.keys():
                            f_bond = relation_features(self.atomic_nums[a1], self.atomic_nums[a2])
                        else:
                            continue      

                    if args.atom_messages:
                        self.f_bonds.append(f_bond)
                        self.f_bonds.append(f_bond)
                    else:
                        self.f_bonds.append(self.f_atoms[a1] + f_bond)
                        self.f_bonds.append(self.f_atoms[a2] + f_bond)
                        
                    # Update index mappings
                    b1 = self.n_bonds
                    b2 = b1 + 1
                    self.a2b[a2].append(b1)  # b1 = a1 --> a2
                    self.b2a.append(a1)
                    self.a2b[a1].append(b2)  # b2 = a2 --> a1
                    self.b2a.append(a2)
                    self.b2revb.append(b2)
                    self.b2revb.append(b1)
                    self.n_bonds += 2
                    self.bonds.append(np.array([a1, a2]))


class BatchMolGraph:
    """
    A BatchMolGraph represents the graph structure and featurization of a batch of molecules.

    A BatchMolGraph contains the attributes of a MolGraph plus:
    - smiles_batch: A list of smiles strings.
    - n_mols: The number of molecules in the batch.
    - atom_fdim: The dimensionality of the atom features.
    - bond_fdim: The dimensionality of the bond features (technically the combined atom/bond features).
    - a_scope: A list of tuples indicating the start and end atom indices for each molecule.
    - b_scope: A list of tuples indicating the start and end bond indices for each molecule.
    - max_num_bonds: The maximum number of bonds neighboring an atom in this batch.
    - b2b: (Optional) A mapping from a bond index to incoming bond indices.
    - a2a: (Optional): A mapping from an atom index to neighboring atom indices.
    """

    def __init__(self, mol_graphs, args: Namespace):
        self.smiles_batch = [mol_graph.smiles for mol_graph in mol_graphs]
        self.n_mols = len(self.smiles_batch)

        self.atom_fdim = get_atom_fdim(args)
        self.bond_fdim = get_bond_fdim(args) + (not args.atom_messages) * self.atom_fdim # * 2
        # Start n_atoms and n_bonds at 1 b/c zero padding
        self.n_atoms = 1  # number of atoms (start at 1 b/c need index 0 as padding)
        self.n_bonds = 1  # number of bonds (start at 1 b/c need index 0 as padding)
        self.n_fgs = 1
        self.atom_num = []
        self.fg_num = []
        self.a_scope = []  # list of tuples indicating (start_atom_index, num_atoms) for each molecule
        self.b_scope = []  # list of tuples indicating (start_bond_index, num_bonds) for each molecule
        self.fg_scope = []

        # All start with zero padding so that indexing with zero padding returns zeros
        f_atoms = [[0] * self.atom_fdim]  # atom features
        f_bonds = [[0] * self.bond_fdim]  # combined atom/bond features
        f_fgs = [] # fg features
        a2b = [[]]  # mapping from atom index to incoming bond indices
        b2a = [0]  # mapping from bond index to the index of the atom the bond is coming from
        b2revb = [0]  # mapping from bond index to the index of the reverse bond
        bonds = [[0,0]]
        for mol_graph in mol_graphs:
            f_atoms.extend(mol_graph.f_atoms)
            f_bonds.extend(mol_graph.f_bonds)
            f_fgs.extend(mol_graph.f_fgs)

            for a in range(mol_graph.n_atoms):
                a2b.append([b + self.n_bonds for b in mol_graph.a2b[a]]) #  if b!=-1 else 0

            for b in range(mol_graph.n_bonds):
                b2a.append(self.n_atoms + mol_graph.b2a[b])
                b2revb.append(self.n_bonds + mol_graph.b2revb[b])
                bonds.append([b2a[-1], 
                              self.n_atoms + mol_graph.b2a[mol_graph.b2revb[b]]])
            self.a_scope.append((self.n_atoms, mol_graph.n_atoms))
            self.b_scope.append((self.n_bonds, mol_graph.n_bonds))
            self.fg_scope.append((self.n_fgs, mol_graph.n_fgs))
            self.atom_num.append(mol_graph.n_atoms)
            self.fg_num.append(mol_graph.n_fgs)
            self.n_atoms += mol_graph.n_atoms
            self.n_bonds += mol_graph.n_bonds
            self.n_fgs += mol_graph.n_fgs
        
        bonds = np.array(bonds).transpose(1,0)
        
        self.max_num_bonds = max(1, max(len(in_bonds) for in_bonds in a2b)) # max with 1 to fix a crash in rare case of all single-heavy-atom mols
        
        self.f_atoms = torch.FloatTensor(f_atoms)
        self.f_bonds = torch.FloatTensor(f_bonds)
        self.f_fgs = torch.FloatTensor(f_fgs)
        self.a2b = torch.LongTensor([a2b[a][:self.max_num_bonds] + [0] * (self.max_num_bonds - len(a2b[a])) for a in range(self.n_atoms)])
        self.b2a = torch.LongTensor(b2a)
        self.bonds = torch.LongTensor(bonds)
        self.b2revb = torch.LongTensor(b2revb)
        self.b2b = None  # try to avoid computing b2b b/c O(n_atoms^3)
        self.a2a = None  # only needed if using atom messages

    def get_components(self) -> Tuple[torch.FloatTensor, torch.FloatTensor,
                                      torch.LongTensor, torch.LongTensor, torch.LongTensor,
                                      List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Returns the components of the BatchMolGraph.

        :return: A tuple containing PyTorch tensors with the atom features, bond features, and graph structure
        and two lists indicating the scope of the atoms and bonds (i.e. which molecules they belong to).
        """
        return self.f_atoms, self.f_bonds, self.a2b, self.b2a, self.b2revb, self.a_scope, self.atom_num, self.fg_num, self.f_fgs, self.fg_scope

    def get_b2b(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each bond index to all the incoming bond indices.

        :return: A PyTorch tensor containing the mapping from each bond index to all the incoming bond indices.
        """

        if self.b2b is None:
            b2b = self.a2b[self.b2a]  # num_bonds x max_num_bonds
            # b2b includes reverse edge for each bond so need to mask out
            revmask = (b2b != self.b2revb.unsqueeze(1).repeat(1, b2b.size(1))).long()  # num_bonds x max_num_bonds
            self.b2b = b2b * revmask

        return self.b2b

    def get_a2a(self) -> torch.LongTensor:
        """
        Computes (if necessary) and returns a mapping from each atom index to all neighboring atom indices.

        :return: A PyTorch tensor containing the mapping from each bond index to all the incodming bond indices.
        """
        if self.a2a is None:
            # b = a1 --> a2
            # a2b maps a2 to all incoming bonds b
            # b2a maps each bond b to the atom it comes from a1
            # thus b2a[a2b] maps atom a2 to neighboring atoms a1
            self.a2a = self.b2a[self.a2b]  # num_atoms x max_num_bonds

        return self.a2a
    
def mol2graph(smiles_batch: List[str],
              args: Namespace, prompt: bool) -> BatchMolGraph:
    """
    Converts a list of SMILES strings to a BatchMolGraph containing the batch of molecular graphs.

    :param smiles_batch: A list of SMILES strings.
    :param args: Arguments.
    :return: A BatchMolGraph containing the combined molecular graph for the molecules
    """
    mol_graphs = []
    for smiles in smiles_batch:
        # if smiles in SMILES_TO_GRAPH:
        #     mol_graph = SMILES_TO_GRAPH[smiles]
        # else:
        if len(smiles[0]) == 1:
            mol_graph = MolGraph(smiles, args, prompt)
        else:
            mol_graph = MolGraph(smiles[0], args, prompt)
            # if not args.no_cache:
            #     SMILES_TO_GRAPH[smiles] = mol_graph
        mol_graphs.append(mol_graph)

    return BatchMolGraph(mol_graphs, args)

# just copy from the another utils.py
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
    return args, logger


def set_save_path(args):
    if args.mode == 'baseline_CPI':
        args.save_path = os.path.join('exp_results', args.baseline_model, str(args.seed))
    else:
        args.save_path = os.path.join('exp_results', args.data_name, str(args.seed))
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
    

def generate_siamse_smi(smiles_list, args, query_prot_ids, 
                        support_prot, support_dataset, strategy='random'):
    smiles, labels = [], []
    for idx, smi in enumerate(smiles_list):
        prot_id = query_prot_ids[idx]
        support_idx = np.where(support_prot == prot_id)[0]
        if strategy == 'random':
            siamse_idx = np.random.choice(support_idx, 1)[0]
        support_data = support_dataset[siamse_idx]
        smiles.append(support_data.smiles)
        labels.append(support_data.targets)
    return smiles, labels


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