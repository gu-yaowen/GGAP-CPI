"""
Author: Yaowen Gu -- NYU -- 17-10-2023

A collection of data-prepping functions
    - split_data():             split ChEMBL csv into train/test taking similarity and cliffs into account. If you want
                                to process your own data, use this function
    - process_data():           see split_data()
    - load_data():              load a pre-processed dataset from the benchmark
    - fetch_data():             download molecular bioactivity data from ChEMBL for a specific drug target

"""

import os
import pickle
from MoleculeACE.benchmark.cliffs import ActivityCliffs, get_tanimoto_matrix, \
                                        moleculeace_similarity, get_fc
from sklearn.cluster import SpectralClustering
from sklearn.model_selection import train_test_split
from chemprop.data import MoleculeDataset
from typing import List
import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from chemprop.data.utils import get_data, get_task_names
from utils import check_molecule, chembl_to_uniprot, get_protein_sequence
from DeepPurpose.utils import encode_drug, encode_protein
from rdkit import Chem
import networkx as nx
from torch.utils import data
from torch_geometric.data import DataLoader
from CPI_baseline.utils import TestbedDataset, MolTrans_Data_Encoder

DATASET = pickle.load(open('data/datasetList.pkl', 'rb'))

MOLECULEACE_DATALIST = DATASET['MOLECULEACE_DATALIST']
OUR_DATALIST = DATASET['OURS']
    

def process_data_QSAR(args, logger):
    # check the validity of SMILES
    df = pd.read_csv(args.data_path)
    df[args.smiles_columns] = df[args.smiles_columns].apply(check_molecule)
    df = df.dropna(subset=args.smiles_columns)
    df = df.reset_index(drop=True)

    if args.split_sizes:
        _, valid_ratio, test_ratio = args.split_sizes
    # get splitting index and calculate the activity cliff based on MoleculeACE
    if args.split_type == 'moleculeACE':
        if 'split' not in df.columns and 'cliff_mol' not in df.columns:
            df = split_data(df[args.smiles_columns].values.tolist(),
                            bioactivity=df[args.target_columns].values.tolist(),
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
        train_idx = list(set(train_idx) - set(val_idx))
    train_data, val_data, test_data = tuple([[data[i] for i in train_idx],
                                        [data[i] for i in val_idx],
                                        [data[i] for i in test_idx]])
    train_data, val_data, test_data = MoleculeDataset(train_data), \
                                    MoleculeDataset(val_data), \
                                    MoleculeDataset(test_data)
    logger.info(f'total size: {len(data)}, train size: {len(train_data)}, '
                f'val size: {len(val_data)}, test size: {len(test_data)}')
    
    return df, test_idx, train_data, val_data, test_data


def process_data_CPI(args, logger):
    args.smiles_columns = ['smiles']
    args.target_columns = ['y']

    df_data = pd.DataFrame()
    chembl_list = []
    if args.split_sizes:
        _, valid_ratio, test_ratio = args.split_sizes

    if not os.path.exists(args.data_path):
        # integrate bioactivity data
        if 'MoleculeACE' in args.data_path:
            dataset = MOLECULEACE_DATALIST
            datadir = 'MoleculeACE'
        elif 'Ours' in args.data_path:
            dataset = OUR_DATALIST
            datadir = 'Ours'

        for assay_name in dataset:
            df = pd.read_csv(f'data/{datadir}/{assay_name}.csv')
            df[args.smiles_columns] = df[args.smiles_columns].applymap(check_molecule)
            df = df.dropna(subset=args.smiles_columns)

            if 'split' not in df.columns and 'cliff_mol' not in df.columns:
                df = split_data(df[args.smiles_columns].values,
                                    bioactivity=df[args.target_columns].values,
                                    in_log10=True, similarity=0.9, test_size=test_ratio, random_state=args.seed)
                df.to_csv(args.data_path, index=False)
                df['Chembl_id'] = df['UniProt_id']
                df_data = pd.concat([df_data, df])
                chembl_list.append(assay_name.split('_')[0])
        
        pos_num, neg_num = len(df_data[df_data['cliff_mol']==1]), len(df_data[df_data['cliff_mol']==0])
        logger.info(f'ACs: {pos_num}, non-ACs: {neg_num}')

        # protein ID mapping and sequence retrieval
        logger.info('Mapping ChEMBL IDs to UniProt IDs...')
        chembl_uni = dict(zip(chembl_list,
                            [chembl_to_uniprot(chembl_id) for chembl_id in chembl_list]))
        logger.info('Getting target sequences...')
        uni_seq = dict(zip(chembl_uni.values(),
                        [get_protein_sequence(uni_id) for uni_id in chembl_uni.values()]))
        df_data['UniProt_id'] = df_data['Chembl_id'].map(chembl_uni)
        df_data['Sequence'] = df_data['UniProt_id'].map(uni_seq)
        df_data = df_data.dropna(subset=['UniProt_id', 'Sequence'])
        df_data = df_data.reset_index(drop=True)
        logger.info(f'Saving data to {args.data_path}')
        df_data.to_csv(args.data_path, index=False)  
    else:
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
    logger.info(f'total size: {len(df_data)}, train size: {len(train_idx)}, '
                f'val size: {len(val_idx)}, test size: {len(test_idx)}')

    if args.baseline_model == 'DeepDTA':
        df = pd.DataFrame(zip(X_drug, X_target, y))
        df.rename(columns={0:'SMILES', 1: 'Sequence', 2: 'Label'}, inplace=True)

        drug_encoding = 'CNN' 
        target_encoding = 'CNN'
        df = encode_drug(df, drug_encoding, 'SMILES', 'drug_encoding')
        df = encode_protein(df, target_encoding, 'Sequence', 'target_encoding')
        train_data, val_data, test_data = df.iloc[train_idx], df.iloc[val_idx], df.iloc[test_idx]
        train_data = train_data.reset_index(drop=True)
        val_data = val_data.reset_index(drop=True)
        test_data = test_data.reset_index(drop=True)

    elif args.baseline_model == 'GraphDTA':
        train_data = df_data.iloc[train_idx].reset_index(drop=True)
        val_data = df_data.iloc[val_idx].reset_index(drop=True)
        test_data = df_data.iloc[test_idx].reset_index(drop=True)

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

        train_smiles, val_smiles, test_smiles = train_data['smiles'].values, \
                                                val_data['smiles'].values, \
                                                test_data['smiles'].values
        
        train_protein = [seq_cat(t) for t in train_data['Sequence'].values]
        val_protein = [seq_cat(t) for t in val_data['Sequence'].values]
        test_protein = [seq_cat(t) for t in test_data['Sequence'].values]

        train_label, val_label, test_label = train_data['y'].values, \
                                             val_data['y'].values, \
                                             test_data['y'].values
        
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
        
    elif args.baseline_model == 'MolTrans':
        train_data = df_data.iloc[train_idx].reset_index(drop=True)
        val_data = df_data.iloc[val_idx].reset_index(drop=True)
        test_data = df_data.iloc[test_idx].reset_index(drop=True)
        train_data = MolTrans_Data_Encoder(train_data.index.values,
                                          train_data['y'].values, train_data)
        train_data = data.DataLoader(train_data, batch_size=64, shuffle=True, drop_last=True)
        if len(val_data) > 0:
            val_data = MolTrans_Data_Encoder(val_data.index.values,
                                            val_data['y'].values, val_data)
            val_data = data.DataLoader(val_data, batch_size=64, shuffle=False, drop_last=True)
        else:
            val_data = []
        test_data = MolTrans_Data_Encoder(test_data.index.values,
                                         test_data['y'].values, test_data)
        test_data = data.DataLoader(test_data, batch_size=64, shuffle=False, drop_last=False)

    return df_data, test_idx, train_data, val_data, test_data


def split_data(smiles: List[str], bioactivity: List[float], n_clusters: int = 5,
               in_log10 = True, test_size: float = 0.2, random_state: int = 0,
               similarity: float = 0.9, potency_fold: int = 10, remove_stereo: bool = True):
    """ Split data into train/test according to activity cliffs and compounds characteristics.

    :param smiles: (List[str]) list of SMILES strings
    :param bioactivity: (List[float]) list of bioactivity values
    :param n_clusters: (int) number of clusters the data is split into for getting homogeneous data splits
    :param in_log10: (bool) are the bioactivity values in log10?
    :param test_size: (float) test split
    :param similarity:  (float) similarity threshold for calculating activity cliffs
    :param potency_fold: (float) potency difference threshold for calculating activity cliffs
    :param remove_stereo: (bool) Remove racemic mixtures altogether?

    :return: df[smiles, exp_mean [nM], y, cliff_mol, split]
    """

    original_smiles = smiles
    original_bioactivity = bioactivity

    if remove_stereo:
        stereo_smiles_idx = [smiles.index(i) for i in find_stereochemical_siblings(smiles)]
        smiles = [smi for i, smi in enumerate(smiles) if i not in stereo_smiles_idx]
        bioactivity = [act for i, act in enumerate(bioactivity) if i not in stereo_smiles_idx]
        if len(stereo_smiles_idx) > 0:
            print(f"Removed {len(stereo_smiles_idx)} stereoisomers")

    check_matching(original_smiles, original_bioactivity, smiles, bioactivity)
    if not in_log10:
        y_log = -np.log10(bioactivity)
    else:
        y_log = bioactivity

    cliffs = ActivityCliffs(smiles, bioactivity)
    cliff_mols = cliffs.get_cliff_molecules(return_smiles=False, similarity=similarity, potency_fold=potency_fold)

    check_cliffs(cliffs)

    # Perform spectral clustering on a tanimoto distance matrix
    spectral = SpectralClustering(n_clusters=n_clusters, random_state=random_state, affinity='precomputed')
    clusters = spectral.fit(get_tanimoto_matrix(smiles)).labels_

    train_idx, test_idx = [], []
    for cluster in range(n_clusters):

        cluster_idx = np.where(clusters == cluster)[0]
        clust_cliff_mols = [cliff_mols[i] for i in cluster_idx]

        # Can only split stratiefied on cliffs if there are at least 2 cliffs present, else do it randomly
        if sum(clust_cliff_mols) > 2:
            clust_train_idx, clust_test_idx = train_test_split(cluster_idx, test_size=test_size,
                                                               random_state=random_state,
                                                               stratify=clust_cliff_mols, shuffle=True)
        else:
            clust_train_idx, clust_test_idx = train_test_split(cluster_idx, test_size=test_size,
                                                               random_state=random_state,
                                                               shuffle=True)

        train_idx.extend(clust_train_idx)
        test_idx.extend(clust_test_idx)

    train_test = []
    for i in range(len(smiles)):
        if i in train_idx:
            train_test.append('train')
        elif i in test_idx:
            train_test.append('test')
        else:
            raise ValueError(f"Can't find molecule {i} in train or test")

    # Check if there is any intersection between train and test molecules
    assert len(np.intersect1d(train_idx, test_idx)) == 0, 'train and test intersect'
    assert len(np.intersect1d(np.array(smiles)[np.where(np.array(train_test) == 'train')],
                              np.array(smiles)[np.where(np.array(train_test) == 'test')])) == 0, \
        'train and test intersect'

    df_out = pd.DataFrame({'smiles': smiles,
                         'exp_mean [nM]': bioactivity,
                         'y': y_log,
                         'cliff_mol': cliff_mols,
                         'split': train_test})

    return df_out


def process_data(smiles: List[str], bioactivity: List[float], n_clusters: int = 5, test_size: float = 0.2,
                 similarity: float = 0.9, potency_fold: int = 10, remove_stereo: bool = False):
    """ Split data into train/test according to activity cliffs and compounds characteristics.

    :param smiles: (List[str]) list of SMILES strings
    :param bioactivity: (List[float]) list of bioactivity values
    :param n_clusters: (int) number of clusters the data is split into for getting homogeneous data splits
    :param test_size: (float) test split
    :param similarity:  (float) similarity threshold for calculating activity cliffs
    :param potency_fold: (float) potency difference threshold for calculating activity cliffs
    :param remove_stereo: (bool) Remove racemic mixtures altogether?

    :return: df[smiles, exp_mean [nM], y, cliff_mol, split]
    """
    return split_data(smiles, bioactivity, n_clusters, test_size, similarity,  potency_fold, remove_stereo)


def fetch_data(chembl_targetid='CHEMBL2047', endpoints=['EC50']):
    """Download and prep the data from CHEMBL. Throws out duplicates, problematic molecules, and extreme outliers"""
    from MoleculeACE.benchmark.data_fetching import main_curator
    import os

    # fetch + curate data
    data = main_curator.main(chembl_targetid=chembl_targetid, endpoints=endpoints)
    # write to Data directory
    filename = os.path.join('Data', f"{chembl_targetid}_{'_'.join(endpoints)}.csv")
    data.to_csv(filename)


def find_stereochemical_siblings(smiles: List[str]):
    """ Detects molecules that have different SMILES strings, but ecode for the same molecule with
    different stereochemistry. For racemic mixtures it is often unclear which one is measured/active

    Args:
        smiles: (lst) list of SMILES strings

    Returns: (lst) List of SMILES having a similar molecule with different stereochemistry

    """
    from MoleculeACE.benchmark.cliffs import get_tanimoto_matrix

    lower = np.tril(get_tanimoto_matrix(smiles, radius=4, nBits=4096), k=0)
    identical = np.where(lower == 1)
    identical_pairs = [[smiles[identical[0][i]], smiles[identical[1][i]]] for i, j in enumerate(identical[0])]

    return list(set(sum(identical_pairs, [])))


def check_matching(original_smiles, original_bioactivity, smiles, bioactivity):
    assert len(smiles) == len(bioactivity), "length doesn't match"
    for smi, label in zip(original_smiles, original_bioactivity):
        if smi in smiles:
            assert bioactivity[smiles.index(smi)] == label, f"{smi} doesn't match label {label}"

def is_cliff(smiles1, smiles2, y1, y2, similarity: float = 0.9, potency_fold: float = 10):
    """ Calculates if two molecules are activity cliffs """
    sim = moleculeace_similarity([smiles1, smiles2], similarity=similarity)[0][1]
    fc = get_fc([y1, y2])[0][1]

    return sim == 1 and fc >= potency_fold

def check_cliffs(cliffs, n: int = 10):

    # Find the location of 10 random cliffs and check if they are actually cliffs
    m = n
    if np.sum(cliffs.cliffs) < 2*n:
        n = int(np.sum(cliffs.cliffs)/2)

    cliff_loc = np.where(cliffs.cliffs == 1)
    random_cliffs = np.random.randint(0, len(cliff_loc[0]), n)
    cliff_loc = [(cliff_loc[0][c], cliff_loc[1][c]) for c in random_cliffs]

    for i, j in cliff_loc:
        assert is_cliff(cliffs.smiles[i], cliffs.smiles[j], cliffs.bioactivity[i], cliffs.bioactivity[j])

    if len(cliffs.cliffs)-n < m:
        m = len(cliffs.cliffs)-n
    # Find the location of 10 random non-cliffs and check if they are actually non-cliffs
    non_cliff_loc = np.where(cliffs.cliffs == 0)
    random_non_cliffs = np.random.randint(0, len(non_cliff_loc[0]), m)
    non_cliff_loc = [(non_cliff_loc[0][c], non_cliff_loc[1][c]) for c in random_non_cliffs]

    for i, j in non_cliff_loc:
        assert not is_cliff(cliffs.smiles[i], cliffs.smiles[j], cliffs.bioactivity[i], cliffs.bioactivity[j])

# Convertion from SMILES to graph data for GraphDTA
def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na','Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb','Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H','Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr','Cr', 'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6,7,8,9,10]) +
                    [atom.GetIsAromatic()])

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def smiles_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    
    c_size = mol.GetNumAtoms()
    
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append( feature / sum(feature) )

    edges = []
    for bond in mol.GetBonds():
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])
        
    return c_size, features, edge_index


def seq_cat(prot, max_seq_len=1000):
    seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
    seq_dict = {v:(i+1) for i,v in enumerate(seq_voc)}
    x = np.zeros(max_seq_len)
    for i, ch in enumerate(prot[:max_seq_len]): 
        x[i] = seq_dict[ch]
    return x  