"""
Author: Yaowen Gu -- NYU -- 17-10-2023

A collection of data-prepping functions
    - split_data():             split ChEMBL csv into train/test taking similarity and cliffs into account. If you want
                                to process your own data, use this function
    - process_data():           see split_data()
    - load_data():              load a pre-processed dataset from the benchmark
    - fetch_data():             download molecular bioactivity data from ChEMBL for a specific drug target

"""

from MoleculeACE.benchmark.cliffs import ActivityCliffs, get_tanimoto_matrix
from sklearn.cluster import SpectralClustering
from sklearn.model_selection import train_test_split
from chemprop.data import MoleculeDataset
from typing import List
import pandas as pd
import numpy as np
import random
from chemprop.data.utils import get_data, get_task_names
from utils import check_molecule, chembl_to_uniprot, get_protein_sequence
from DeepPurpose.utils import encode_drug, encode_protein


MOLECULEACE_DATALIST = ['CHEMBL1862_Ki', 'CHEMBL1871_Ki', 'CHEMBL2034_Ki', 'CHEMBL2047_EC50',
                        'CHEMBL204_Ki', 'CHEMBL2147_Ki', 'CHEMBL214_Ki', 'CHEMBL218_EC50',
                        'CHEMBL219_Ki', 'CHEMBL228_Ki', 'CHEMBL231_Ki', 'CHEMBL233_Ki',
                        'CHEMBL234_Ki', 'CHEMBL235_EC50', 'CHEMBL236_Ki', 'CHEMBL237_EC50',
                        'CHEMBL237_Ki', 'CHEMBL238_Ki', 'CHEMBL239_EC50', 'CHEMBL244_Ki',
                        'CHEMBL262_Ki', 'CHEMBL264_Ki', 'CHEMBL2835_Ki', 'CHEMBL287_Ki',
                        'CHEMBL2971_Ki', 'CHEMBL3979_EC50', 'CHEMBL4005_Ki', 'CHEMBL4203_Ki',
                        'CHEMBL4616_EC50', 'CHEMBL4792_Ki']
    

def process_data_QSAR(args, logger):
    # check the validity of SMILES
    df = pd.read_csv(args.data_path)
    df[args.smiles_columns] = df[args.smiles_columns].applymap(check_molecule)
    df = df.dropna(subset=args.smiles_columns)

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

    # integrate bioactivity data
    for assay_name in MOLECULEACE_DATALIST:
        df = pd.read_csv(f'data/MoleculeACE/{assay_name}.csv')
        df[args.smiles_columns] = df[args.smiles_columns].applymap(check_molecule)
        df = df.dropna(subset=args.smiles_columns)

        if 'split' not in df.columns and 'cliff_mol' not in df.columns:
            df = split_data(df[args.smiles_columns].values,
                                bioactivity=df[args.target_columns].values,
                                in_log10=True, similarity=0.9, test_size=test_ratio, random_state=args.seed)
            df.to_csv(args.data_path, index=False)
        df['Chembl_id'] = assay_name.split('_')[0]
        df_data = pd.concat([df_data, df])
        chembl_list.append(assay_name.split('_')[0])

    pos_num, neg_num = len(df[df['cliff_mol']==1]), len(df[df['cliff_mol']==0])
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
    chembl_list_2 = df_data['Chembl_id'].unique()
    logger.info('{} are not included in the dataset'.format(set(chembl_list) - set(chembl_list_2)))

    X_drug = df_data['smiles'].values
    X_target = df_data['Sequence'].values
    y = df_data['y'].values
    train_idx, test_idx = list(df_data[df_data['split'].values == 'train'].index), \
                        list(df_data[df_data['split'].values == 'test'].index)
    val_idx = random.sample(list(train_idx), int(len(df_data) * valid_ratio))
    train_idx = list(set(train_idx) - set(val_idx))

    args.cpi_model = 'DeepDTA'
    if args.cpi_model == 'DeepDTA':
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
        logger.info(f'total size: {len(df)}, train size: {len(train_data)}, '
                    f'val size: {len(val_data)}, test size: {len(test_data)}')
    # elif args.cpi_model == 'GraphDTA':
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


def check_cliffs(cliffs, n: int = 10):
    from MoleculeACE.benchmark.cliffs import is_cliff

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