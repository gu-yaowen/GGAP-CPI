import os, esm, torch, pickle, molvs, requests, argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from typing import List
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from sklearn.cluster import SpectralClustering
from sklearn.model_selection import train_test_split
from graphein.protein.graphs import construct_graph
from graphein.protein.config import ProteinGraphConfig
from graphein.ml import GraphFormatConvertor
from graphein.protein.utils import download_alphafold_structure
from graphein.protein.edges.distance import (add_peptide_bonds,
                                             add_hydrogen_bond_interactions,
                                             add_disulfide_interactions,
                                             add_ionic_interactions,
                                             add_aromatic_interactions,
                                             add_aromatic_sulphur_interactions,
                                             add_cation_pi_interactions
                                            )
from MoleculeACE.benchmark.cliffs import ActivityCliffs, get_tanimoto_matrix, \
                                        moleculeace_similarity, get_fc
from data_prep import split_data
from utils import set_seed, get_protein_sequence, check_molecule

def extract_sequence_from_pdb(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)

    sequences = []
    for model in structure:
        for chain in model:
            sequence = ''
            for residue in chain:
                sequence += seq1(residue.get_resname())
            sequences.append(sequence)
    return sequences

def generate_protein_graph(df):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    model = model.to(device)
    batch_converter = alphabet.get_batch_converter()
    model.eval()
    # path = 'refined-set'
    target = [i.split('.')[0] for i in df['Uniprot_id'].unique()]
    exist_pdb_file = [i.split('.')[0] for i in os.listdir('data/PDB')]
    exist_file = os.listdir('data/Protein_pretrained_feat')
    for pro in tqdm(target):
        # protein graph residue topology
        if f'{pro}.pkl' in exist_file:
            continue
        else:
            sequence = df[df['Uniprot_id'] == pro]['Sequence'].values[0]
            new_edge_funcs = {"edge_construction_functions": [add_peptide_bonds,
                                                            # add_aromatic_interactions,
                                                            add_hydrogen_bond_interactions,
                                                            add_disulfide_interactions,
                                                            add_ionic_interactions,
                                                            add_aromatic_sulphur_interactions,
                                                            add_cation_pi_interactions]
                                }
            convertor = GraphFormatConvertor(src_format="nx", dst_format="pyg")
            if pro in exist_pdb_file:
                config = ProteinGraphConfig(**new_edge_funcs, verbose=0, 
                                        pdb_path=f'data/PDB/{pro}.pdb')
                g = convertor(construct_graph(config=config, path=f'data/PDB/{pro}.pdb',
                                verbose=False))
            else:
                config = ProteinGraphConfig(**new_edge_funcs)
                g = convertor(construct_graph(config=config, uniprot_id=pro, verbose=False))

            convertor = GraphFormatConvertor(src_format="nx", dst_format="pyg")
            g = convertor(g)

            # protein graph node feature
            prot_data = [(pro, sequence)]
            _, _, batch_tokens = batch_converter(prot_data)
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], return_contacts=True)
            esm_emb = results["representations"][33][0, 1: batch_lens-1].cpu().numpy()

            with open(f'../Protein_pretrained_feat/{pro}.pkl', 'wb') as f:
                pickle.dump({pro: [sequence, esm_emb, g]}, f)
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataset', type=str,
                        help='Dataset file name')
    parser.add_argument('--task', type=str, default='CPI',
                        choices=['CPI', 'QSAR'],
                        help='Task type for your data (CPI or target-specific).'
                             'The data processing could be a little different')
    parser.add_argument('--split', type=str, default='random',
                        choices=['random', 'ac'],
                        help='Data splitting method')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Train ratio to split data into train/test sets')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    args = parser.parse_args()
    args.dataset += '.csv'

    df = pd.read_csv(f'data/{args.dataset}')

    set_seed(args.seed)

    if df['y'].dtype == int or len(df['y'].unique()) == 2:
        df['y'] = df['y'].astype(int)
        df['cliff_mol'] = 0
        task_type = 'classification'
    else:
        df['y'] = df['y'].astype(float)
        task_type = 'regression'
    print(f'Task type: {task_type}')

    if 'exp_mean [nM]' not in df.columns:
        df['exp_mean [nM]'] = 0

    # check ligand structure
    df['smiles'] = df['smiles'].apply(check_molecule)
    len_before = len(df)
    df = df.dropna(subset=['smiles'])
    print(f'{len_before - len(df)} invalid ligands are removed.')

    # check protein structure
    assert 'Uniprot_id' in df.columns, \
        '"Uniprot_id" column not found in the dataset, please provide the Uniprot_id.\n' \
        'If you dont have UniProt IDs but have PDB files, ' \
        'you can fill the "Uniprot_id" column with PDB file name which should be stored in "data/PDB" folder.'
    # get protein sequence
    if 'Sequence' not in df.columns:
        df['Sequence'] = df['Uniprot_id'].map(lambda x: extract_sequence_from_pdb(f'data/PDB/{x}') 
                                            if '.pdb' in x else get_protein_sequence(x))
        
    # get protein graph
    generate_protein_graph(df)

    # data splitting
    df_all = []
    if 'split' not in df.columns or (args.task == 'QSAR' and not os.path.exist(f'data/{args.dataset}')):
        for target in df['Uniprot_id'].unique():
            subset = df[df['Uniprot_id'] == target]
            subset = subset.reset_index(drop=True)
            if args.split == 'random':
                subset['split'] = np.random.choice(['train', 'test'], len(subset), p=[args.train_ratio, 1-args.train_ratio])
            elif args.split == 'ac':
                subset = split_data(subset['smiles'].values.tolist(),
                                    bioactivity=subset['y'].values.tolist(),
                                    in_log10=True, similarity=0.9, test_size=1-args.train_ratio, random_state=args.seed)
            else:
                raise ValueError('Cannot use activity cliff-based splitting for classification tasks.')

            if args.task == 'QSAR':
                if not os.path.exist(f'data/{args.dataset}'):
                    subset.to_csv(f'data/{args.dataset}/{target}.csv', index=False)
                subset.to_csv(f'data/{target}.csv', index=False)
            elif args.task == 'CPI':
                df_all.append(subset)
                df_all.to_csv(f'data/{args.dataset}.csv', index=False)
    
    print('Data processing finished. You can run model training/testing now.')