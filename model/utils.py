import numpy as np
from graphein.protein.graphs import construct_graph
from graphein.protein.config import ProteinGraphConfig
from graphein.ml import GraphFormatConvertor
from graphein.protein.edges.distance import (add_peptide_bonds,
                                             add_hydrogen_bond_interactions,
                                             add_disulfide_interactions,
                                             add_ionic_interactions,
                                             add_aromatic_interactions,
                                             add_aromatic_sulphur_interactions,
                                             add_cation_pi_interactions
                                            )

def generate_siamse_smi(data, query_prot_ids, 
                        support_dataset, support_prot,
                        strategy='random', num=1):
    smiles, labels = data.smiles(), data.targets()
    support_smiles, support_labels = np.array(support_dataset.smiles()).flatten(), \
                                     np.array(support_dataset.targets()).flatten()
    # repeat
    if strategy == 'random':
        smiles_rep = np.repeat(smiles, num)
        labels_rep = np.repeat(labels, num)
    elif strategy == 'full':
        smiles_rep = np.repeat(smiles, len(support_smiles))
        labels_rep = np.repeat(labels, len(support_smiles))
    # sampling from support set
    siam_smiles, siam_labels = [], []
    for idx, smi in enumerate(smiles):
        prot_id = query_prot_ids[idx]
        support_idx = np.where(support_prot == prot_id)[0]
        if strategy == 'random':
            siamse_idx = np.random.choice(support_idx, num)
        elif strategy == 'full':
            siamse_idx = support_idx
        siam_smiles.extend([support_smiles[idx] for idx in list(siamse_idx)])
        siam_labels.extend([support_labels[idx] for idx in list(siamse_idx)])

    assert len(smiles_rep) == len(siam_smiles)
    return [np.array(smiles_rep), np.array(labels_rep)], [np.array(siam_smiles), np.array(siam_labels)]


def generate_protein_graph(prot_dict):
    new_edge_funcs = {"edge_construction_functions": [add_peptide_bonds,
                                                        add_aromatic_interactions,
                                                        add_hydrogen_bond_interactions,
                                                        add_disulfide_interactions,
                                                        add_ionic_interactions,
                                                        add_aromatic_sulphur_interactions,
                                                        add_cation_pi_interactions]
                        }
    config = ProteinGraphConfig(**new_edge_funcs)
    convertor = GraphFormatConvertor(src_format="nx", dst_format="pyg")
    
    for uni_id in (prot_dict.keys()):
        uni = uni_id.split('_')[0]
        try:
            g = construct_graph(config=config, uniprot_id=uni, verbose=False)
            prot_dict[uni_id] = prot_dict[uni_id] + [convertor(g)]
        except:
            logger.info('No PDB ID, try using AlphaFold2 predicted structure')
            try:
                fp = download_alphafold_structure(uni, aligned_score=False)
                g = construct_graph(config=config, path=fp, verbose=False)
                prot_dict[uni_id] = prot_dict[uni_id] + [convertor(g)]
            except:
                logger.info('No AlphaFold2 predicted structure found!!')
                prot_dict[uni_id] = prot_dict[uni_id] + [None]
            pass
    return prot_dict