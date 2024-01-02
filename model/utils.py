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