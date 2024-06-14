import os
import torch
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
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
from chemprop.nn_utils import initialize_weights

from utils import get_metric_func
from model.models import KANO_Prot, KANO_ESM, KANO_Prot_ablation
from model.loss import CompositeLoss
from KANO_model.model import add_functional_prompt
from KANO_model.utils import build_optimizer, build_lr_scheduler, build_loss_func


def set_up_model(args, logger):
    assert args.mode in ['train', 'retrain', 'finetune', 'inference', 'baseline_inference']
    if args.ablation == 'none':
        if args.train_model == 'KANO_Prot':
            model = KANO_Prot(args,
                            classification=True, multiclass=False,
                            multitask=False, prompt=True).to(args.device)
        elif args.train_model == 'KANO_ESM':
            model = KANO_ESM(args, 
                            classification=True, multiclass=False, 
                            multitask=False, prompt=True).to(args.device)
        initialize_weights(model)
    else:
        model = KANO_Prot_ablation(args, 
                                   classification=True, multiclass=False,
                                   multitask=False, prompt=True).to(args.device)

    if args.checkpoint_path is not None and args.ablation in ['none', 'GCN', 'Attn', 'ESM']:
        model.molecule_encoder.load_state_dict(torch.load(args.checkpoint_path, map_location='cpu'), strict=False)
        logger.info('load KANO pretrained model') if args.print else None
    logger.info(f'model: {model}') if args.print else None

    args.init_lr = 0.0001
    # Optimizers
    optimizer = build_optimizer(model, args)
    logger.info(f'optimizer: {optimizer}') if args.print else None

    # Learning rate schedulers
    scheduler = build_lr_scheduler(optimizer, args)
    logger.info(f'scheduler: {scheduler}') if args.print else None

    # Loss function
    loss_func = CompositeLoss(args, args.loss_func_wt).to(args.device)
    logger.info(f'loss function: {loss_func}, loss weights: {args.loss_func_wt}') if args.print else None

    args.metric_func = get_metric_func(args)
    logger.info(f'metric function: {args.metric_func}') if args.print else None
    args.previous_epoch = 0
    
    if args.mode == 'finetune':
        model.load_state_dict(torch.load(os.path.join(args.model_path,
                                        f'{args.train_model}_best_model.pt'), map_location='cpu')['state_dict'])
        logger.info(f'load model from {args.model_path} for finetuning') if args.print else None
    elif args.mode == 'retrain':
        try:
            pre_file = torch.load(args.save_model_path, map_location='cpu')
        except:
            pre_file = torch.load(args.save_model_path.split('.')[0] + '_ft.pt', map_location='cpu')
        model.load_state_dict(pre_file['state_dict'])
        logger.info(f'load model from {args.save_model_path} for retraining') if args.print else None
        optimizer.load_state_dict(pre_file['optimizer'])
        logger.info(f'optimizer: {optimizer}') if args.print else None
        logger.info(f'load optimizer from {args.save_model_path} for retraining') if args.print else None
        args.previous_epoch = pre_file['epoch']
        logger.info(f'retrain from epoch {args.previous_epoch}, { args.epochs - args.previous_epoch} lasting') if args.print else None
    elif args.mode in ['inference', 'baseline_infernce']:
        model.cpu()
        model.load_state_dict(torch.load(args.save_best_model_path, map_location='cpu')['state_dict'])
        model.to(args.device)
        logger.info(f'load model from {args.save_best_model_path} for inference') if args.print else None
    
    return args, model, optimizer, scheduler, loss_func
        
       
def generate_siamse_smi(data, query_prot_ids, 
                        support_dataset, support_prot,
                        strategy='random', num=1):
    query_smiles, query_labels = np.array(data.smiles()).flatten(), np.array(data.targets()).flatten()
    support_smiles, support_labels = np.array(support_dataset.smiles()).flatten(), \
                                     np.array(support_dataset.targets()).flatten()
    query_prot_ids, support_prot = np.array(query_prot_ids), np.array(support_prot)

    uni_prot = np.unique(np.array(query_prot_ids))
    smiles, label, siam_smiles, siam_label = [], [], [], []
    for prot in tqdm(uni_prot, desc='Generating siamese pairs'):
        q_smiles, q_label = query_smiles[np.where(query_prot_ids == prot)[0]],\
                            query_labels[np.where(query_prot_ids == prot)[0]]
        s_smiles, s_label = support_smiles[support_prot == prot], support_labels[support_prot == prot]
        if strategy == 'random':
            siamse_idx = np.random.choice(len(s_smiles), num*len(s_smiles))
            smiles.extend(np.repeat(q_smiles, num))
            label.extend(np.repeat(q_label, num))
            siam_smiles.extend(s_smiles[siamse_idx])
            siam_label.extend(s_label[siamse_idx])
        elif strategy == 'full':
            smiles.extend(np.repeat(q_smiles, len(s_smiles)))
            label.extend(np.repeat(q_label, len(s_smiles)))
            siam_smiles.extend(np.repeat(s_smiles, len(q_smiles)))
            siam_label.extend(np.repeat(s_label, len(q_smiles)))
        elif strategy == 'TopN_Sim':
            if len(query_prot_ids) == len(support_prot):
                if (query_prot_ids == support_prot).all():
                    siamse_idx, _ = calculate_topk_similarity(q_smiles, s_smiles, top_k=num+1)
                    siamse_idx = siamse_idx[:, 1:].flatten()
            else:
                siamse_idx, _ = calculate_topk_similarity(q_smiles, s_smiles, top_k=num)
                siamse_idx = siamse_idx.flatten()
            smiles.extend(np.repeat(q_smiles, num))
            label.extend(np.repeat(q_label, num))
            siam_smiles.extend(s_smiles[siamse_idx])
            siam_label.extend(s_label[siamse_idx])

    assert len(smiles) == len(siam_smiles)
    return [np.array(smiles), np.array(label)], [np.array(siam_smiles), np.array(siam_label)]


def generate_protein_graph(prot_dict):
    new_edge_funcs = {"edge_construction_functions": [add_peptide_bonds,
                                                        # add_aromatic_interactions,
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


def tanimoto_similarity_matrix(fps1, fps2):
    fp_matrix1 = np.array(fps1)
    fp_matrix2 = np.array(fps2)
    
    dot_product = np.dot(fp_matrix1, fp_matrix2.T)
    
    norm_sq1 = np.sum(fp_matrix1, axis=1)
    norm_sq2 = np.sum(fp_matrix2, axis=1)
    tanimoto_sim = dot_product / (norm_sq1[:, None] + norm_sq2[None, :] - dot_product)
    
    return tanimoto_sim


def calculate_topk_similarity(smiles_list1, smiles_list2, top_k=1):
    """
    Calculate the Tanimoto Similarity between SMILES strings based on ECFP4 fingerprints
    Then, return the indexs with topK similarity
    """
    mols1 = [Chem.MolFromSmiles(smile) for smile in smiles_list1]
    mols2 = [Chem.MolFromSmiles(smile) for smile in smiles_list2]
    # Calculate ECFP4 fingerprints
    fps1 = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) for mol in mols1]
    fps2 = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) for mol in mols2]
    
    fps1_np = np.array([np.frombuffer(fp.ToBitString().encode(), 'u1') - ord('0') for fp in fps1])
    fps2_np = np.array([np.frombuffer(fp.ToBitString().encode(), 'u1') - ord('0') for fp in fps2])

    # Calculate the Tanimoto similarity
    similarity_matrix = tanimoto_similarity_matrix(fps1, fps2)

    # return indexs with topN similarity
    topk_indices = np.argsort(-similarity_matrix, axis=1)[:, :top_k]
    return topk_indices, similarity_matrix
