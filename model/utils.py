import os
import torch
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
from chemprop.nn_utils import initialize_weights

from utils import get_metric_func
from model.models import KANO_Prot, KANO_Prot_Siams, KANO_Siams
from model.loss import CompositeLoss
from KANO_model.utils import build_optimizer, build_lr_scheduler, build_loss_func


def set_up_model(args, logger):
    assert args.mode in ['train', 'retrain', 'finetune', 'inference']
    if args.train_model == 'KANO_Prot_Siams':
        model = KANO_Prot_Siams(args,
                        classification=True, multiclass=False,
                        multitask=False, prompt=True).to(args.device)
    elif args.train_model == 'KANO_Prot':
        model = KANO_Prot(args,
                        classification=True, multiclass=False,
                        multitask=False, prompt=True).to(args.device)
    elif args.train_model == 'KANO_Siams':
        model = KANO_Siams(args, 
                        classification=True, multiclass=False,
                        multitask=False, prompt=True).to(args.device)
    initialize_weights(model)

    if args.checkpoint_path is not None:
        model.molecule_encoder.load_state_dict(torch.load(args.checkpoint_path, map_location='cpu'), strict=False)
        logger.info('load KANO pretrained model') if args.print else None
    logger.info(f'model: {model}') if args.print else None

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
    
    if args.mode == 'finetune':
        model.load_state_dict(torch.load(os.path.join(args.model_path, f'{args.train_model}_best_model.pt'))['state_dict'])
        logger.info(f'load model from {args.model_path} for finetuning') if args.print else None
    elif args.mode == 'retrain':
        pre_file = torch.load(os.path.join(args.model_path, f'{args.train_model}_model.pt'))
        model.load_state_dict(pre_file['state_dict'])
        logger.info(f'load model from {args.model_path} for retraining') if args.print else None
        optimizer.load_state_dict(pre_file['optimizer'])
        logger.info(f'optimizer: {optimizer}') if args.print else None
        logger.info(f'load optimizer from {args.model_path} for retraining') if args.print else None

        args.epoch = args.epochs - pre_file['epoch']
        args.warmup_epochs = 0
        args.init_lr = optimizer.param_groups[0]['lr']
        scheduler = build_lr_scheduler(optimizer, args)
        logger.info(f'scheduler: {scheduler}') if args.print else None
        logger.info(f'retraining from epoch {pre_file["epoch"]}, {args.epoch} lasting') if args.print else None

    return args, model, optimizer, scheduler, loss_func
        
       
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