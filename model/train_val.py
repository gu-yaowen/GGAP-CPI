import os
import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from chemprop.data import MoleculeDataset, StandardScaler
from chemprop.nn_utils import NoamLR
from chemprop.train.evaluate import evaluate_predictions
from torch_geometric.data import Batch
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.metrics import roc_auc_score, average_precision_score
from model.utils import generate_siamse_smi, generate_label, \
                calc_multiclass, calc_binaryclass
from KANO_model.utils import mol2graph, create_mol_graph


def retrain_scheduler(args, data, optimizer, scheduler, n_iter):
    query_smiles, query_labels, type_id = data
    iter_size = args.batch_size
    for epoch in range(args.previous_epoch):
        for i in tqdm(range(0, len(query_smiles), iter_size)):
            if isinstance(scheduler, NoamLR):
                scheduler.step()
        if isinstance(scheduler, ExponentialLR):
            scheduler.step()
    return scheduler


def predict_epoch(args, model, prot_graph_dict, lig_graph_dict, data, data_prot, scaler):
    model.eval()
    query_smiles, query_labels, type_id = data
    query_labels = torch.tensor(query_labels).float().to(args.device)
    pred = []
    reg_label, cls_label = generate_label(query_labels, type_id, args=args)
    reg_label, cls_label = np.array(reg_label), np.array(cls_label)
    pred_reg, pred_cls = [], []
    iter_size = 512 if 512 < len(query_smiles) else len(query_smiles)
    # iter_size = args.batch_size
    # every 20w data, use mol2graph to generate graph

    num = 50000 // iter_size
    smi_200k = list(set(query_smiles[: iter_size * num if iter_size * num < len(query_smiles) else len(query_smiles)]))
    lig_graph_dict = Parallel(n_jobs=-1)(delayed(create_mol_graph)(smiles, args, args.prompt) for smiles in smi_200k)
    lig_graph_dict = {smi: graph for smi, graph in zip(smi_200k, lig_graph_dict)}

    for i in tqdm(range(0, len(query_smiles), iter_size)):
        smiles = query_smiles[i:i + iter_size]
        prot_ids = data_prot[i:i + iter_size]
        batch_prot = Batch.from_data_list([prot_graph_dict[prot_id] for prot_id in prot_ids]).to(args.device)

        if i % (iter_size * num) == 0:
            smi_200k = list(set(query_smiles[i: i+iter_size * num if i + iter_size * num < len(query_smiles) else len(query_smiles)]))
            lig_graph_dict = Parallel(n_jobs=-1)(delayed(create_mol_graph)(smiles, args, args.prompt) for smiles in smi_200k)
            lig_graph_dict = {smi: graph for smi, graph in zip(smi_200k, lig_graph_dict)}
        if args.graph_input:
            batch_lig = mol2graph([lig_graph_dict[smi] for smi in smiles], args, prompt=False)

        with torch.no_grad():
            if args.graph_input:
                batch_pred, mol1, prot, mol_attn = model(batch_lig, batch_prot)
            else:
                batch_pred, mol1, prot, mol_attn = model(smiles, batch_prot)
            # batch_pred, mol1, prot, mol_attn = model(batch_lig, batch_prot)

        if args.dataset_type == 'regression':
            batch_pred = batch_pred.cpu().numpy().flatten()
            batch_pred = scaler.inverse_transform(batch_pred).tolist()
            pred.extend(batch_pred)
        elif args.dataset_type == 'classification':
            batch_pred = torch.sigmoid(batch_pred).cpu().numpy().flatten().tolist()
            pred.extend(batch_pred)
        elif args.dataset_type == 'joint':
            batch_pred = [batch_pred[0].cpu().numpy().flatten().tolist(), 
                          torch.sigmoid(batch_pred[1]).cpu().numpy().tolist()]
            # batch_pred[0] = scaler.inverse_transform(batch_pred[0])
            pred_reg.extend(batch_pred[0])
            pred_cls += batch_pred[1]
            
    num_data = len(query_smiles)

    if args.dataset_type in ['classification', 'regression']:
        pred = np.array(pred).reshape(num_data, -1).flatten()
        label = np.array(query_labels.cpu().numpy()).reshape(num_data, -1).flatten()
        # if scaler:
        #     label = scaler.inverse_transform(label)
        return np.array(pred.tolist()).reshape(-1, 1), np.array(label.tolist()).reshape(-1, 1)
    else:
        pred_reg = np.array(pred_reg).reshape(num_data, -1).flatten()
        pred_cls = np.array(pred_cls).reshape(num_data, -1)
        if scaler:
            pred_reg = scaler.inverse_transform(pred_reg)
        return [pred_reg.reshape(-1, 1), pred_cls.reshape(-1, 1)], \
                [reg_label.reshape(-1, 1), cls_label.reshape(-1, 1)]


def train_epoch(args, logger, model, prot_graph_dict, lig_graph_dict, data, data_prot,
                loss_func, optimizer, scheduler, n_iter):
    model.train()
    query_smiles, query_labels, type_id = data
    data_idx = list(range(len(query_smiles)))
    random.seed(args.epoch)
    random.shuffle(data_idx)

    query_smiles, query_labels = query_smiles[data_idx], query_labels[data_idx]
    data_prot = data_prot[data_idx]
    query_labels = torch.tensor(query_labels).float().to(args.device)

    query_reg_label, query_cls_label = generate_label(query_labels, type_id, args=args)
    # print(query_cls_label[0])
    if args.dataset_type in ['joint', 'regression']:
        query_reg_label = np.array(query_reg_label).reshape(-1, 1)
        valid_idx = np.where(query_reg_label != 999)[0]
        args.scaler = StandardScaler().fit(query_reg_label[valid_idx])
        query_reg_label[valid_idx] = args.scaler.transform(
                                                query_reg_label[valid_idx])
        if args.dataset_type == 'regression':
            query_reg_label = query_reg_label[valid_idx]
            query_smiles = query_smiles[valid_idx]
            data_prot = data_prot[valid_idx]

    query_cls_label = torch.tensor(query_cls_label).float().to(args.device)
    query_reg_label = torch.tensor(query_reg_label).float().to(args.device)

    loss_sum, iter_count = 0, 0
    iter_size = args.batch_size
    if args.dataset_type == 'regression':
        loss_collect = {'Total': 0, 'MSE': 0}
    elif args.dataset_type == 'classification':
        loss_collect = {'Total': 0, 'AUC': 0, 'AUPR': 0, 'CrossEntropy': 0}
    elif args.dataset_type == 'joint':
        loss_collect = {'Total': 0, 'MSE': 0, 'CrossEntropy': 0}

    # catch up the scheduler to the current iteration
    pred_all, label_all = [], []
    loss_all = [0] if args.dataset_type != 'joint' else [0, 0, 0]

    num = 50000 // args.batch_size
    smi_200k = list(set(query_smiles[: iter_size * num if iter_size * num < len(query_smiles) else len(query_smiles)]))
    lig_graph_dict = Parallel(n_jobs=-1)(delayed(create_mol_graph)(smiles, args, args.prompt) for smiles in smi_200k)
    lig_graph_dict = {smi: graph for smi, graph in zip(smi_200k, lig_graph_dict)}

    for i in tqdm(range(0, len(query_smiles), iter_size)):
        if i + iter_size > len(query_smiles):
            break
        prot_ids = data_prot[i:i + iter_size]
        batch_prot = Batch.from_data_list([prot_graph_dict[prot_id] for prot_id in prot_ids]).to(args.device)        
        smiles, label = query_smiles[i:i + iter_size], query_reg_label[i:i + iter_size]

        if i % (iter_size * num) == 0 and i != 0:
            smi_200k = list(set(query_smiles[i: i+iter_size * num if i + iter_size * num < len(query_smiles) else len(query_smiles)]))
            lig_graph_dict = Parallel(n_jobs=-1)(delayed(create_mol_graph)(smiles, args, args.prompt) for smiles in smi_200k)
            lig_graph_dict = {smi: graph for smi, graph in zip(smi_200k, lig_graph_dict)}

        if args.graph_input:
            batch_lig = mol2graph([lig_graph_dict[smi] for smi in smiles], args, prompt=False)
        if args.dataset_type == 'regression':
            reg_label = label
        elif args.dataset_type == 'classification':
            cls_label = label
        elif args.dataset_type == 'joint':
            reg_label, cls_label = query_reg_label[i:i + iter_size], query_cls_label[i:i + iter_size]
        if len(set(label)) == 1:
            l = set(label)[0]
            print(f'All labels are the same: {l}, skip the iteration!')
            continue
        model.zero_grad()
        
        pred, mol1, prot, mol_attn = model(batch_lig, batch_prot)

        iter_count += 1
        if args.dataset_type == 'regression':
            loss = loss_func(pred, reg_label, None)
            loss_all = [loss_all[0] + loss[0].item()]
        elif args.dataset_type == 'classification':
            loss = loss_func(pred, None, cls_label)
            loss_all = [loss_all[0] + loss[0].item()]
            pred = torch.sigmoid(pred)
            pred_all.extend(pred.detach().cpu().numpy().flatten().tolist())
            label_all.extend(cls_label.detach().cpu().numpy().flatten().tolist())
            if len(set(cls_label.detach().cpu().numpy().flatten().tolist())) == 1:
                logger.info(f'All labels are the same: {label}, skip the iteration!')
                continue
        elif args.dataset_type == 'joint':
            loss = loss_func(pred, reg_label, cls_label)
            loss_all = [loss_all[0] + loss[0].item(),
                        loss_all[1] + loss[1].item(), 
                        loss_all[2] + loss[2].item()]
            reg_pred, cls_pred = pred[0], pred[1]
            pred_all.extend(cls_pred.detach().cpu().numpy().flatten().tolist())
            label_all.extend(cls_label.detach().cpu().numpy().flatten().tolist())

            if len(set(cls_label.detach().cpu().numpy().flatten().tolist())) == 1:
                logger.info(f'All labels are the same: {label}, skip the iteration!')
                continue

        loss[0].backward()
        optimizer.step()

        if isinstance(scheduler, NoamLR):
            scheduler.step()
        n_iter += len(smiles)
    if args.dataset_type == 'regression':
        loss_collect['Total'] += loss_all[0] / iter_count
        loss_collect['MSE'] += loss_all[0] / iter_count
    elif args.dataset_type == 'classification':
        loss_collect['Total'] += loss_all[0] / iter_count
        loss_collect['AUC'] += roc_auc_score(np.array(label_all), np.array(pred_all))
        loss_collect['AUPR'] += average_precision_score(np.array(label_all), np.array(pred_all))
        loss_collect['ACC'] += np.mean(np.array(label_all) == np.array(pred_all).round())
        loss_collect['CrossEntropy'] += loss_all[0] / iter_count
    elif args.dataset_type == 'joint':
        loss_collect['Total'] += loss_all[0] / iter_count
        loss_collect['MSE'] += loss_all[1] / iter_count
        loss_collect['CrossEntropy'] += loss_all[2] / iter_count
    return n_iter, loss_collect


def evaluate_epoch(args, model, prot_graph_dict, lig_graph_dict, data, data_prot, scaler):
    pred, label = predict_epoch(args, model, prot_graph_dict, lig_graph_dict, data, data_prot, scaler)
    if args.dataset_type in ['regression', 'classification']:
        results = evaluate_predictions(pred, label, args.num_tasks,
                                    args.metric_func, args.dataset_type)
    elif args.dataset_type == 'joint':
        pred_reg, pred_cls = pred
        label_reg, label_cls = label
        valid_idx = label_reg != 999
        pred_reg, label_reg = pred_reg[valid_idx], label_reg[valid_idx]
        valid_idx = [i for i in range(len(label_cls)) if label_cls[i][0] != 999]
        pred_cls, label_cls = pred_cls[valid_idx], label_cls[valid_idx]
        reg_func = ['rmse', 'mae', 'r2']
        results_reg = evaluate_predictions(pred_reg.reshape(-1, 1), label_reg.reshape(-1, 1), 1,
                                           reg_func, 'regression')
        cls_func = ['auc', 'aupr', 'acc']
        # results_cls = calc_multiclass(pred_cls, label_cls, cls_func)
        results_cls = calc_binaryclass(pred_cls, label_cls, cls_func)
        results = {**results_reg, **results_cls}
    return results, pred
