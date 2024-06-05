import torch
import random
import pickle
import numpy as np
from tqdm import tqdm
from chemprop.data import MoleculeDataset
from chemprop.nn_utils import NoamLR
from chemprop.train.evaluate import evaluate_predictions
from torch_geometric.data import Batch
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.metrics import roc_auc_score, average_precision_score
from model.utils import generate_siamse_smi


def retrain_scheduler(args, data, optimizer, scheduler, n_iter):
    query_smiles, query_labels = data
    iter_size = args.batch_size
    for epoch in range(args.previous_epoch):
        for i in tqdm(range(0, len(query_smiles), iter_size)):
            if isinstance(scheduler, NoamLR):
                scheduler.step()
        if isinstance(scheduler, ExponentialLR):
            scheduler.step()
    return scheduler


def predict_epoch(args, model, prot_graph_dict, data, data_prot, siams_data, scaler, strategy='random'):
    model.eval()
    query_smiles, query_labels = data
    query_labels = torch.tensor(query_labels).to(args.device)
    pred = []
    iter_size = 256 if 256 < len(query_smiles) else len(query_smiles)

    for i in tqdm(range(0, len(query_smiles), iter_size)):
        smiles = query_smiles[i:i + iter_size]
        prot_ids = data_prot[i:i + iter_size]
        batch_prot = Batch.from_data_list([prot_graph_dict[prot_id] for prot_id in prot_ids]).to(args.device)

        with torch.no_grad():
            batch_pred, mol1, prot, mol_attn = model(smiles, batch_prot)

        if args.dataset_type == 'classification':
            batch_pred = torch.sigmoid(batch_pred[0]).cpu().numpy().flatten()
        else:
            batch_pred = batch_pred[0].cpu().numpy().flatten()

        if scaler:
            batch_pred = scaler.inverse_transform(batch_pred)
        pred.extend(batch_pred.tolist())

    num_data = int(len(query_smiles) / args.siams_num) 
    pred = np.array(pred).reshape(num_data, -1).mean(axis=1, keepdims=True).flatten()
    label = np.array(query_labels.cpu().numpy()).reshape(num_data, -1).mean(axis=1, keepdims=True)
    if scaler:
        label = scaler.inverse_transform(label)
    return pred.tolist(), label.tolist()


def train_epoch(args, model, prot_graph_dict, data, data_prot, siams_data, 
                loss_func, optimizer, scheduler, n_iter):
    model.train()
    query_smiles, query_labels = data
    data_idx = list(range(len(query_smiles)))
    random.seed(0)
    random.shuffle(data_idx)

    query_smiles, query_labels = query_smiles[data_idx], torch.tensor(query_labels[data_idx]).float().to(args.device)
    data_prot = data_prot[data_idx]
    reg_label = query_labels.view(-1, 1).to(args.device)

    loss_sum, iter_count = 0, 0
    iter_size = args.batch_size
    if args.dataset_type == 'regression':
        loss_collect = {'Total': 0, 'MSE': 0, 'CLS': 0, 'CL': 0}
    elif args.dataset_type == 'classification':
        loss_collect = {'AUC': 0, 'AUPR': 0, 'CrossEntropy': 0}

    # catch up the scheduler to the current iteration
    pred_all, label_all = [], []
    loss_all = [0, 0, 0, 0] if args.dataset_type == 'regression' else [0]
    for i in tqdm(range(0, len(query_smiles), iter_size)):
        if i + iter_size > len(query_smiles):
            break
        prot_ids = data_prot[i:i + iter_size]
        batch_prot = Batch.from_data_list([prot_graph_dict[prot_id] for prot_id in prot_ids]).to(args.device)
        smiles, label = query_smiles[i:i + iter_size], query_labels[i:i + iter_size]
        reg_label_ = reg_label[i:i + iter_size]
        if len(set(label)) == 1:
            logger.info(f'All labels are the same: {label}, skip the iteration!')
            continue
        model.zero_grad()

        pred, mol1, prot, mol_attn = model(smiles, batch_prot)
        loss = loss_func(pred, [mol1, None], [None, None], [reg_label_, None, None], None)

        iter_count += 1
        if args.dataset_type == 'regression':
            loss_all = [loss_all[0] + loss[0].item(), loss_all[1] + loss[1].item(),
                        loss_all[2] + loss[2].item(), loss_all[3] + loss[3].item()]
        elif args.dataset_type == 'classification':
            loss_all = [loss_all[0] + loss[0].item()]
            pred = torch.sigmoid(pred[0])
            pred_all.extend(pred.detach().cpu().numpy().flatten().tolist())
            label_all.extend(label.detach().cpu().numpy().flatten().tolist())
        loss[0].backward()
        optimizer.step()

        if isinstance(scheduler, NoamLR):
            scheduler.step()
        n_iter += len(smiles)
    if args.dataset_type == 'regression':
        loss_collect['Total'] += loss_all[0] / iter_count
        loss_collect['MSE'] += loss_all[1] / iter_count
        loss_collect['CL'] += loss_all[2] / iter_count
        loss_collect['CLS'] += loss_all[3] / iter_count
    elif args.dataset_type == 'classification':
        loss_collect['AUC'] += roc_auc_score(np.array(label_all), np.array(pred_all))
        loss_collect['AUPR'] += average_precision_score(np.array(label_all), np.array(pred_all))
        loss_collect['CrossEntropy'] += loss_all[0] / iter_count
    return n_iter, loss_collect


def evaluate_epoch(args, model, prot_graph_dict, data, data_prot, siams_data, scaler, strategy='random'):
    data_idx = list(range(len(data[0])))
    random.shuffle(data_idx)
    data_idx = data_idx[:3000]
    data = [data[0][data_idx], data[1][data_idx]]
    data_prot = data_prot[data_idx]

    pred, label = predict_epoch(args, model, prot_graph_dict, data, data_prot, siams_data, scaler, strategy)
    results = evaluate_predictions(pred, label, args.num_tasks,
                                   args.metric_func, args.dataset_type)

    return results
