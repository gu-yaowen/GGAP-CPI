import torch
import random
import numpy as np
from chemprop.data import MoleculeDataset
from chemprop.nn_utils import NoamLR
from chemprop.train.evaluate import evaluate_predictions
from torch_geometric.data import Batch

from model.utils import generate_siamse_smi


def predict_epoch(args, model, data, data_prot, siams_data, scaler, strategy='random'):
    model.eval()
    query_smiles, query_labels = data
    siams_smiles, siams_labels = siams_data

    query_labels = torch.tensor(query_labels).to(args.device)
    siams_labels = torch.tensor(siams_labels).flatten().to(args.device)
    data_prot = np.repeat(data_prot, args.siams_num)
    # data_prot = data_prot[data_idx]

    pred = []
    iter_size = args.batch_size

    for i in range(0, len(query_smiles), iter_size):
        # batch_data = MoleculeDataset(data[i:i + iter_size])
        # smiles, feat, _ = batch_data.smiles(), batch_data.features(), batch_data.targets()
        smiles = query_smiles[i:i + iter_size]
        siams_smiles_, siams_labels_ = siams_smiles[i:i + iter_size], siams_labels[i:i + iter_size]
        prot_ids = data_prot[i:i + iter_size]
        # siams_smiles, siams_labels = generate_siamse_smi(batch_data.smiles(), prot_ids,
        #                                                 support_prot, support_data,
        #                                                 strategy='random', num=args.siams_num)
        # smiles = [smiles_ for smiles_ in smiles for _ in range(args.siams_num)]

        if args.train_model == 'KANO_Siams_Prot':
            prot_feat = [pickle.load(open(f'data/Protein_pretrained_feat/{prot_id}.pkl', 'rb')) 
                         for prot_id in prot_ids]
            prot_feat, prot_graph = [list(d.values())[0][1] for d in prot_feat], \
                                    [list(d.values())[0][-1] for d in prot_feat]
            for i in range(args.batch_size):
                prot_graph[i].x = torch.tensor(prot_feat[i])
            batch_prot = Batch.from_data_list(prot_graph).to(args.device)

        with torch.no_grad():
            if args.train_model == 'KANO_Siams_Prot':
                batch_pred, mol1, mol2, prot, mol_attn = model_prot(smiles, siams_smiles_, batch_prot)
            elif args.train_model == 'KANO_Siams':
                batch_pred, [mol1, mol2] = model(smiles, siams_smiles_)

        # batch_pred = (batch_pred[0].cpu().numpy() + siams_labels).reshape(-1, args.siams_num).mean(axis=1, keepdims=True)
        batch_pred = batch_pred[0].cpu().numpy().flatten() + siams_labels_.cpu().numpy()

        if scaler:
            batch_pred = scaler.inverse_transform(batch_pred)
    
        pred.extend(batch_pred.tolist())
    unique_num = len(set(query_smiles))
    pred = np.array(pred).reshape(unique_num, -1).mean(axis=1, keepdims=True)
    label = np.array(query_labels.cpu().numpy()).reshape(unique_num, -1).mean(axis=1, keepdims=True)
    return pred.tolist(), label.tolist()


def train_epoch(args, model, data, data_prot, siams_data, 
                loss_func, optimizer, scheduler, n_iter):
    model.train()
    query_smiles, query_labels = data
    siams_smiles, siams_labels = siams_data

    data_idx = list(range(len(query_smiles)))
    random.shuffle(data_idx)

    query_smiles, query_labels = query_smiles[data_idx], torch.tensor(query_labels[data_idx]).to(args.device)
    siams_smiles, siams_labels = siams_smiles[data_idx], torch.tensor(siams_labels[data_idx]).flatten().to(args.device)
    data_prot = np.repeat(data_prot, args.siams_num)[data_idx]

    reg_label = (query_labels - siams_labels).view(-1, 1).float().to(args.device)
    cls_label = torch.tensor([1 if abs(label_dis) >= 2 else 0
                                for label_dis in reg_label]).float().to(args.device)

    # data = MoleculeDataset([data[i] for i in data_idx])
    # data_prot = data_prot[data_idx]
    # support_prot, support_data = data_prot, data
    # siams_smi, siams_lab = generate_siamse_smi(data.smiles(), data_prot,
    #                                            support_prot, support_data,
    #                                            strategy='random', num=args.siams_num)

    loss_sum, iter_count = 0, 0
    iter_size = args.batch_size
    loss_collect = {'Total': 0, 'MSE': 0, 'CLS': 0, 'CL': 0}

    for i in range(0, len(query_smiles), iter_size):
        if i + iter_size > len(query_smiles):
            break
        
        # batch_data = MoleculeDataset(data[i:i + iter_size])
        prot_ids = data_prot[i:i + iter_size]

        if args.train_model == 'KANO_Siams_Prot':
            prot_feat = [pickle.load(open(f'data/Protein_pretrained_feat/{prot_id}.pkl', 'rb')) 
                         for prot_id in prot_ids]
            prot_feat, prot_graph = [list(d.values())[0][1] for d in prot_feat], \
                                    [list(d.values())[0][-1] for d in prot_feat]
            for i in range(args.batch_size):
                prot_graph[i].x = torch.tensor(prot_feat[i])
            batch_prot = Batch.from_data_list(prot_graph).to(args.device)

        # smiles, feat, label = batch_data.smiles(), batch_data.features(), batch_data.targets()
        smiles, label = query_smiles[i:i + iter_size], query_labels[i:i + iter_size]
        siams_smiles_, siams_labels_ = siams_smiles[i:i + iter_size], siams_labels[i:i + iter_size]
        reg_label_, cls_label_ = reg_label[i:i + iter_size], cls_label[i:i + iter_size]

        # siams_smiles, siams_labels = siams_smi[i:i + iter_size].flatten().tolist(), \
        #                         siams_lab[i:i + iter_size].flatten.tolist()
        # mask = torch.Tensor([[x is not None for x in tb] for tb in label]).to(args.device)
        # label = torch.Tensor([[0 if x is None else x for x in tb] for tb in label]).to(args.device)

        # label = torch.tensor([label_ for label_ in label 
        #                     for _ in range(args.siams_num)]).view(-1, 1).to(args.device)

        # siams_labels = torch.Tensor(siams_labels).to(args.device)
        # reg_label = label - siams_labels
        # reg_label = reg_label.view(-1, 1).to(args.device)

        # cls_label = torch.tensor([1 if abs(label_dis) >= 2 else 0 
        #                             for label_dis in reg_label], dtype=torch.float32).to(args.device)
        # smiles = [smiles_ for smiles_ in smiles for _ in range(args.siams_num)]

        # if next(model.parameters()).is_cuda:
        #     mask, label = mask.cuda(), label.cuda()

        # class_weights = torch.ones(label.shape)
        # if args.cuda:
        #     class_weights = class_weights.cuda()
            
        model.zero_grad()
        if args.train_model == 'KANO_Siams_Prot':
            pred, mol1, mol2, prot, mol_attn = model_prot(smiles, siams_smiles_, batch_prot)
            loss = loss_func(pred, mol1, mol2, reg_label_, cls_label_)
        elif args.train_model == 'KANO_Siams':
            pred, [mol1, mol2] = model(smiles, siams_smiles_)
            loss = loss_func(pred, [mol1, mol1], [mol2, mol2], reg_label_, cls_label_)

        loss_collect['Total'] += loss[0].item()
        loss_collect['MSE'] += loss[1].item()
        loss_collect['CL'] += loss[2].item()
        loss_collect['CLS'] += loss[3].item()

        iter_count += 1
        loss[0].backward()
        optimizer.step()

        if isinstance(scheduler, NoamLR):
            scheduler.step()
            
        n_iter += len(smiles)
    loss_collect = {k: v / iter_count for k, v in loss_collect.items()}
    return n_iter, loss_collect


def evaluate_epoch(args, model, data, data_prot, siams_data, scaler, strategy='random'):
    pred, label = predict_epoch(args, model, data, data_prot, siams_data, scaler, strategy)

    results = evaluate_predictions(pred, label, args.num_tasks,
                                   args.metric_func, args.dataset_type)

    return results


# def predict_epoch(args, model, data, data_prot, support_data, support_prot, scaler=None):
#     model.eval()

#     pred = []
#     iter_size = args.batch_size

#     for i in range(0, len(data), iter_size):
#         batch_data = MoleculeDataset(data[i:i + iter_size])
    
#         prot_ids = data_prot[i:i + iter_size]
#         siams_smiles, _ = generate_siamse_smi(batch_data.smiles(), args, prot_ids,
#                                               support_prot, support_data, strategy='random')

#         smiles, feat, _ = batch_data.smiles(), batch_data.features(), batch_data.targets()
        
#         with torch.no_grad():
#             batch_pred, _ = model(smiles, siams_smiles)
#             batch_pred = batch_pred[0].cpu().numpy()

#         if scaler:
#             batch_pred = scaler.inverse_transform(batch_pred)
    
#         pred.extend(batch_pred.tolist())
#     return pred


# def train_epoch(args, model, data, data_prot, loss_func, optimizer, scheduler, n_iter):
#     model.train()

#     data_idx = list(range(len(data)))
#     random.shuffle(data_idx)
#     data = [data[i] for i in data_idx]
#     data_prot = data_prot[data_idx]
#     support_prot, support_data = data_prot, data

#     loss_sum, iter_count = 0, 0
#     iter_size = args.batch_size
#     loss_collect = {'Total': 0, 'MSE': 0, 'CLS': 0, 'CL': 0}

#     for i in range(0, len(data), iter_size):
#         if i + iter_size > len(data):
#             break

#         batch_data = MoleculeDataset(data[i:i + iter_size])
#         prot_ids = data_prot[i:i + iter_size]

#         if args.train_model == 'KANO_Siams_Prot':
#             prot_feat = [pickle.load(open(f'data/Protein_pretrained_feat/{prot_id}.pkl', 'rb')) 
#                          for prot_id in prot_ids]
#             prot_feat, prot_graph = [list(d.values())[0][1] for d in prot_feat], \
#                                     [list(d.values())[0][-1] for d in prot_feat]
#             for i in range(args.batch_size):
#                 prot_graph[i].x = torch.tensor(prot_feat[i])
#             batch_prot = Batch.from_data_list(prot_graph).to(args.device)

#         siams_smiles, siams_labels = generate_siamse_smi(batch_data.smiles(), args, prot_ids,
#                                                         support_prot, support_data, strategy='random')

#         smiles, feat, label = batch_data.smiles(), batch_data.features(), batch_data.targets()
#         mask = torch.Tensor([[x is not None for x in tb] for tb in label])
#         label = torch.Tensor([[0 if x is None else x for x in tb] for tb in label])

#         reg_label = torch.tensor(label).to(args.device)
#         cls_label = reg_label - torch.tensor(siams_labels).to(args.device)
#         cls_label = torch.tensor([1 if label_dis >= 2 else 0 
#                                     for label_dis in cls_label], dtype=torch.float32).to(args.device)

#         if next(model.parameters()).is_cuda:
#             mask, label = mask.cuda(), label.cuda()

#         class_weights = torch.ones(label.shape)
#         if args.cuda:
#             class_weights = class_weights.cuda()
            
        # model.zero_grad()
        # if args.train_model == 'KANO_Siams_Prot':
        #     pred, mol1, mol2, prot, mol_attn = model_prot(smiles, siams_smiles, batch_prot)
        #     loss = loss_func(pred, mol1, mol2, reg_label, cls_label)
        # elif args.train_model == 'KANO_Siams':
        #     pred, [mol1, mol2] = model(smiles, siams_smiles)
        #     loss = loss_func(pred, [mol1, mol1], [mol2, mol2], reg_label, cls_label)

        # loss = [(loss_ * class_weights * mask).sum() / mask.sum() for loss_ in loss]
        # if args.dataset_type == 'multiclass':
        #     label = label.long()
        #     loss = torch.cat([loss_func(pred[:, label_idx, :], label[:, label_idx]).unsqueeze(1) 
        #                       for label_idx in range(pred.size(1))], dim=1) * class_weights * mask
        # else:

        # loss = loss_func(pred, [mol1, mol2], [mol1, mol2], reg_label, cls_label)
        # loss = [(loss_ * class_weights * mask).sum() / mask.sum() for loss_ in loss]
        
        # loss_collect['Total'] += loss[0].item()
        # loss_collect['MSE'] += loss[1].item()
        # loss_collect['CLS'] += loss[2].item()
        # loss_collect['CL'] += loss[3].item()

        # iter_count += len(batch_data)

        # loss[0].backward()
        # optimizer.step()

#         if isinstance(scheduler, NoamLR):
#             scheduler.step()
            
#         n_iter += len(batch_data)

#     return n_iter, loss_collect

# def evaluate_epoch(args, model, data, data_prot, support_data, support_prot, scaler):
#     pred = predict_epoch(args, model, data, data_prot, support_data, support_prot, scaler)

#     label = data.targets()

#     results = evaluate_predictions(pred, label, args.num_tasks,
#                                    args.metric_func, args.dataset_type)

#     return results
