import torch
import random
from chemprop.data import MoleculeDataset
from chemprop.nn_utils import NoamLR
from chemprop.train.evaluate import evaluate_predictions

from model.utils import generate_siamse_smi


def predict_epoch(args, model, data, data_prot, support_data, support_prot, scaler=None):
    model.eval()

    pred = []
    iter_size = args.batch_size

    for i in range(0, len(data), iter_size):
        batch_data = MoleculeDataset(data[i:i + iter_size])
    
        prot_ids = data_prot[i:i + iter_size]
        siams_smiles, _ = generate_siamse_smi(batch_data.smiles(), args, prot_ids,
                                              support_prot, support_data, strategy='random')

        smiles, feat, _ = batch_data.smiles(), batch_data.features(), batch_data.targets()
        
        with torch.no_grad():
            batch_pred, _ = model(smiles, siams_smiles)
            batch_pred = batch_pred[0].cpu().numpy()

        if scaler:
            batch_pred = scaler.inverse_transform(batch_pred)
    
        pred.extend(batch_pred.tolist())
    return pred


def train_epoch(args, model, data, data_prot, loss_func, optimizer, scheduler, n_iter):
    model.train()

    data_idx = list(range(len(data)))
    random.shuffle(data_idx)
    data = [data[i] for i in data_idx]
    data_prot = data_prot[data_idx]
    support_prot, support_data = data_prot, data

    loss_sum, iter_count = 0, 0
    iter_size = args.batch_size
    loss_collect = {'Total': 0, 'MSE': 0, 'CLS': 0, 'CL': 0}

    for i in range(0, len(data), iter_size):
        if i + iter_size > len(data):
            break
        
        batch_data = MoleculeDataset(data[i:i + iter_size])
        prot_ids = data_prot[i:i + iter_size]
        
        siams_smiles, siams_labels = generate_siamse_smi(batch_data.smiles(), args, prot_ids,
                                                        support_prot, support_data, strategy='random')

        smiles, feat, label = batch_data.smiles(), batch_data.features(), batch_data.targets()
        mask = torch.Tensor([[x is not None for x in tb] for tb in label])
        label = torch.Tensor([[0 if x is None else x for x in tb] for tb in label])

        reg_label = torch.tensor(label).to(args.device)
        cls_label = reg_label - torch.tensor(siams_labels).to(args.device)
        cls_label = torch.tensor([1 if label_dis >= 2 else 0 
                                    for label_dis in cls_label], dtype=torch.float32).to(args.device)

        if next(model.parameters()).is_cuda:
            mask, label = mask.cuda(), label.cuda()

        class_weights = torch.ones(label.shape)
        if args.cuda:
            class_weights = class_weights.cuda()
            
        model.zero_grad()
        pred, [mol1, mol2] = model(smiles, siams_smiles)

        # if args.dataset_type == 'multiclass':
        #     label = label.long()
        #     loss = torch.cat([loss_func(pred[:, label_idx, :], label[:, label_idx]).unsqueeze(1) 
        #                       for label_idx in range(pred.size(1))], dim=1) * class_weights * mask
        # else:

        loss = loss_func(pred, [mol1, mol2], [mol1, mol2], reg_label, cls_label)
        loss = [i / mask.sum() for i in loss]

        loss_collect['Total'] += loss[0].item()
        loss_collect['MSE'] += loss[1].item()
        loss_collect['CLS'] += loss[2].item()
        loss_collect['CL'] += loss[3].item()

        iter_count += len(batch_data)

        loss[0].backward()
        optimizer.step()

        if isinstance(scheduler, NoamLR):
            scheduler.step()
            
        n_iter += len(batch_data)

    return n_iter, loss_collect

def evaluate_epoch(args, model, data, data_prot, support_data, support_prot, scaler):
    pred = predict_epoch(args, model, data, data_prot, support_data, support_prot, scaler)

    label = data.targets()

    results = evaluate_predictions(pred, label, args.num_tasks,
                                   args.metric_func, args.dataset_type)

    return results
