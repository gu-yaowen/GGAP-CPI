import random
from chemprop.data import MoleculeDataset
import torch
from chemprop.nn_utils import NoamLR
from chemprop.train.evaluate import evaluate_predictions


def predict_epoch(args, model, data, scaler=None):
    model.eval()

    pred = []
    iter_size = args.batch_size

    for i in range(0, len(data), iter_size):
        batch_data = MoleculeDataset(data[i:i + iter_size])
        smiles, feat, _ = batch_data.smiles(), batch_data.features(), batch_data.targets()

        step = 'finetune'
        with torch.no_grad():
            batch_pred = model(step, args.prompt, smiles, feat)
        batch_pred = batch_pred.data.cpu().numpy()

        if scaler:
            batch_pred = scaler.inverse_transform(batch_pred)
    
        pred.extend(batch_pred.tolist())
    return pred


def train_epoch(args, model, data, loss_func, optimizer, scheduler, n_iter):
    model.train()

    data_idx = list(range(len(data)))
    random.shuffle(data_idx)
    data = [data[i] for i in data_idx]

    loss_sum, iter_count = 0, 0
    iter_size = args.batch_size

    for i in range(0, len(data), iter_size):
        if i + iter_size > len(data):
            break

        batch_data = MoleculeDataset(data[i:i + iter_size])
        smiles, feat, label = batch_data.smiles(), batch_data.features(), batch_data.targets()
        mask = torch.Tensor([[x is not None for x in tb] for tb in label])
        label = torch.Tensor([[0 if x is None else x for x in tb] for tb in label])

        if next(model.parameters()).is_cuda:
            mask, label = mask.cuda(), label.cuda()

        class_weights = torch.ones(label.shape)
        if args.cuda:
            class_weights = class_weights.cuda()
            
        model.zero_grad()
        pred = model('finetune', args.prompt, smiles, feat)

        if args.dataset_type == 'multiclass':
            label = label.long()
            loss = torch.cat([loss_func(pred[:, label_idx, :], label[:, label_idx]).unsqueeze(1) 
                              for label_idx in range(pred.size(1))], dim=1) * class_weights * mask
        else:
            loss = loss_func(pred, label) * class_weights * mask
        loss = loss.sum() / mask.sum()

        loss_sum += loss.item()
        iter_count += len(batch_data)

        loss.backward()
        optimizer.step()

        if isinstance(scheduler, NoamLR):
            scheduler.step()
            
        n_iter += len(batch_data)

        # Log and/or add to tensorboard
        # if (n_iter // args.batch_size) % args.log_frequency == 0:
        #     lrs = scheduler.get_lr()
        #     pnorm = compute_pnorm(model)
        #     gnorm = compute_gnorm(model)
        #     loss_avg = loss_sum / iter_count
        #     loss_sum, iter_count = 0, 0
    return n_iter, loss_sum


def evaluate_epoch(args, model, data, scaler):
    pred = predict_epoch(args=args, model=model, data=data, scaler=scaler)

    label = data.targets()

    results = evaluate_predictions(pred, label, args.num_tasks,
                                   args.metric_func, args.dataset_type)

    return results
