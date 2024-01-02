import random
from chemprop.data import MoleculeDataset
import torch
from chemprop.nn_utils import NoamLR
from chemprop.train.evaluate import evaluate_predictions
from utils import get_metric_func

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

    return n_iter, loss_sum


def evaluate_epoch(args, model, data, scaler):
    pred = predict_epoch(args=args, model=model, data=data, scaler=scaler)

    label = data.targets()

    results = evaluate_predictions(pred, label, args.num_tasks,
                                   args.metric_func, args.dataset_type)

    return results


def train_KANO(args):
    args, logger = set_up(args)

    # check in the current task is finished previously, if so, skip
    if os.path.exists(os.path.join(args.save_path, 'KANO_test_pred.csv')):
        logger.info(f'current task {args.data_name} has been finished, skip...')
        return
    
    df, test_idx, train_data, val_data, test_data = process_data_QSAR(args, logger)
    if len(train_data) <= args.batch_size:
        args.batch_size = 64

    if args.features_scaling:
        features_scaler = train_data.normalize_features(replace_nan_token=0)
        val_data.normalize_features(features_scaler)
        test_data.normalize_features(features_scaler)
    else:
        features_scaler = None
    
    if args.dataset_type == 'regression':
        _, train_targets = train_data.smiles(), train_data.targets()
        scaler = StandardScaler().fit(train_targets)
        scaled_targets = scaler.transform(train_targets).tolist()
        train_data.set_targets(scaled_targets)
    else:
        # get class sizes for classification
        # get_class_sizes(data)
        scaler = None

    # load KANO model
    model = build_model(args, encoder_name=args.encoder_name)
    if args.checkpoint_path is not None:
        model.encoder.load_state_dict(torch.load(args.checkpoint_path, map_location='cpu'), strict=False)
    if args.step == 'functional_prompt':
        add_functional_prompt(model, args)
    if args.cuda:
        model = model.cuda()
    logger.info('load KANO model')
    logger.info(f'model: {model}')

    # Optimizers
    optimizer = build_optimizer(model, args)
    logger.info(f'optimizer: {optimizer}')

    # Learning rate schedulers
    args.train_data_size = len(train_data)
    scheduler = build_lr_scheduler(optimizer, args)
    logger.info(f'scheduler: {scheduler}')

    # Loss function
    loss_func = build_loss_func(args)
    logger.info(f'loss function: {loss_func}')

    args.metric_func = get_metric_func(args)
    logger.info(f'metric function: {args.metric_func}')

    n_iter = 0
    args.prompt = False
    metric_dict = set_collect_metric(args)
    best_score = float('inf') if args.minimize_score else -float('inf')
    
    # training
    logger.info(f'training...')
    for epoch in range(args.epochs):
        n_iter, loss = train_epoch(args, model, train_data, loss_func, optimizer, scheduler, n_iter)
        
        if isinstance(scheduler, ExponentialLR):
            scheduler.step()
        if len(val_data) > 0:
            val_scores = evaluate_epoch(args, model, val_data, None)
        else:
            val_scores = evaluate_epoch(args, model, train_data, None)

        test_pred = predict_epoch(args, model, test_data, scaler)
        test_scores = evaluate_predictions(test_pred, test_data.targets(),
                                        args.num_tasks, args.metric_func, args.dataset_type)
        
        logger.info('Epoch : {:02d}, Training Loss : {:.4f}, ' \
                    'Validation score : {:.4f}, Test score : {:.4f}'.format(epoch, loss,
                    list(val_scores.values())[0][0], list(test_scores.values())[0][0]))
        metric_dict = collect_metric_epoch(args, metric_dict, loss, val_scores, test_scores)
        
        # if args.minimize_score and list(val_scores.values())[0][0] < best_score or \
        #         not args.minimize_score and list(val_scores.values())[0][0] > best_score:
        if loss < best_score:
            best_score, best_epoch = list(val_scores.values())[0][-1], epoch
            best_test_score = list(test_scores.values())[0][-1]
            save_checkpoint(os.path.join(args.save_path, 'KANO_model.pt'), model, scaler, features_scaler, args) 
            # logger.info('Best model saved at epoch : {:02d}, Validation score : {:.4f}'.format(best_epoch, best_score))
    logger.info('Final best performed model in {} epoch, val score: {:.4f}, '
                'test score: {:.4f}'.format(best_epoch, best_score, best_test_score))

    # save results
    pickle.dump(metric_dict, open(os.path.join(args.save_path, 'metric_dict.pkl'), 'wb'))
    df['Prediction'] = None
    df.loc[test_idx, 'Prediction'] = test_pred
    df[df['split']=='test'].to_csv(os.path.join(args.save_path, 'KANO_test_pred.csv'), index=False)
    test_data = df[df['split']=='test']
    rmse, rmse_cliff = calc_rmse(test_data['y'].values, test_data['Prediction'].values), \
                       calc_cliff_rmse(y_test_pred=test_data['Prediction'].values,
                                       y_test=test_data['y'].values,
                                       cliff_mols_test=test_data['cliff_mol'].values)
    logger.info('Prediction saved, RMSE: {:.4f}, RMSE_cliff: {:.4f}'.format(rmse, rmse_cliff))

    logger.handlers.clear()
    