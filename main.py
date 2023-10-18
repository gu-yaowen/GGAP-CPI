import os
import numpy as np
import pandas as pd
from chemprop.data import StandardScaler
from KANO_model.model import build_model, add_functional_prompt
from KANO_model.utils import build_optimizer, build_lr_scheduler, build_loss_func
from data_prep import process_data_QSAR, process_data_CPI
from warnings import simplefilter
import torch
import logging
from chemprop.train.evaluate import evaluate_predictions
from train_val import predict_epoch, train_epoch, evaluate_epoch
from chemprop.train.evaluate import evaluate_predictions
from torch.optim.lr_scheduler import ExponentialLR
from args import add_args
from utils import set_save_path, set_seed, set_collect_metric, \
                  collect_metric_epoch, get_metric_func, save_checkpoint
import DeepPurpose.DTI as models
from DeepPurpose.utils import generate_config
from MoleculeACE.benchmark.utils import Data, calc_rmse, calc_cliff_rmse
from MoleculeACE_baseline import load_MoleculeACE_model
import pickle


def define_logging(args, logger):
    """ Define logging handler.

    :param args: Namespace object
    :param logger: logger object
    """
    handler = logging.FileHandler(os.path.join(args.save_path, 'logs.log'))
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    return


def set_up(args):
    """ Set up arguments, logger, seed, save path.

    :param args: Namespace object
    :return: args, logger
    """
    set_save_path(args)
    logger = logging.getLogger("my_logger")
    logger.setLevel(logging.INFO)
    define_logging(args, logger)

    simplefilter(action='ignore', category=Warning)
    logger.info(f'current task: {args.data_name}')
    logger.info(f'arguments: {args}')

    set_seed(args.seed)
    logger.info(f'random seed: {args.seed}')
    logger.info(f'save path: {args.save_path}')
    return args, logger


def train_main(args):
    args, logger = set_up(args)

    df, test_idx, train_data, val_data, test_data = process_data_QSAR(args, logger)

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
        if args.val:
            val_scores = evaluate_epoch(args, model, val_data, scaler)
        else:
            val_scores = evaluate_epoch(args, model, train_data, scaler)

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
            save_checkpoint(os.path.join(args.save_path, 'model.pt'), model, scaler, features_scaler, args) 
            # logger.info('Best model saved at epoch : {:02d}, Validation score : {:.4f}'.format(best_epoch, best_score))
    logger.info('Final best performed model in {} epoch, val score: {:.4f}, '
                'test score: {:.4f}'.format(best_epoch, best_score, best_test_score))

    # save results
    pickle.dump(metric_dict, open(os.path.join(args.save_path, 'metric_dict.pkl'), 'wb'))
    df['Prediction'] = None
    df.loc[test_idx, 'Prediction'] = test_pred
    df[df['split']=='test'].to_csv(os.path.join(args.save_path, 'test_pred.csv'), index=False)
    rmse, rmse_cliff = calc_rmse(test_data['y'], test_data['Prediction']), \
                       calc_cliff_rmse(y_test_pred=test_data['Prediction'], y_test=test_data['y'],
                                       cliff_mols_test=test_data['cliff_mol'])
    logger.info('Prediction saved, RMSE: {:.4f}, RMSE_cliff: {:.4f}'.format(rmse, rmse_cliff))

    logger.handlers.clear()
    return

def predict_main(args):
    return

def run_baseline_QSAR(args):
    args, logger = set_up(args)
    
    logger.info(f'current task: {args.data_name}')

    # Note: as the Data class in Molecule ACE directly extracts split index from the original dataset, 
    # it is highly recommended to run KANO first to keep consistency between the baseline.
    data = Data(args.data_name)
    descriptor, model = load_MoleculeACE_model(args, logger)

    # Data augmentation for Sequence-based models
    if args.baseline_model in ['CNN', 'LSTM', 'Transformer']:
        AUGMENTATION_FACTOR = 10
        data.augment(AUGMENTATION_FACTOR)
        data.shuffle()

    data(descriptor)

    logger.info(f'training {args.baseline_model}...')
    model.train(data.x_train, data.y_train)

    # save model
    model_save_path = os.path.join(args.save_path, f'{args.baseline_model}_model.pkl')
    model_save_path = model_save_path.replace(
                        '.pkl','.h5') if args.baseline_model is 'LSTM' else model_save_path
    with open(model_save_path, 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    preds = model.predict(data.x_test)
    # collect test data
    df_test = pd.DataFrame()
    df_test['smiles'] = data.smiles_test
    df_test['y'] = data.y_test
    df_test['cliff_mol'] = data.cliff_mols_test
    df_test['Prediction'] = preds

    rmse = calc_rmse(df_test['y'].values, df_test['Prediction'].values)
    rmse_cliff = calc_cliff_rmse(y_test_pred=df_test['Prediction'].values,
                                 y_test=df_test['y'].values,
                                 cliff_mols_test=df_test['cliff_mol'].values)
    df_test.to_csv(os.path.join(args.save_path, f'{args.baseline_model}_test_pred.csv'), index=False)
    logger.info(f'Prediction saved, RMSE: {rmse:.4f}, RMSE_cliff: {rmse_cliff:.4f}')

    logger.handlers.clear()
    return

def run_baseline_CPI(args):
    args, logger = set_up(args)

    df_all, test_idx, train_data, val_data, test_data = process_data_CPI(args, logger)

    if args.baseline_model == 'DeepDTA':
        drug_encoding = 'CNN' 
        target_encoding = 'CNN'
        # Note: the hyperparameters are reported as the best performing ones in DeepPurpose
        # for the KIBA and DAVIS dataset
        config = generate_config(drug_encoding = drug_encoding, 
                            target_encoding = target_encoding, 
                            cls_hidden_dims = [1024,1024,512], 
                            train_epoch = 100, 
                            LR = 0.001, 
                            batch_size = 256,
                            cnn_drug_filters = [32,64,96],
                            cnn_target_filters = [32,64,96],
                            cnn_drug_kernels = [4,6,8],
                            cnn_target_kernels = [4,8,12]
                            )
        model = models.model_initialize(**config)
        logger.info('load DeepDTA model from DeepPurpose')
        logger.info(f'model: {model}') 
        model = models.model_initialize(**config)

        model.train(train_data, val_data, test_data)

        # get predictions
        test_pred = model.predict(test_data)
        save_checkpoint(os.path.join(args.save_path,f'{args.baseline_model}_model.pt'),
                        model=model, args=args) 

        test_data_all = df_all[df_all['split']=='test']
        test_data['UniProt_id'] = test_data_all['UniProt_id'].values
        test_data['cliff_mol'] = test_data_all['cliff_mol'].values

        if 'Chembl_id' in test_data_all.columns:
            test_data['Chembl_id'] = test_data_all['Chembl_id'].values
            task = test_data_all['Chembl_id'].unique()[0]
        else:
            task = test_data_all['UniProt_id'].unique()[0]

        test_data['Prediction'] = test_pred
        test_data = test_data.rename(columns={'Label': 'y'})
        test_data.to_csv(os.path.join(args.save_path, f'{args.baseline_model}_test_pred.csv'), index=False)
        rmse, rmse_cliff = [], []

        for target in task:
            test_data_target = test_data[test_data['UniProt_id']==target]
            rmse.append(calc_rmse(test_data_target['y'], test_data_target['Prediction']))
            rmse_cliff.append(calc_cliff_rmse(y_test_pred=test_data_target['Prediction'], y_test=test_data_target['y'],
                                            cliff_mols_test=test_data_target['cliff_mol']))
        logger.info('Prediction saved, RMSE: {:.4f}±{:.4f}, '
                    'RMSE_cliff: {:.4f}±{:.4f}'.format(np.mean(rmse), np.std(rmse),
                                                       np.mean(rmse_cliff), np.std(rmse_cliff)))

        logger.handlers.clear()        
    return

if __name__ == '__main__':
    args = add_args()

    if args.mode == 'train':
        train_main(args)
    elif args.mode == 'inference':
        predict_main(args)
    elif args.mode == 'baseline_QSAR':
        run_baseline_QSAR(args)
    elif args.mode == 'baseline_CPI':
        run_baseline_CPI(args)