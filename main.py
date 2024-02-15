import os
import torch
import pickle
import random
import numpy as np
import pandas as pd
from chemprop.data import StandardScaler, MoleculeDataset
from chemprop.data.utils import get_data
from chemprop.train.evaluate import evaluate_predictions
from torch.optim.lr_scheduler import ExponentialLR
from MoleculeACE.benchmark.utils import Data, calc_rmse, calc_cliff_rmse

from args import add_args
from data_prep import process_data_QSAR, process_data_CPI
from utils import set_save_path, set_seed, set_collect_metric, \
                  collect_metric_epoch, save_checkpoint, \
                  define_logging, set_up
from model.train_val import retrain_scheduler, train_epoch, evaluate_epoch, predict_epoch
from model.utils import generate_siamse_smi, set_up_model


def run_QSAR(args):
    args, logger = set_up(args)
    
    # check in the current task is finished previously, if so, skip
    if os.path.exists(os.path.join(args.save_path, f'{args.baseline_model}_test_pred.csv')):
        logger.info(f'current task {args.data_name} for '
                    f'model {args.baseline_model} has been finished, skip...') if args.print else None
        return

    logger.info(f'current task: {args.data_name}') if args.print else None

    df, test_idx, train_data, val_data, test_data = process_data_QSAR(args, logger)

    data = get_data(path=args.data_path, 
                    smiles_columns=args.smiles_columns,
                    target_columns=args.target_columns,
                    ignore_columns=args.ignore_columns)

    if args.split_sizes:
        _, valid_ratio, test_ratio = args.split_sizes
        train_idx, test_idx = df[df['split']=='train'].index, df[df['split']=='test'].index
        val_idx = random.sample(list(train_idx), int(len(df) * valid_ratio))
        train_idx = list(set(train_idx) - set(val_idx))

    train_prot, val_prot, test_prot = np.array([args.data_name for i in range(len(train_idx))]), \
                                    np.array([args.data_name for i in range(len(val_idx))]), \
                                    np.array([args.data_name for i in range(len(test_idx))]) 
    train_data, val_data, test_data = tuple([[data[i] for i in train_idx],
                                            [data[i] for i in val_idx],
                                            [data[i] for i in test_idx]])
    train_data, val_data, test_data = MoleculeDataset(train_data), \
                                        MoleculeDataset(val_data), \
                                        MoleculeDataset(test_data)

    if len(train_data) * args.siams_num <= args.batch_size:
        args.batch_size = 64
        logger.info(f'batch size is too large, reset to {args.batch_size}') if args.print else None

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

    # load model, optimizer, scheduler, loss function
    args.train_data_size = len(train_data)
    args, model, optimizer, scheduler, loss_func = set_up_model(args, logger)

    n_iter = 0
    metric_dict = set_collect_metric(args)
    best_score = float('inf') if args.minimize_score else -float('inf')

    # process data for siamese pairs
    logger.info(f'generating siamese pairs...') if args.print else None
    query_train, siams_train = generate_siamse_smi(train_data, train_prot,
                                                  train_data, train_prot,
                                                  strategy='full', num=args.siams_num)
    if len(val_data) > 0:
        query_val, siams_val = generate_siamse_smi(val_data, val_prot,
                                                train_data, train_prot,
                                                strategy='full', num=args.siams_num)
    else:
        query_val, siams_val = generate_siamse_smi(train_data, train_prot,
                                                train_data, train_prot,
                                                strategy='random', num=args.siams_num)       
    query_test, siams_test = generate_siamse_smi(test_data, test_prot,
                                                train_data, train_prot,
                                                strategy='full', num=args.siams_num)
    logger.info(f'generated siamese data size: train {len(query_train[0])}, '
                f'val {len(query_val[0])}, test {len(query_test[0])}') if args.print else None

    # training
    logger.info(f'training...') if args.print else None
    best_loss = 999
    for epoch in range(args.epochs):
        n_iter, loss_collect = train_epoch(args, model, query_train, train_prot, siams_train, 
                                           loss_func, optimizer, scheduler, n_iter)
        if isinstance(scheduler, ExponentialLR):
            scheduler.step()

        val_scores = evaluate_epoch(args, model, query_val, val_prot, siams_val, None)
        
        test_pred, _ = predict_epoch(args, model, query_test, test_prot, siams_test, scaler)
        test_scores = evaluate_predictions(test_pred, test_data.targets(),
                                           args.num_tasks, args.metric_func, args.dataset_type)

        logger.info('Epoch : {:02d}, Loss_Total: {:.3f}, Loss_MSE: {:.3f}, Loss_CLS: {:.3f}, Loss_CL: {:.3f}, ' \
                    'Validation score : {:.3f}, Test score : {:.3f}'.format(epoch, 
                    loss_collect['Total'], loss_collect['MSE'], loss_collect['CLS'], loss_collect['CL'],
                    list(val_scores.values())[0][0], list(test_scores.values())[0][0])) if args.print else None
        metric_dict = collect_metric_epoch(args, metric_dict, loss_collect, val_scores, test_scores, 
                                           model, epoch, optimizer, scaler, features_scaler, args)
        if epoch < args.epochs - 1:
            save_checkpoint(args.save_model_path, 
                            model, scaler, features_scaler, epoch, optimizer, args) 
        if loss_collect['MSE'] < best_loss or epoch == 0:
            best_loss = loss_collect['MSE']
            best_score, best_epoch = list(val_scores.values())[0][-1], epoch
            best_test_score = list(test_scores.values())[0][-1]
            save_checkpoint(args.save_best_model_path, 
                            model, scaler, features_scaler, epoch, optimizer, args) 
    logger.info('Final best performed model in {} epoch, val score: {:.4f}, '
                'test score: {:.4f}'.format(best_epoch, best_score, best_test_score)) if args.print else None

    # test the best model
    model.load_state_dict(torch.load(args.save_best_model_path)['state_dict'])
    test_pred, _ = predict_epoch(args, model, query_test, test_prot, siams_test, scaler, strategy='full')

    # save results
    pickle.dump(metric_dict, open(args.save_metric_path, 'wb'))
    df['Prediction'] = None
    df.loc[test_idx, 'Prediction'] = test_pred
    df[df['split']=='test'].to_csv(args.save_pred_path, index=False)
    test_data = df[df['split']=='test']
    rmse, rmse_cliff = calc_rmse(test_data['y'].values, test_data['Prediction'].values), \
                       calc_cliff_rmse(y_test_pred=test_data['Prediction'].values,
                                       y_test=test_data['y'].values,
                                       cliff_mols_test=test_data['cliff_mol'].values)
    logger.info('Prediction saved, RMSE: {:.4f}, RMSE_cliff: {:.4f}'.format(rmse, rmse_cliff)) if args.print else None

    logger.handlers.clear()
    return


def run_CPI(args):
    args, logger = set_up(args)

    df_all, test_idx, train_data, val_data, test_data = process_data_CPI(args, logger)

    data = get_data(path=args.data_path, 
                    smiles_columns=args.smiles_columns,
                    target_columns=args.target_columns,
                    ignore_columns=args.ignore_columns)

    if args.split_sizes:
        _, valid_ratio, test_ratio = args.split_sizes
        train_idx, test_idx = df_all[df_all['split']=='train'].index, df_all[df_all['split']=='test'].index
        val_idx = random.sample(list(train_idx), int(len(df_all) * valid_ratio))
        train_idx = list(set(train_idx) - set(val_idx))

    train_prot, val_prot, test_prot = df_all.loc[train_idx, 'Uniprot_id'].values, \
                                      df_all.loc[val_idx, 'Uniprot_id'].values, \
                                      df_all.loc[test_idx, 'Uniprot_id'].values

    train_data, val_data, test_data = tuple([[data[i] for i in train_idx],
                                            [data[i] for i in val_idx],
                                            [data[i] for i in test_idx]])
    train_data, val_data, test_data = MoleculeDataset(train_data), \
                                      MoleculeDataset(val_data), \
                                      MoleculeDataset(test_data)

    if len(train_data) * args.siams_num <= args.batch_size:
        args.batch_size = 64
        logger.info(f'batch size is too large, reset to {args.batch_size}') if args.print else None

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
        # scaler = None
    else:
        # get class sizes for classification
        # get_class_sizes(data)
        scaler = None
    
    # load model, optimizer, scheduler, loss function
    args.train_data_size = len(train_data)
    args, model, optimizer, scheduler, loss_func = set_up_model(args, logger)

    n_iter = 0
    metric_dict = set_collect_metric(args)
    best_score = float('inf') if args.minimize_score else -float('inf')

    if args.train_model == 'KANO_Prot_Siams':
        # process data for siamese pairs
        logger.info(f'generating siamese pairs...') if args.print else None
        query_train, siams_train = generate_siamse_smi(train_data, train_prot,
                                                    train_data, train_prot,
                                                    strategy='TopN_Sim', num=args.siams_num)
        if len(val_data) > 0:
            query_val, siams_val = generate_siamse_smi(val_data, val_prot,
                                                    train_data, train_prot,
                                                    strategy='TopN_Sim', num=args.siams_num)
        else:
            query_val, siams_val = query_train, siams_train
            val_prot = train_prot     

        query_test, siams_test = generate_siamse_smi(test_data, test_prot,
                                                    train_data, train_prot,
                                                    strategy='TopN_Sim', num=args.siams_num)        
        logger.info(f'generated siamese data size: train {len(query_train[0])}, '
                    f'val {len(query_val[0])}, test {len(query_test[0])}') if args.print else None
    else:
        query_train, siams_train = [np.array(train_data.smiles()).flatten(), 
                                    np.array(train_data.targets()).flatten()], None
        if len(val_data) > 0:
            query_val, siams_val = [np.array(val_data.smiles()).flatten(), 
                                    np.array(val_data.targets()).flatten()], None
        else:
            query_val, siams_val = query_train, siams_train
            val_prot = train_prot
            # scaler = None
        query_test, siams_test = [np.array(test_data.smiles()).flatten(),
                                  np.array(test_data.targets()).flatten()], None
    
    # load protein features
    if args.train_model in ['KANO_Prot', 'KANO_Prot_Siams']:
        logger.info(f'loading protein features...') if args.print else None
        prot_list = df_all['Uniprot_id'].unique()
        prot_feat = [pickle.load(open(f'data/Protein_pretrained_feat/{prot_id}.pkl', 'rb')) 
                    for prot_id in prot_list]
        prot_feat, prot_graph = [list(d.values())[0][1] for d in prot_feat], \
                                        [list(d.values())[0][-1] for d in prot_feat]
        for i in range(len(prot_graph)):
            try:
                prot_graph[i].x = torch.tensor(prot_feat[i][: prot_graph[i].num_nodes]).to(args.device)
                if prot_graph[i].x.shape[0] < prot_graph[i].num_nodes:
                    prot_graph[i].x = torch.cat([prot_graph[i].x,
                                                torch.zeros(prot_graph[i].num_nodes - prot_graph[i].x.shape[0],
                                                            prot_graph[i].x.shape[1]).to(args.device)])
            except:
                print(prot_graph[i])

        prot_graph_dict = dict(zip(prot_list, prot_graph))
        del prot_feat, prot_graph
    else:
        prot_graph_dict = None

    # training
    logger.info(f'training...') if args.print else None
    best_loss = 999
    if args.mode == 'retrain':
        logger.info(f'retraining...') if args.print else None
        scheduler = retrain_scheduler(args, query_train, optimizer, scheduler, n_iter)

    for epoch in range(args.epochs):
        n_iter, loss_collect = train_epoch(args, model, prot_graph_dict, query_train, train_prot, siams_train, 
                                           loss_func, optimizer, scheduler, n_iter)
        if isinstance(scheduler, ExponentialLR):
            scheduler.step()

        # val_scores = evaluate_epoch(args, model, prot_graph_dict, query_val, val_prot,
        #                             siams_val, scaler if len(val_data) > 0 else None)
        val_scores = evaluate_epoch(args, model, prot_graph_dict, query_val, val_prot,
                                    siams_val, scaler)

        test_pred, _ = predict_epoch(args, model, prot_graph_dict, query_test, test_prot, siams_test, scaler)

        test_scores = evaluate_predictions(test_pred, test_data.targets(),
                                           args.num_tasks, args.metric_func, args.dataset_type)

        logger.info('Epoch : {:02d}, Loss_Total: {:.3f}, Loss_MSE: {:.3f}, Loss_CLS: {:.3f}, Loss_CL: {:.3f}, ' \
                    'Validation score : {:.3f}, Test score : {:.3f}'.format(epoch, 
                    loss_collect['Total'], loss_collect['MSE'], loss_collect['CLS'], loss_collect['CL'],
                    list(val_scores.values())[0][0], list(test_scores.values())[0][0])) if args.print else None
        metric_dict = collect_metric_epoch(args, metric_dict, loss_collect, val_scores, test_scores)
        
        if epoch < args.epochs - 1:
            save_checkpoint(args.save_model_path, 
                            model, scaler, features_scaler, epoch, optimizer, args)
        if loss_collect['MSE'] < best_loss or epoch == 0:
            best_loss = loss_collect['MSE']
            best_score, best_epoch = list(val_scores.values())[0][-1], epoch
            best_test_score = list(test_scores.values())[0][-1]
            save_checkpoint(args.save_best_model_path, 
                            model, scaler, features_scaler, epoch, optimizer, args) 
    logger.info('Final best performed model in {} epoch, val score: {:.4f}, '
                'test score: {:.4f}'.format(best_epoch, best_score, best_test_score)) if args.print else None

    # test the best model
    model.load_state_dict(torch.load(args.save_best_model_path)['state_dict'])
    test_pred, _ = predict_epoch(args, model, prot_graph_dict, query_test, test_prot, siams_test, scaler, strategy='full')

    # save results
    pickle.dump(metric_dict, open(args.save_metric_path, 'wb'))

    test_data_all = df_all[df_all['split']=='test']

    if 'Chembl_id' in test_data_all.columns:
        test_data_all['Chembl_id'] = test_data_all['Chembl_id'].values
        task = test_data_all['Chembl_id'].unique()
    else:
        task = test_data_all['Uniprot_id'].unique()
    
    test_data_all['Prediction'] = np.array(test_pred).flatten()[:len(test_data_all)] # some baseline may have padding, delete the exceeds
    test_data_all = test_data_all.rename(columns={'Label': 'y'})
    test_data_all.to_csv(args.save_pred_path, index=False)
    rmse = calc_rmse(test_data_all['y'].values, test_data_all['Prediction'].values)
    rmse_cliff = calc_cliff_rmse(y_test_pred=test_data_all['Prediction'].values,
                                 y_test=test_data_all['y'].values,
                                 cliff_mols_test=test_data_all['cliff_mol'].values)
    # for target in task:
    #     if 'Chembl_id' in test_data_all.columns:
    #         test_data_target = test_data_all[test_data_all['Chembl_id']==target]
    #     else:
    #         test_data_target = test_data_all[test_data_all['Uniprot_id']==target]
    #     rmse.append(calc_rmse(test_data_target['y'].values, test_data_target['Prediction'].values))
    #     rmse_cliff.append(calc_cliff_rmse(y_test_pred=test_data_target['Prediction'].values,
    #                                       y_test=test_data_target['y'].values,
    #                                     cliff_mols_test=test_data_target['cliff_mol'].values))
    logger.info(f'Prediction saved, RMSE: {np.mean(rmse):.4f}±{np.std(rmse):.4f}, '
                    f'RMSE_cliff: {np.mean(rmse_cliff):.4f}±{np.std(rmse_cliff):.4f}') if args.print else None
    logger.handlers.clear()      
    return


def run_baseline_QSAR(args):
    from MoleculeACE_baseline import load_MoleculeACE_model

    args, logger = set_up(args)
    
    # check in the current task is finished previously, if so, skip
    # if os.path.exists(os.path.join(args.save_path, f'{args.baseline_model}_test_pred.csv')):
    #     if args.print:
    #         logger.info(f'current task {args.data_name} for model {args.baseline_model} has been finished, skip...')
    #     return
    logger.info(f'current task: {args.data_name}')

    if args.baseline_model == 'KANO':
        from KANO_model.train_val import train_KANO
        train_KANO(args, logger)
        return
    # Note: as the Data class in Molecule ACE directly extracts split index from the original dataset, 
    # it is highly recommended to run KANO first to keep consistency between the baseline.
    data = Data(args.data_path)

    descriptor, model = load_MoleculeACE_model(args, logger)

    # Data augmentation for Sequence-based models
    if args.baseline_model in ['CNN', 'LSTM', 'Transformer']:
        AUGMENTATION_FACTOR = 10
        data.augment(AUGMENTATION_FACTOR)
        data.shuffle()

    data(descriptor)
    logger.info('training size: {}, test size: {}'.format(len(data.x_train), len(data.x_test))) if args.print else None                                                             
    logger.info(f'training {args.baseline_model}...') if args.print else None

    model.train(data.x_train, data.y_train)

    # save model
    model_save_path = os.path.join(args.save_path, f'{args.baseline_model}_model.pkl')
    model_save_path = model_save_path.replace(
                        '.pkl','.h5') if args.baseline_model == 'LSTM' else model_save_path
    if args.baseline_model == 'LSTM':
        model.model.save(model_save_path)
    else:
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
    df_test.to_csv(args.save_pred_path, index=False)
    logger.info(f'Prediction saved, RMSE: {rmse:.4f}, RMSE_cliff: {rmse_cliff:.4f}') if args.print else None
    logger.handlers.clear()
    return


def run_baseline_CPI(args):

    args, logger = set_up(args)

    df_all, test_idx, train_data, val_data, test_data = process_data_CPI(args, logger)

    if args.baseline_model == 'DeepDTA':
        import DeepPurpose.DTI as models
        from DeepPurpose.utils import generate_config

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
        logger.info(f'load {args.baseline_model} model from DeepPurpose') if args.print else None
        if len(val_data) > 0:
            model.train(train=train_data, val=val_data, test=test_data)
        else:
            model.train(train=train_data, val=None, test=test_data)
        # get predictions
        test_pred = model.predict(test_data)
        model.save_model(args.save_path) 

    elif args.baseline_model == 'GraphDTA':
        from CPI_baseline.GraphDTA import GraphDTA

        model = GraphDTA(args, logger)
        # Note: the hyperparameters are reported as the best performing ones
        # for the KIBA and DAVIS dataset
        logger.info(f'load {args.baseline_model} model') if args.print else None
        logger.info(f'training {args.baseline_model}...') if args.print else None

        model.train(args, logger, train_data, val_data)
        # get predictions
        _, test_pred = model.predict(test_data)

    elif args.baseline_model == 'MolTrans':
        from CPI_baseline.MolTrans import MolTrans
        from CPI_baseline.utils import MolTrans_config_DBPE

        config = MolTrans_config_DBPE()
        model = MolTrans(args, logger, config)
        logger.info(f'load {args.baseline_model} model') if args.print else None
        logger.info(f'training {args.baseline_model}...') if args.print else None
        if len(val_data) > 0:
            model.train(args, logger, train_data, val_loader=val_data)
        else:
            model.train(args, logger, train_data, val_loader=train_data)
        # get predictions
        _, test_pred = model.predict(test_data)

    test_data_all = df_all[df_all['split']=='test']

    if 'Chembl_id' in test_data_all.columns:
        test_data_all['Chembl_id'] = test_data_all['Chembl_id'].values
        task = test_data_all['Chembl_id'].unique()
    else:
        task = test_data_all['Uniprot_id'].unique()

    test_data_all['Prediction'] = test_pred[:len(test_data_all)] # some baselines may have padding, delete the exceeds
    test_data_all = test_data_all.rename(columns={'Label': 'y'})
    test_data_all.to_csv(args.save_pred_path, index=False)
    rmse = calc_rmse(test_data_all['y'].values, test_data_all['Prediction'].values)
    rmse_cliff = calc_cliff_rmse(y_test_pred=test_data_all['Prediction'].values,
                                        y_test=test_data_all['y'].values,
                                    cliff_mols_test=test_data_all['cliff_mol'].values)
    logger.info(f'Prediction saved, RMSE: {rmse:.4f}, '
                    f'RMSE_cliff: {rmse_cliff:.4f}') if args.print else None
    logger.handlers.clear()                     
    return


if __name__ == '__main__':
    args = add_args()

    if args.mode in ['train', 'finetune', 'retrain']:
        if args.train_model == 'KANO_Siams':
            run_QSAR(args)
        elif args.train_model in ['KANO_Prot', 'KANO_Prot_Siams']:
            run_CPI(args)

    elif args.mode == 'inference':
        predict_main(args)
    elif args.mode == 'baseline_QSAR':
        run_baseline_QSAR(args)
    elif args.mode == 'baseline_CPI':
        run_baseline_CPI(args)