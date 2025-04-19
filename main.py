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
                  define_logging, set_up, get_protein_feature, get_ligand_feature
from model.train_val import retrain_scheduler, train_epoch, evaluate_epoch, predict_epoch
                #   train_epoch_chunkwise, predict_epoch_chunkwise
from model.utils import generate_siamse_smi, set_up_model, generate_label


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
        if 'valid' in df_all['split'].unique():
            val_idx = df_all[df_all['split']=='valid'].index
        else:
            val_idx = random.sample(list(train_idx), int(len(train_idx) * valid_ratio))
            # train_idx = list(set(train_idx) - set(val_idx))

    type_id = df_all['type_id'].values
    train_id, val_id, test_id = type_id[train_idx], type_id[val_idx], type_id[test_idx]

    train_prot, val_prot, test_prot = df_all.loc[train_idx, 'Uniprot_id'].values, \
                                      df_all.loc[val_idx, 'Uniprot_id'].values, \
                                      df_all.loc[test_idx, 'Uniprot_id'].values

    train_data, val_data, test_data = tuple([[data[i] for i in train_idx],
                                            [data[i] for i in val_idx],
                                            [data[i] for i in test_idx]])
    train_data, val_data, test_data = MoleculeDataset(train_data), \
                                      MoleculeDataset(val_data), \
                                      MoleculeDataset(test_data)

    if len(train_data) <= args.batch_size:
        # args.batch_size = 64
        logger.info(f'batch size is too large, reset to {args.batch_size}') if args.print else None

    if args.features_scaling:
        features_scaler = train_data.normalize_features(replace_nan_token=0)
        val_data.normalize_features(features_scaler)
        test_data.normalize_features(features_scaler)
    else:
        features_scaler = None

    if args.dataset_type == 'classification':
        pos_weight = len(df_all[df_all['split']=='train']['y']) / \
                     df_all[df_all['split']=='train']['y'].sum()
        args.pos_weight = pos_weight
        logger.info(f'positive weight: {pos_weight}') if args.print else None     
        args.scaler = None

    args.train_data_size = len(train_data)
    n_iter = 0
    best_score = float('inf') if args.minimize_score else -float('inf')
    query_train = [np.array(train_data.smiles()).flatten(), 
                   np.array(train_data.targets()), 
                   train_id]
    query_val = [np.array(val_data.smiles()).flatten(), 
                np.array(val_data.targets()),
                val_id]
    query_test = [np.array(test_data.smiles()).flatten(),
                  np.array(test_data.targets()),
                  test_id]    
    # load protein features
    prot_graph_dict = get_protein_feature(args, logger, df_all)

    # load ligand features
    lig_graph_dict = None
    args.chunk_files = False
    args.graph_input = True
    # load model, optimizer, scheduler, loss function
    args, model, optimizer, scheduler, loss_func = set_up_model(args, logger)
    metric_dict = set_collect_metric(args)
    
    # training
    logger.info(f'training...') if args.print else None
    best_loss = 999 if args.dataset_type in ['regression', 'joint'] else -999
    if args.mode == 'retrain':
        logger.info(f'retraining...') if args.print else None

    for epoch in range(args.previous_epoch+1, args.epochs):
        args.epoch = epoch
        n_iter, loss_collect = train_epoch(args, logger,
                                           model, 
                                           prot_graph_dict, lig_graph_dict, 
                                           query_train, train_prot, 
                                           loss_func, optimizer, scheduler, n_iter)
        if isinstance(scheduler, ExponentialLR):
            scheduler.step()
        if len(query_val[1]) > 0:
            val_scores, _ = evaluate_epoch(args, 
                                           model, 
                                           prot_graph_dict, lig_graph_dict, 
                                           query_val, val_prot, 
                                           args.scaler)
        else:
            val_scores = metric_dict
        if len(query_test[1]) > 0:
            test_scores, _ = evaluate_epoch(args, 
                                            model, 
                                            prot_graph_dict, lig_graph_dict, 
                                            query_test, test_prot, 
                                            args.scaler)
        else:
            test_scores = val_scores
        if args.dataset_type == 'regression':
            logger.info('Epoch : {:02d}, Loss_Total: {:.3f}, Loss_MSE: {:.3f}, ' \
                        'Validation score : {:.3f}, Test score : {:.3f}'.format(epoch, 
                        loss_collect['Total'], loss_collect['MSE'],
                        list(val_scores.values())[0][0], list(test_scores.values())[0][0])) if args.print else None
        elif args.dataset_type == 'classification':
            logger.info('Epoch : {:02d}, Loss_CLS: {:.3f}, Train ACC: {:.3f}, Train AUC: {:.3f}, Train AUPR: {:.3f}, ' \
                        'Validation ACC: {:.3f}, Validation AUC : {:.3f}, Validation AUPR: {:.3f}, ' \
                        'Test ACC: {:.3f}, Test AUC : {:.3f}, Test AUPR: {:.3f}'.format(epoch, 
                        loss_collect['CrossEntropy'], loss_collect['ACC'], loss_collect['AUC'], loss_collect['AUPR'],
                        list(val_scores.values())[0][0], list(val_scores.values())[1][0], list(val_scores.values())[2][0],
                        list(test_scores.values())[0][0], list(test_scores.values())[1][0], list(test_scores.values())[2][0])) if args.print else None
        elif args.dataset_type == 'joint':
            logger.info('Epoch : {:02d}, Loss_Total: {:.3f}, Loss_MSE: {:.3f}, Loss_CLS: {:.3f}, ' \
                        'Validation RMSE : {:.3f}, Validation Acc: {:.3f}, ' \
                        'Test RMSE : {:.3f}, Test Acc: {:.3f}'.format(epoch, 
                        loss_collect['Total'], loss_collect['MSE'], loss_collect['CrossEntropy'],
                        list(val_scores.values())[0][0], list(val_scores.values())[-1][0],
                        list(test_scores.values())[0][0], list(test_scores.values())[-1][0])) if args.print else None

        if epoch < args.epochs - 1:
            save_checkpoint(args.save_model_path, 
                            model, args.scaler, features_scaler, epoch, optimizer, scheduler, args)
        if args.dataset_type == 'regression':
            if loss_collect['MSE'] < best_loss or epoch == 0:
                best_loss = loss_collect['MSE']
                best_score, best_epoch = list(val_scores.values())[0][-1], epoch
                best_test_score = list(test_scores.values())[0][-1]
                save_checkpoint(args.save_best_model_path, 
                                model, args.scaler, features_scaler, epoch, optimizer, scheduler, args) 
        elif args.dataset_type == 'classification':
            if loss_collect['AUC'] > best_loss or epoch == 0:
                best_loss = loss_collect['AUC']
                best_score, best_epoch = list(val_scores.values())[0][-1], epoch
                best_test_score = list(test_scores.values())[0][-1]
                save_checkpoint(args.save_best_model_path, 
                                model, args.scaler, features_scaler, epoch, optimizer, scheduler, args)
        elif args.dataset_type == 'joint':
            if loss_collect['Total'] < best_loss or epoch == 0:
                best_loss = loss_collect['Total']
                best_score, best_epoch = list(val_scores.values())[3][-1], epoch
                best_test_score = list(test_scores.values())[3][-1]
                save_checkpoint(args.save_best_model_path, 
                                model, args.scaler, features_scaler, epoch, optimizer, scheduler, args)
        save_checkpoint(args.save_model_path.split('.')[0]+'_'+str(epoch)+'.pt', 
                        model, args.scaler, features_scaler, epoch, optimizer, scheduler, args)
    logger.info('Final best performed model in {} epoch, val score: {:.4f}, '
                'test score: {:.4f}'.format(best_epoch, best_score, best_test_score)) if args.print else None

    # test the best model
    model.load_state_dict(torch.load(args.save_best_model_path)['state_dict'])
    args.chunk_files = False
    test_pred, _ = predict_epoch(args, 
                                 model, 
                                 prot_graph_dict, lig_graph_dict, 
                                 query_test, test_prot, 
                                 args.scaler)

    test_data_all = df_all[df_all['split']=='test']

    if 'Chembl_id' in test_data_all.columns:
        test_data_all['Chembl_id'] = test_data_all['Chembl_id'].values
        task = test_data_all['Chembl_id'].unique()
    else:
        task = test_data_all['Uniprot_id'].unique()
    if args.dataset_type in ['classification', 'regression']:
        test_data_all['Prediction'] = np.array(test_pred).flatten()[:len(test_data_all)] # some baseline may have padding, delete the exceeds
    elif args.dataset_type == 'joint':
        test_data_all['Prediction'] = np.array(test_pred[0]).flatten()[:len(test_data_all)] # some baseline may have padding, delete the exceeds
        cls_pred = np.array(test_pred[1])[:len(test_data_all)] # n * 4
        test_data_all['Prediction_StrongBinder'] = cls_pred[:, 0]
        test_data_all['Prediction_Binder'] = cls_pred[:, 1]
        test_data_all['Prediction_WeakBinder'] = cls_pred[:, 2]
        test_data_all['Prediction_NonBinder'] = cls_pred[:, 3]
    test_data_all = test_data_all.rename(columns={'Label': 'y'})
    test_data_all.to_csv(args.save_pred_path, index=False)
    logger.info(f'Prediction saved in {args.save_pred_path}') if args.print else None
    if args.dataset_type == 'regression':
        rmse = calc_rmse(test_data_all['y'].values, test_data_all['Prediction'].values)
        rmse_cliff = calc_cliff_rmse(y_test_pred=test_data_all['Prediction'].values,
                                    y_test=test_data_all['y'].values,
                                    cliff_mols_test=test_data_all['cliff_mol'].values)
        logger.info(f'Prediction saved, RMSE: {np.mean(rmse):.4f}, '
                        f'RMSE_cliff: {np.mean(rmse_cliff):.4f}') if args.print else None
    else:
        logger.info('Prediction saved') if args.print else None
    logger.handlers.clear()      
    return


def run_baseline_QSAR(args):
    from MoleculeACE_baseline import load_MoleculeACE_model

    args, logger = set_up(args)
    logger.info(f'current task: {args.data_name}')

    if args.baseline_model == 'KANO':
        args.graph_input = False
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
                            train_epoch = 80, # original 100
                            LR = 0.001,
                            batch_size = 512, # original 256
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

    elif args.baseline_model == 'HyperAttentionDTI':
        from CPI_baseline.HyperAttentionDTI import HyperAttentionDTI

        model = HyperAttentionDTI(args, logger)
        # Note: the hyperparameters are reported as the best performing ones
        # for the KIBA, DAVIS, and BindingDB dataset
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

    elif args.baseline_model in ['ECFP_ESM_GBM', 'ECFP_ESM_RF', 'KANO_ESM_GBM', 'KANO_ESM_RF']:
        args.graph_input = False
        from MoleculeACE_baseline import load_MoleculeACE_model
        mol_feat, prot_feat = args.baseline_model.split('_')[0], args.baseline_model.split('_')[1]
        args.baseline_model = args.baseline_model.split('_')[-1]
        descriptor, model = load_MoleculeACE_model(args, logger)
        args.baseline_model = f'{mol_feat}_{prot_feat}_{args.baseline_model}'
        logger.info('training size: {}, test size: {}'.format(len(train_data[0]), len(test_data[0]))) if args.print else None                                                             
        logger.info(f'training {args.baseline_model}...') if args.print else None

        model.train(train_data[1], train_data[0])

        # save model
        model_save_path = os.path.join(args.save_path, f'{args.baseline_model}_model.pkl')
        with open(model_save_path, 'wb') as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

        test_pred = model.predict(test_data[1])

    test_data_all = df_all[df_all['split']=='test']

    if 'Chembl_id' in test_data_all.columns:
        test_data_all['Chembl_id'] = test_data_all['Chembl_id'].values
        task = test_data_all['Chembl_id'].unique()
    else:
        task = test_data_all['Uniprot_id'].unique()

    test_data_all['Prediction'] = test_pred[:len(test_data_all)] # some baselines may have padding, delete the exceeds
    test_data_all = test_data_all.rename(columns={'Label': 'y'})
    test_data_all.to_csv(args.save_pred_path, index=False)
    logger.info(f'Prediction saved in {args.save_pred_path}') if args.print else None
    rmse = calc_rmse(test_data_all['y'].values, test_data_all['Prediction'].values)
    rmse_cliff = calc_cliff_rmse(y_test_pred=test_data_all['Prediction'].values,
                                        y_test=test_data_all['y'].values,
                                    cliff_mols_test=test_data_all['cliff_mol'].values)
    logger.info(f'Prediction saved, RMSE: {rmse:.4f}, '
                    f'RMSE_cliff: {rmse_cliff:.4f}') if args.print else None
    logger.handlers.clear()                     
    return


def predict_main(args):
    args, logger = set_up(args)
    if args.mode == 'inference':
        df_all, test_idx, _, _, test_data = process_data_CPI(args, logger)
        data = get_data(path=args.data_path, 
                        smiles_columns=args.smiles_columns,
                        target_columns=args.target_columns,
                        ignore_columns=args.ignore_columns)
        test_idx = df_all[df_all['split']=='test'].index
        test_prot = df_all.loc[test_idx, 'Uniprot_id'].values
        test_data = [data[i] for i in test_idx]
        test_data = MoleculeDataset(test_data)
        args.batch_size = 256
        if args.dataset_type == 'regression':
            ref_df = pd.read_csv(args.ref_path)
            ref_y = ref_df['y'].values
            args.scaler = StandardScaler().fit(ref_y)
        else:
            args.pos_weight = 0
            args.scaler = None
        args.train_data_size = len(test_data)
        
        args.graph_input = True
        lig_graph_dict = None
        args.chunk_files = False
        args, model, optimizer, scheduler, loss_func = set_up_model(args, logger)
    
        if args.dataset_type == 'joint':
            test_id = df_all['type_id'].values[test_idx]
        query_test = [np.array(test_data.smiles()).flatten(),
                np.array(test_data.targets()),
                test_id if args.dataset_type == 'joint' else []]
        prot_graph_dict = get_protein_feature(args, logger, df_all)
        # lig_graph_dict = get_ligand_feature(args, logger, df_all)

        args.chunk_files = False
        test_pred, _ = predict_epoch(args, model, prot_graph_dict, lig_graph_dict,
                                    query_test, test_prot, args.scaler)
        
    elif args.mode == 'baseline_inference':
        if args.baseline_model == 'DeepDTA':
            import DeepPurpose.DTI as models
            from DeepPurpose.utils import generate_config
            df_all, test_idx, _, _, test_data = process_data_CPI(args, logger)
            drug_encoding = 'CNN'
            target_encoding = 'CNN'
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
            model.load_pretrained(args.save_best_model_path)
            test_pred = model.predict(test_data)

        elif args.baseline_model == 'GraphDTA':
            from CPI_baseline.GraphDTA import GraphDTA
            df_all, test_idx, _, _, test_data = process_data_CPI(args, logger)
            model = GraphDTA(args, logger)
            logger.info(f'predicting...') if args.print else None
            _, test_pred = model.predict(test_data)
            
        elif args.baseline_model == 'HyperAttentionDTI':
            from CPI_baseline.HyperAttentionDTI import HyperAttentionDTI
            df_all, test_idx, _, _, test_data = process_data_CPI(args, logger)
            model = HyperAttentionDTI(args, logger)
            logger.info(f'predicting...') if args.print else None
            _, test_pred = model.predict(test_data)
        elif args.baseline_model in ['ECFP_ESM_GBM', 'ECFP_ESM_RF', 'KANO_ESM_GBM', 'KANO_ESM_RF']:
            args.graph_input = False
            df_all, test_idx, _, _, test_data = process_data_CPI(args, logger)
            from MoleculeACE_baseline import load_MoleculeACE_model
            mol_feat, prot_feat = args.baseline_model.split('_')[0], args.baseline_model.split('_')[1]
            args.baseline_model = args.baseline_model.split('_')[-1]
            descriptor, model = load_MoleculeACE_model(args, logger)
            args.baseline_model = f'{mol_feat}_{prot_feat}_{args.baseline_model}'
            # load ML model
            model = pickle.load(open(args.save_best_model_path, 'rb'))
            logger.info(f'predicting...') if args.print else None
            test_pred = model.predict(test_data[1])
        elif args.baseline_model == 'KANO':
            df_all, test_idx, _, _, test_data = process_data_QSAR(args, logger)
            args.graph_input = False
            from KANO_model.train_val import predict_KANO
            test_pred = predict_KANO(args, logger, test_data)

    test_data_all = df_all[df_all['split']=='test']
    
    if 'Chembl_id' in test_data_all.columns:
        test_data_all['Chembl_id'] = test_data_all['Chembl_id'].values
        task = test_data_all['Chembl_id'].unique()
    else:
        task = test_data_all['Uniprot_id'].unique()
    if args.dataset_type in ['classification', 'regression']:
        test_data_all['Prediction'] = np.array(test_pred).flatten()[:len(test_data_all)] # some baseline may have padding, delete the exceeds
    elif args.dataset_type == 'joint': # currently only for GGAP-CPI
        test_data_all['Prediction'] = np.array(test_pred[0]).flatten()[:len(test_data_all)] # some baseline may have padding, delete the exceeds
        cls_pred = np.array(test_pred[1])[:len(test_data_all)] # n * 4
        test_data_all['Prediction_StrongBinder'] = cls_pred[:, 0]
        test_data_all['Prediction_Binder'] = cls_pred[:, 1]
        test_data_all['Prediction_WeakBinder'] = cls_pred[:, 2]
        test_data_all['Prediction_NonBinder'] = cls_pred[:, 3]
    test_data_all = test_data_all.rename(columns={'Label': 'y'})
    test_data_all.to_csv(args.save_pred_path, index=False)
    logger.info(f'Prediction saved in {args.save_pred_path}') if args.print else None
    logger.handlers.clear()   
    return


if __name__ == '__main__':
    args = add_args()

    if args.mode in ['train', 'finetune', 'retrain']:
        if args.train_model in ['GGAP_CPI', 'KANO_ESM']:
            run_CPI(args)

    elif args.mode in ['inference', 'baseline_inference']:
        predict_main(args)
    elif args.mode == 'baseline_QSAR':
        run_baseline_QSAR(args)
    elif args.mode == 'baseline_CPI':
        run_baseline_CPI(args)