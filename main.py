import os
import torch
import numpy as np
import pandas as pd
from chemprop.data import StandardScaler
from KANO_model.model import build_model, add_functional_prompt
from KANO_model.utils import build_optimizer, build_lr_scheduler, build_loss_func
from data_prep import process_data_QSAR, process_data_CPI
from warnings import simplefilter
from chemprop.train.evaluate import evaluate_predictions
from chemprop.train.evaluate import evaluate_predictions
from torch.optim.lr_scheduler import ExponentialLR
from args import add_args
from utils import set_save_path, set_seed, set_collect_metric, \
                  collect_metric_epoch, get_metric_func, save_checkpoint \
                  define_logging, set_up
from MoleculeACE.benchmark.utils import Data, calc_rmse, calc_cliff_rmse
import pickle


def run_QSAR(args):
    from chemprop.data import MoleculeDataset
    from model.models import KANO_Siams

    args, logger = set_up(args)
    
    # check in the current task is finished previously, if so, skip
    if os.path.exists(os.path.join(args.save_path, f'{args.baseline_model}_test_pred.csv')):
        logger.info(f'current task {args.data_name} for model {args.baseline_model} has been finished, skip...')
        return
    
    logger.info(f'current task: {args.data_name}')    

    data = get_data(path=args.data_path, 
                smiles_columns=args.smiles_columns,
                target_columns=args.target_columns,
                ignore_columns=args.ignore_columns)

    df = pd.read_csv(args.data_path)
    if args.split_sizes:
        _, valid_ratio, test_ratio = args.split_sizes
        train_idx, test_idx = df[df['split']=='train'].index, df[df['split']=='test'].index
        val_idx = random.sample(list(train_idx), int(len(df) * valid_ratio))
        train_idx = list(set(train_idx) - set(val_idx))

    train_prot, val_prot, test_prot = df.loc[train_idx, 'UniProt_id'].values, \
                                  df.loc[val_idx, 'UniProt_id'].values, \
                                  df.loc[test_idx, 'UniProt_id'].values
    train_data, val_data, test_data = tuple([[data[i] for i in train_idx],
                                            [data[i] for i in val_idx],
                                            [data[i] for i in test_idx]])
    train_data, val_data, test_data = MoleculeDataset(train_data), \
                                        MoleculeDataset(val_data), \
                                        MoleculeDataset(test_data)

    model = KANO_Siams(args, 
                       classification=True, multiclass=False,
                       multitask=False, prompt=True).to(args.device)

    data, prot, support_data, support_prot = train_data, train_prot, test_data, test_prot
    data_idx = list(range(len(data)))
    random.shuffle(data_idx)
    data = [data[i] for i in data_idx]
    iter_size = args.batch_size

    for i in range(0, len(data), iter_size):
        if i + iter_size > len(data):
            break
        batch_data = MoleculeDataset(data[i:i + iter_size])
        current_idx = data_idx[i:i + iter_size]
        siams_smiles, siams_labels = generate_siamse_smi(batch_data.smiles(), args, prot_ids,
                                                        support_prot, support_data, strategy='random')

        reg_label = torch.tensor(label).to(args.device)
        cls_label = reg_label - torch.tensor(siams_labels).to(args.device)
        cls_label = torch.tensor([1 if label_dis >= 2 else 0 
                                    for label_dis in cls_label], dtype=torch.float32).to(args.device)
        smiles, feat, label = batch_data.smiles(), batch_data.features(), batch_data.targets()

        pred, [mol1, mol2] = model('finetune', smiles, siams_smiles)

def run_baseline_QSAR(args):
    from MoleculeACE_baseline import load_MoleculeACE_model

    args, logger = set_up(args)
    
    # check in the current task is finished previously, if so, skip
    if os.path.exists(os.path.join(args.save_path, f'{args.baseline_model}_test_pred.csv')):
        logger.info(f'current task {args.data_name} for model {args.baseline_model} has been finished, skip...')
        return
    
    logger.info(f'current task: {args.data_name}')

    if args.baseline_model == 'KANO':
        from KANO_model.train_val import train_KANO
        train_KANO(args)
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
    logger.info('training size: {}, test size: {}'.format(len(data.x_train), len(data.x_test)))                                                                     
    logger.info(f'training {args.baseline_model}...')

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
    df_test.to_csv(os.path.join(args.save_path, f'{args.baseline_model}_test_pred.csv'), index=False)
    logger.info(f'Prediction saved, RMSE: {rmse:.4f}, RMSE_cliff: {rmse_cliff:.4f}')

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
        logger.info(f'load {args.baseline_model} model from DeepPurpose')
        model = models.model_initialize(**config)

        logger.info(f'training {args.baseline_model}...')
        if len(val_data) > 0:
            model.train(train=train_data, val=val_data, test=test_data)
        else:
            model.train(train=train_data, val=None, test=test_data)
        # get predictions
        test_pred = model.predict(test_data)
        model.save_model(os.path.join(args.save_path,f'{args.baseline_model}')) 

    elif args.baseline_model == 'GraphDTA':
        from CPI_baseline.GraphDTA import GraphDTA

        model = GraphDTA(args, logger)
        # Note: the hyperparameters are reported as the best performing ones
        # for the KIBA and DAVIS dataset
        logger.info(f'load {args.baseline_model} model')
        logger.info(f'training {args.baseline_model}...')
        model.train(args, logger, train_data, val_data)
        # get predictions
        _, test_pred = model.predict(test_data)

    elif args.baseline_model == 'MolTrans':
        from CPI_baseline.MolTrans import MolTrans
        from CPI_baseline.utils import MolTrans_config_DBPE
        
        config = MolTrans_config_DBPE()
        model = MolTrans(args, logger, config)
        logger.info(f'load {args.baseline_model} model')
        logger.info(f'training {args.baseline_model}...')  
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
        task = test_data_all['UniProt_id'].unique()

    test_data_all['Prediction'] = test_pred[:len(test_data_all)] # some baseline may have padding, delete the exceeds
    test_data_all = test_data_all.rename(columns={'Label': 'y'})
    test_data_all.to_csv(os.path.join(args.save_path, f'{args.data_name}_test_pred.csv'), index=False)
    rmse, rmse_cliff = [], []

    for target in task:
        if 'Chembl_id' in test_data_all.columns:
            test_data_target = test_data_all[test_data_all['Chembl_id']==target]
        else:
            test_data_target = test_data_all[test_data_all['UniProt_id']==target]
        rmse.append(calc_rmse(test_data_target['y'].values, test_data_target['Prediction'].values))
        rmse_cliff.append(calc_cliff_rmse(y_test_pred=test_data_target['Prediction'].values,
                                          y_test=test_data_target['y'].values,
                                        cliff_mols_test=test_data_target['cliff_mol'].values))
                                        
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