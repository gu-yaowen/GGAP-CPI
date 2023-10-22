import os
from MoleculeACE.benchmark.const import Descriptors
from MoleculeACE.models import RF, SVM, GBM, KNN, MLP, GCN, MPNN, AFP, GAT, CNN, LSTM, Transformer
from utils import get_config

MOLECULEACE_DATALIST = ['CHEMBL1862_Ki', 'CHEMBL1871_Ki', 'CHEMBL2034_Ki', 'CHEMBL2047_EC50',
                        'CHEMBL204_Ki', 'CHEMBL2147_Ki', 'CHEMBL214_Ki', 'CHEMBL218_EC50',
                        'CHEMBL219_Ki', 'CHEMBL228_Ki', 'CHEMBL231_Ki', 'CHEMBL233_Ki',
                        'CHEMBL234_Ki', 'CHEMBL235_EC50', 'CHEMBL236_Ki', 'CHEMBL237_EC50',
                        'CHEMBL237_Ki', 'CHEMBL238_Ki', 'CHEMBL239_EC50', 'CHEMBL244_Ki',
                        'CHEMBL262_Ki', 'CHEMBL264_Ki', 'CHEMBL2835_Ki', 'CHEMBL287_Ki',
                        'CHEMBL2971_Ki', 'CHEMBL3979_EC50', 'CHEMBL4005_Ki', 'CHEMBL4203_Ki',
                        'CHEMBL4616_EC50', 'CHEMBL4792_Ki']


def load_MoleculeACE_model(args, logger):
    if args.baseline_model in ['MLP', 'SVM', 'RF', 'GBM', 'KNN']:
        descriptor = Descriptors.ECFP
        des = 'ECFP'
    elif args.baseline_model in ['GAT', 'GCN', 'AFP', 'MPNN']:
        descriptor = Descriptors.GRAPH
        des = 'Graph'
    elif args.baseline_model == 'Transformer':
        descriptor = Descriptors.TOKENS
        des = 'TOKENS'
    elif args.baseline_model in ['CNN', 'LSTM']:
        descriptor = Descriptors.SMILES
        des = 'SMILES'
    logger.info(f'using {des} as input')

    model_zoo = {'MLP': MLP, 'SVM': SVM, 'RF': RF, 'GBM': GBM, 'KNN': KNN, # Fingerprint-based models
                 'GAT': GAT, 'GCN': GCN, 'AFP': AFP, 'MPNN': MPNN, # Graph-based models
                 'Transformer': Transformer, 'CNN': CNN, 'LSTM': LSTM # Sequence-based models
                 }
    logger.info(f'load {args.baseline_model} model')

    # load config if existed
    if args.data_name in MOLECULEACE_DATALIST:
        config_path = os.path.join('MoleculeACE_configures', 'benchmark',
                                   args.data_name, f'{args.baseline_model}_{des}.yml')
        hyperparameters = get_config(config_path)
        if args.baseline_model == 'LSTM':
            hyperparameters['pretrained_model'] = None
        model = model_zoo[args.baseline_model](**hyperparameters)
    else:
        try:
            config_path = os.path.join('MoleculeACE_configures', 'default',
                                    f'{args.baseline_model}.yml')        
            hyperparameters = get_config(config_path)
            if args.baseline_model == 'LSTM':
                hyperparameters['pretrained_model'] = None
            model = model_zoo[args.baseline_model](**hyperparameters)
            logger.info(f'using default config file {config_path} to load model')
        except:
            config_path = None
            if args.baseline_model == 'LSTM':
                model = model_zoo[args.baseline_model](pretrained_model=None)
            model = model_zoo[args.baseline_model]()
            logger.info(f'no config file found, using default parameters!!')
            pass
    logger.info(f'{model}')
    return descriptor, model