import argparse
import torch
from chemprop.features import get_available_features_generators

def add_args():
    """
    Adds predict arguments to an ArgumentParser.

    :param parser: An ArgumentParser.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # general arguments
    parser.add_argument('--gpu', type=int,
                        # choices=list(range(torch.cuda.device_count())),
                        help='Which GPU to use')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='Turn off cuda')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'inference', 'retrain', 'finetune', 'baseline_QSAR', 'baseline_CPI'],
                        help='Mode to run script in')
    parser.add_argument('--print', action='store_true', default=False,
                        help='Print log')
    parser.add_argument('--data_path', type=str,
                        help='Path to CSV file containing training data',
                        default=None)
    parser.add_argument('--test_path', type=str,
                        help='Path to CSV file containing testing data for which predictions will be made',
                        default=None)
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to model checkpoint (.pt file) for inference, retrain, or finetune')
    parser.add_argument('--dataset_type', type=str, choices=['classification', 'regression', 'multiclass'],
                        help='Type of dataset')
    # parser.add_argument('--save_dir', type=str, default=None,
    #                     help='dir name in exp_results folder where predictions will be saved',
    #                     default='test')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--split_type', type=str, default='moleculeACE',
                        choices=['random', 'scaffold_balanced', 'moleculeACE', 'predetermined'],
                        help='Method of splitting the data into training, validation, and test')
    parser.add_argument('--split_sizes', type=float, nargs='+',
                        default=[0.8, 0.0, 0.2],
                        help='Proportions of data to use for training, validation, and test')
    parser.add_argument('--features_scaling', action='store_true', default=False,
                        help='Turn on scaling of features')                  
    parser.add_argument('--features_generator', type=str, nargs='*',
                        choices=get_available_features_generators(),
                        help='Method of generating additional features')
    parser.add_argument('--features_path', type=str, nargs='*', default=None,
                        help='Path to features to use in FNN (instead of features_generator)')
    parser.add_argument('--no_features_scaling', action='store_true', default=False,
                        help='Turn off scaling of features')
    parser.add_argument('--max_data_size', type=int, default=None,
                        help='Maximum number of data points to load')
    
    # training arguments
    parser.add_argument('--checkpoint_path', type=str,
                        default='KANO_model/dumped/pretrained_graph_encoder/original_CMPN_0623_1350_14000th_epoch.pkl',
                        help='Path to model checkpoint (.pt file)')
    # parser.add_argument('--loss', type=str, default='MSE CLS CL',
    #                     help='Loss function seperated with space. MSE: mean squared error, CLS: cross entropy loss, CL: contrastive loss')
    parser.add_argument('--loss_weights', type=str, default='1 1 1',
                        help='Weights for MSE, CLS, and CL loss functions seperated with space'
                        'Note: MSE: mean squared error, CLS: cross entropy loss, CL: contrastive loss'
                        'Set 0 to ignore the specific loss function')
    parser.add_argument('--siams_num', type=int, default=1, 
                        help='Number of siamese pairs')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--temperature', type=float, default=0.1)
    parser.add_argument('--warmup_epochs', type=int, default=2)
    parser.add_argument('--encoder_name', type=str, default='CMPNN',
                        help='selected molecule encoder')
    parser.add_argument('--metric', type=str, default='rmse',
                        choices=['auc', 'prc-auc', 'rmse', 'mae', 'r2', 'accuracy', 'cross_entropy'],
                        help='Metric to optimize during training')
    
    # model arguments
    # you may not able to change most of these arguments if you use a pretrained model
    parser.add_argument('--train_model', type=str, default='KANO_Prot_Siams', 
                        choices=['KANO_Prot_Siams', 'KANO_Prot', 'KANO_Siams'], 
                        help='KANO_Prot_Siams and KANO_Prot for CPI-type model, KANO_Siams for QSAR-type model')
    parser.add_argument('--baseline_model', type=str, default=None,
                        choices=['MLP', 'SVM', 'RF', 'GBM', 'KNN',
                                 'GAT', 'GCN', 'AFP', 'MPNN', 'CNN',
                                 'Transformer','LSTM', 'KANO',
                                 'DeepDTA', 'GraphDTA', 'MolTrans'],
                        help='Type of baseline model to train if select mode as baseline_QSAR or baseline_CPI')
    parser.add_argument('--hidden_size', type=int, default=300)
    parser.add_argument('--ffn_hidden_size', type=int, default=300)
    parser.add_argument('--ffn_num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--activation', type=str, default='ReLU')
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--step', type=str, default='functional_prompt')
    parser.add_argument('--num_heads', type=int, default=5)
    parser.add_argument('--pooling', type=str, default='cross_attn', choices=['cross_attn', 'mean'])

    args = parser.parse_args()
    # add and modify some args
    if '.csv' not in args.data_path:
        args.data_path += '.csv'
    args.data_name = args.data_path.split('/')[-1].split('.')[0]
    if not args.no_cuda and torch.cuda.is_available():
        args.cuda = True
    else:
        args.cuda = False
    args.atom_messages = False
    args.use_input_features = None
    args.bias = False
    args.undirected = False
    args.features_only = False
    args.max_lr = args.lr * 10
    args.init_lr = args.lr
    args.final_lr = args.lr
    args.num_lrs = 1
    args.num_runs = 1
    args.smiles_columns = 'smiles'
    args.target_columns = 'y'
    args.output_size = args.num_tasks = 1
    
    loss_func = ['MSE', 'CLS', 'CL']
    loss_wt = args.loss_weights.split(' ')
    args.loss_func_wt = dict(zip(loss_func, loss_wt))

    if args.baseline_model in ['DeepDTA', 'GraphDTA', 'MolTrans']:
        args.mode == 'baseline_CPI'
    if args.metric in ['auc', 'prc-auc', 'accuracy', 'r2']:
        args.minimize_score = False
    elif args.metric in ['rmse', 'mae', 'cross_entropy']:
        args.minimize_score = True

    return args