import os
import numpy as np
import pandas as pd
from math import sqrt
from scipy import stats
import torch
from torch.utils import data
from torch_geometric.data import InMemoryDataset
from torch_geometric import data as DATA
from subword_nmt.apply_bpe import BPE
import codecs


class TestbedDataset(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis', 
                 xd=None, xt=None, y=None, transform=None,
                 pre_transform=None,smile_graph=None):

        #root is required for save preprocessed data, default is '/tmp'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, xt, y,smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass
        #return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    # Customize the process method to fit the task of drug-target affinity prediction
    # Inputs:
    # XD - list of SMILES, XT: list of encoded target (categorical or one-hot),
    # Y: list of labels (i.e. affinity)
    # Return: PyTorch-Geometric format processed data
    def process(self, xd, xt, y,smile_graph):
        assert (len(xd) == len(xt) and len(xt) == len(y)), "The three lists must be the same length!"
        data_list = []
        data_len = len(xd)
        for i in range(data_len):
            print('Converting SMILES to graph: {}/{}'.format(i+1, data_len))
            smiles = xd[i]
            target = xt[i]
            labels = y[i]
            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index = smile_graph[smiles]
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                                y=torch.FloatTensor([labels]))
            GCNData.target = torch.LongTensor([target])
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            # append graph, label and target sequence to data list
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])

def rmse(y,f):
    rmse = sqrt(((y - f)**2).mean(axis=0))
    return rmse
def mse(y,f):
    mse = ((y - f)**2).mean(axis=0)
    return mse
def pearson(y,f):
    rp = np.corrcoef(y, f)[0,1]
    return rp
def spearman(y,f):
    rs = stats.spearmanr(y, f)[0]
    return rs
def ci(y,f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci


class MolTrans_Data_Encoder(data.Dataset):

    def __init__(self, list_IDs, labels, df_dti):
        'Initialization'
        self.labels = labels
        self.list_IDs = list_IDs
        self.df = df_dti
        vocab_path = 'CPI_baseline/MolTrans_encode/protein_codes_uniprot.txt'
        bpe_codes_protein = codecs.open(vocab_path)
        self.pbpe = BPE(bpe_codes_protein, merges=-1, separator='')
        sub_csv = pd.read_csv('CPI_baseline/MolTrans_encode/subword_units_map_uniprot.csv')

        idx2word_p = sub_csv['index'].values
        self.words2idx_p = dict(zip(idx2word_p, range(0, len(idx2word_p))))

        vocab_path = 'CPI_baseline/MolTrans_encode/drug_codes_chembl.txt'
        bpe_codes_drug = codecs.open(vocab_path)
        self.dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
        sub_csv = pd.read_csv('CPI_baseline/MolTrans_encode/subword_units_map_chembl.csv')

        idx2word_d = sub_csv['index'].values
        self.words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

    def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        # Load data and get label
        index = self.list_IDs[index]
        #d = self.df.iloc[index]['DrugBank ID']
        d = self.df.iloc[index]['smiles']
        p = self.df.iloc[index]['Sequence']
        
        #d_v = drug2single_vector(d)
        d_v, input_mask_d = drug2emb_encoder(d, self.pbpe, self.words2idx_d)
        p_v, input_mask_p = protein2emb_encoder(p, self.dbpe, self.words2idx_p)
        
        #print(d_v.shape)
        #print(input_mask_d.shape)
        #print(p_v.shape)
        #print(input_mask_p.shape)
        y = self.labels[index]
        return d_v, p_v, input_mask_d, input_mask_p, y
    
    
def protein2emb_encoder(x, pbpe, words2idx_p):
    max_p = 545
    t1 = pbpe.process_line(x).split()  # split
    try:
        i1 = np.asarray([words2idx_p[i] for i in t1])  # index
    except:
        i1 = np.array([0])
        #print(x)

    l = len(i1)
   
    if l < max_p:
        i = np.pad(i1, (0, max_p - l), 'constant', constant_values = 0)
        input_mask = ([1] * l) + ([0] * (max_p - l))
    else:
        i = i1[:max_p]
        input_mask = [1] * max_p
        
    return i, np.asarray(input_mask)


def drug2emb_encoder(x, dbpe, words2idx_d):
    max_d = 50
    #max_d = 100
    t1 = dbpe.process_line(x).split()  # split
    try:
        i1 = np.asarray([words2idx_d[i] for i in t1])  # index
    except:
        i1 = np.array([0])
        #print(x)
    
    l = len(i1)

    if l < max_d:
        i = np.pad(i1, (0, max_d - l), 'constant', constant_values = 0)
        input_mask = ([1] * l) + ([0] * (max_d - l))

    else:
        i = i1[:max_d]
        input_mask = [1] * max_d

    return i, np.asarray(input_mask)


def MolTrans_config_DBPE():
    config = {}
    config['batch_size'] = 16
    config['input_dim_drug'] = 23532
    config['input_dim_target'] = 16693
    config['train_epoch'] = 13
    config['max_drug_seq'] = 50
    config['max_protein_seq'] = 545
    config['emb_size'] = 384
    config['dropout_rate'] = 0.1
    
    #DenseNet
    config['scale_down_ratio'] = 0.25
    config['growth_rate'] = 20
    config['transition_rate'] = 0.5
    config['num_dense_blocks'] = 4
    config['kernal_dense_size'] = 3
    
    # Encoder
    config['intermediate_size'] = 1536
    config['num_attention_heads'] = 12
    config['attention_probs_dropout_prob'] = 0.1
    config['hidden_dropout_prob'] = 0.1
    config['flat_dim'] = 78192
    return config