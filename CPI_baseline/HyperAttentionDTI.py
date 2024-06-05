# -*- coding: utf-8 -*-
"""
@Time:Created on 2020/8/23 10:10
@author: Qichang Zhao
@Filename: model.py
@Software: PyCharm
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datetime import datetime
from CPI_baseline.utils import EarlyStopping
from MoleculeACE import calc_rmse


class HyperAttentionDTI():
    def __init__(self, args, logger):
        self.args = args
        self.hp = hyperparameter()
        self.model = AttentionDTI(self.hp)
        if args.mode == 'baseline_inference':
            logger.info('load pretrained HyperAttentionDTI model') if args.print else None
            self.load_model(args.save_best_model_path)
        if args.gpu is not None:
            self.device = torch.device("cuda:%d" % args.gpu)
            logger.info("Using GPU: %d" % args.gpu)
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU")

        self.model = self.model.to(self.device)
        weight_p, bias_p = [], []
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for name, p in self.model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        self.LOG_INTERVAL = 100
        self.loss = nn.MSELoss()
        self.optimizer = optim.AdamW([{'params': weight_p, 'weight_decay': self.hp.weight_decay},
                         {'params': bias_p, 'weight_decay': 0}], lr=self.hp.Learning_rate)
        self.scheduler = optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=self.hp.Learning_rate,
                                                max_lr=self.hp.Learning_rate*10, cycle_momentum=False)
        self.early_stopping = EarlyStopping(savepath = self.args.save_path,
                                            patience=self.hp.Patience, verbose=True, delta=0)
        self.best_rmse = 1000
        self.best_model = None

    def train(self, agrs, logger, train_loader, val_loader=None):
        for epoch in range(1, self.hp.Epoch + 1):
            self.model.train()
            for batch_idx, data in enumerate(train_loader):
                self.optimizer.zero_grad()
                compounds, proteins, labels = data
                compounds = compounds.to(self.device)
                proteins = proteins.to(self.device)
                labels = labels.to(self.device)

                output = self.model(compounds, proteins)

                loss = self.loss(output, labels.view(-1, 1).float())
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                if batch_idx % self.LOG_INTERVAL == 0:
                    logger.info('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                        batch_idx, len(train_loader),
                                        100. * batch_idx / len(train_loader), loss.item()))
            if len(val_loader) > 0:
                label, pred = self.predict(val_loader)
            else:
                label, pred = self.predict(train_loader)
            rmse = calc_rmse(label, pred)
            if rmse < self.best_rmse:
                self.best_rmse = rmse
                self.save_model(self.args.save_path)
                self.best_model = self.model
            logger.info('Epoch: {}, RMSE: {:.4f}, Best RMSE: {:.4f}'.format(epoch, rmse, self.best_rmse))
            self.early_stopping(rmse, self.model, epoch)
        
    def predict(self, loader):
        self.model.eval()
        total_preds = torch.Tensor()
        total_labels = torch.Tensor()
        with torch.no_grad():
            for data in loader:
                compounds, proteins, labels = data
                compounds = compounds.to(self.device)
                proteins = proteins.to(self.device)
                labels = labels.to(self.device)
                output = self.model(compounds, proteins)
                total_preds = torch.cat((total_preds, output.cpu()), 0)
                total_labels = torch.cat((total_labels, labels.view(-1, 1).cpu()), 0)
        return total_labels.numpy().flatten(), total_preds.numpy().flatten()

    def save_model(self, path):
        torch.save(self.model.state_dict(), os.path.join(path, 'HyperAttentionDTI.pt'))

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location='cpu'), strict=False)



class hyperparameter():
    def __init__(self):
        self.current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        self.Learning_rate = 5e-5
        self.Epoch = 80
        self.Batch_size = 64
        self.Resume = False
        self.Patience = 50
        self.FC_Dropout = 0.5
        self.test_split = 0.2
        self.validation_split = 0.2
        self.decay_interval = 10
        self.lr_decay = 0.5
        self.weight_decay = 1e-4
        self.embed_dim = 64

        self.protein_kernel = [4, 8, 12]
        self.drug_kernel = [4, 6, 8]
        self.conv = 40
        self.char_dim = 64


class AttentionDTI(nn.Module):
    def __init__(self,hp,
                 protein_MAX_LENGH = 1000,
                 drug_MAX_LENGH = 100):
        super(AttentionDTI, self).__init__()
        self.dim = hp.char_dim
        self.conv = hp.conv
        self.drug_MAX_LENGH = drug_MAX_LENGH
        self.drug_kernel = hp.drug_kernel
        self.protein_MAX_LENGH = protein_MAX_LENGH
        self.protein_kernel = hp.protein_kernel

        self.protein_embed = nn.Embedding(26, self.dim,padding_idx=0)
        self.drug_embed = nn.Embedding(65, self.dim,padding_idx=0)
        self.Drug_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels= self.conv,  kernel_size=self.drug_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels= self.conv*2,  kernel_size=self.drug_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv*2, out_channels= self.conv*4,  kernel_size=self.drug_kernel[2]),
            nn.ReLU(),
        )
        self.Drug_max_pool = nn.MaxPool1d(self.drug_MAX_LENGH-self.drug_kernel[0]-self.drug_kernel[1]-self.drug_kernel[2]+3)
        self.Protein_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=self.conv, kernel_size=self.protein_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2, kernel_size=self.protein_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv * 2, out_channels=self.conv * 4, kernel_size=self.protein_kernel[2]),
            nn.ReLU(),
        )
        self.Protein_max_pool = nn.MaxPool1d(self.protein_MAX_LENGH - self.protein_kernel[0] - self.protein_kernel[1] - self.protein_kernel[2] + 3)
        self.attention_layer = nn.Linear(self.conv*4,self.conv*4)
        self.protein_attention_layer = nn.Linear(self.conv * 4, self.conv * 4)
        self.drug_attention_layer = nn.Linear(self.conv * 4, self.conv * 4)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.leaky_relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(self.conv*8, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 1)

    def forward(self, drug, protein):
        drugembed = self.drug_embed(drug)
        proteinembed = self.protein_embed(protein)
        drugembed = drugembed.permute(0, 2, 1)
        proteinembed = proteinembed.permute(0, 2, 1)

        drugConv = self.Drug_CNNs(drugembed)
        proteinConv = self.Protein_CNNs(proteinembed)

        drug_att = self.drug_attention_layer(drugConv.permute(0, 2, 1))
        protein_att = self.protein_attention_layer(proteinConv.permute(0, 2, 1))

        d_att_layers = torch.unsqueeze(drug_att, 2).repeat(1, 1, proteinConv.shape[-1], 1)  # repeat along protein size
        p_att_layers = torch.unsqueeze(protein_att, 1).repeat(1, drugConv.shape[-1], 1, 1)  # repeat along drug size
        Atten_matrix = self.attention_layer(self.relu(d_att_layers + p_att_layers))
        Compound_atte = torch.mean(Atten_matrix, 2)
        Protein_atte = torch.mean(Atten_matrix, 1)
        Compound_atte = self.sigmoid(Compound_atte.permute(0, 2, 1))
        Protein_atte = self.sigmoid(Protein_atte.permute(0, 2, 1))

        drugConv = drugConv * 0.5 + drugConv * Compound_atte
        proteinConv = proteinConv * 0.5 + proteinConv * Protein_atte

        drugConv = self.Drug_max_pool(drugConv).squeeze(2)
        proteinConv = self.Protein_max_pool(proteinConv).squeeze(2)

        pair = torch.cat([drugConv, proteinConv], dim=1)
        pair = self.dropout1(pair)
        fully1 = self.leaky_relu(self.fc1(pair))
        fully1 = self.dropout2(fully1)
        fully2 = self.leaky_relu(self.fc2(fully1))
        fully2 = self.dropout3(fully2)
        fully3 = self.leaky_relu(self.fc3(fully2))
        predict = self.out(fully3)
        return predict