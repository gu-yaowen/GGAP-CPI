import torch
import torch.nn as nn
import torch.nn.functional as F


class CompositeLoss(nn.Module):
    def __init__(self, args, loss_func_wt):
        super(CompositeLoss, self).__init__()
        self.args = args
        self.mse_weight = float(loss_func_wt['MSE']) if 'MSE' in loss_func_wt.keys() else 0
        self.classification_weight = float(loss_func_wt['CLS']) if 'CLS' in loss_func_wt.keys() else 0

        if self.args.dataset_type == 'regression':
            self.sup_loss = nn.MSELoss(reduction='mean')
        elif self.args.dataset_type == 'classification':
            self.sup_loss = nn.BCEWithLogitsLoss(reduction='mean', 
                                             pos_weight=torch.tensor([self.args.pos_weight]))
        
    def forward(self, output, reg_labels, cls_labels):

        if self.args.dataset_type == 'regression':
            mask_reg = [i for i in range(len(reg_labels)) if reg_labels[i] != 999]
            output = output[mask_reg]
            reg_labels = reg_labels[mask_reg]
            mse_loss = self.sup_loss(output, reg_labels)
            return mse_loss, mse_loss, None

        elif self.args.dataset_type == 'classification':
            cls_loss = self.sup_loss(output.squeeze(), cls_labels.squeeze())
            return cls_loss, None, cls_loss
