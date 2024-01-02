import torch
import torch.nn as nn
import torch.nn.functional as F

def ContrastiveLoss(anchor, positive, negatives, margin, temperature=1.0):
    positive_distance = F.pairwise_distance(anchor, positive)

    anchor_expanded = anchor.unsqueeze(1)  # [batch_size, 1, embedding_size]
    negatives_expanded = negatives.unsqueeze(0)  # [1, batch_size, embedding_size]
    negative_distances = torch.sqrt(torch.sum((anchor_expanded - negatives_expanded) ** 2, dim=2))

    positive_distance /= temperature
    negative_distances /= temperature

    losses = torch.clamp(margin + positive_distance.unsqueeze(1) - negative_distances, min=0)

    batch_size = anchor.size(0)
    num_negatives = negatives.size(0)
    mask = torch.zeros(batch_size, num_negatives, dtype=torch.bool, device=anchor.device)
    for i in range(batch_size):
        mask[i, i::batch_size] = True

    losses = losses.masked_fill(mask, 0)

    return losses.mean()

class CompositeLoss(nn.Module):
    def __init__(self, loss_func_wt, margin=1.0, temperature=1.0):
        super(CompositeLoss, self).__init__()
        if 'MSE' in loss_func_wt.keys():
            self.mse_weight = float(loss_func_wt['MSE'])
        if 'CLS' in loss_func_wt.keys():
            self.classification_weight = float(loss_func_wt['CLS'])
        if 'CL' in loss_func_wt.keys():
            self.contrastive_weight = float(loss_func_wt['CL'])

        self.temperature = temperature
        self.margin = margin
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, output, query, support, reg_labels, cls_labels):
        output1, siams_output = output
        mol1, mol1_ = query[0], query[1]
        mol2, mol2_ = support[0], support[1]
        # MSE loss
        if self.mse_weight > 0:
            mse_loss = self.mse_loss(output1, reg_labels)
        else:
            mse_loss = 0

        # contrastive loss
        if self.contrastive_weight > 0:
            contrastive_loss = ContrastiveLoss(mol1_, mol1, 
                                               torch.concat([mol1, mol2_, mol2, mol2_]),
                                               self.margin, self.temperature)
        else:
            contrastive_loss = 0

        # classification loss
        if self.classification_weight > 0:
            classification_loss = self.bce_loss(siams_output.squeeze(), cls_labels)
        else:
            classification_loss = 0
            
        final_loss =  self.mse_weight * mse_loss +\
                      self.contrastive_weight * contrastive_loss +\
                      self.classification_weight * classification_loss

        return final_loss, mse_loss, contrastive_loss, classification_loss