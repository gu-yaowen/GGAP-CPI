import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from MoleculeACE import calc_rmse


class GraphDTA():
    def __init__(self, args, logger):
        self.args = args
        # based on the reported performance in GraphDTA paper, 
        # we use GIN as the best-performing model
        self.model = GINConvNet()
        if args.gpu is not None:
            self.device = torch.device("cuda:%d" % args.gpu)
            logger.info("Using GPU: %d" % args.gpu)
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU")
        self.model = self.model.to(self.device)

        self.TRAIN_BATCH_SIZE = 512
        self.TEST_BATCH_SIZE = 512
        self.LR = 0.0005
        self.LOG_INTERVAL = 20
        self.NUM_EPOCHS = 2
        logger.info("Batch size %d, Learning rate %f, Num epochs %d" %
                    (self.TRAIN_BATCH_SIZE, self.LR, self.NUM_EPOCHS))
        
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.LR)
        self.best_rmse = 1000
        self.best_model = None

    def train(self, args, logger, train_loader, val_loader=None):
        for epoch in range(self.NUM_EPOCHS):
            self.model.train()
            for batch_idx, data in enumerate(train_loader):
                data = data.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss(output, data.y.view(-1, 1).float().to(self.device))
                loss.backward()
                self.optimizer.step()
                if batch_idx % self.LOG_INTERVAL == 0:
                    logger.info('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                        batch_idx * len(data.x), len(train_loader.dataset), 
                                        100. * batch_idx / len(train_loader), loss.item()))
            if len(val_loader) > 0:
                label, pred = self.predict(val_loader)
            else:
                label, pred = self.predict(train_loader)
            rmse = calc_rmse(label, pred)
            if rmse < self.best_rmse:
                self.best_rmse = rmse
                self.save_model(args.save_path)
                self.best_model = self.model
            logger.info('Epoch: {}, RMSE: {:.4f}, Best RMSE: {:.4f}'.format(epoch, rmse, self.best_rmse))
        return self.best_model
    
    def predict(self, loader):
        self.model.eval()
        total_preds = torch.Tensor()
        total_labels = torch.Tensor()
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                output = self.model(data)
                total_preds = torch.cat((total_preds, output.cpu()), 0)
                total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
        return total_labels.numpy().flatten(), total_preds.numpy().flatten()

    def save_model(self, path):
        torch.save(self.model.state_dict(), os.path.join(path, 'GraphDTA.pt'))


# GINConv model
class GINConvNet(torch.nn.Module):
    def __init__(self, n_output=1,num_features_xd=78, num_features_xt=25,
                 n_filters=32, embed_dim=128, output_dim=128, dropout=0.2):

        super(GINConvNet, self).__init__()

        dim = 32
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output
        # convolution layers
        nn1 = Sequential(Linear(num_features_xd, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = torch.nn.BatchNorm1d(dim)

        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = torch.nn.BatchNorm1d(dim)

        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = torch.nn.BatchNorm1d(dim)

        nn4 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv4 = GINConv(nn4)
        self.bn4 = torch.nn.BatchNorm1d(dim)

        nn5 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv5 = GINConv(nn5)
        self.bn5 = torch.nn.BatchNorm1d(dim)

        self.fc1_xd = Linear(dim, output_dim)

        # 1D convolution on protein sequence
        self.embedding_xt = nn.Embedding(num_features_xt + 1, embed_dim)
        self.conv_xt_1 = nn.Conv1d(in_channels=1000, out_channels=n_filters, kernel_size=8)
        self.fc1_xt = nn.Linear(32*121, output_dim)

        # combined layers
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, self.n_output)        # n_output = 1 for regression task

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        target = data.target

        x = F.relu(self.conv1(x, edge_index))
        x = self.bn1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.bn2(x)
        x = F.relu(self.conv3(x, edge_index))
        x = self.bn3(x)
        x = F.relu(self.conv4(x, edge_index))
        x = self.bn4(x)
        x = F.relu(self.conv5(x, edge_index))
        x = self.bn5(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.fc1_xd(x))
        x = F.dropout(x, p=0.2, training=self.training)

        embedded_xt = self.embedding_xt(target)
        conv_xt = self.conv_xt_1(embedded_xt)
        # flatten
        xt = conv_xt.view(-1, 32 * 121)
        xt = self.fc1_xt(xt)

        # concat
        xc = torch.cat((x, xt), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        return out
