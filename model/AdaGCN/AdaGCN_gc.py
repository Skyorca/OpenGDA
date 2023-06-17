import sys
sys.path.append('..')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from base_model.ppmi_conv import PPMIConv
from torch.autograd import Function
from utils import *
from torch_geometric.nn import global_mean_pool


class GCN(nn.Module):
    def __init__(self, num_gcn_layers,hidden_dims,dropout):
        super(GCN, self).__init__()
        conv_layers = []
        for i in range(num_gcn_layers):
            conv_layers.append(GCNConv(hidden_dims[i], hidden_dims[i + 1]))
        self.conv_layers = nn.ModuleList(conv_layers)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index):
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x, edge_index)
            if i < len(self.conv_layers) - 1:
                x = self.relu(x)
                x = self.dropout(x)
        return x

class IGCN(nn.Module):
    def __init__(self, num_gcn_layers,hidden_dims,dropout,k,device):
        super(IGCN, self).__init__()
        self.k = k
        conv_layers = []
        for i in range(num_gcn_layers):
            conv_layers.append(PPMIConv(hidden_dims[i], hidden_dims[i + 1],path_len=k,device=device,cached=False))
        self.conv_layers = nn.ModuleList(conv_layers)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, cache_name):
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x, edge_index, cache_name)
            if i < len(self.conv_layers) - 1:
                x = self.relu(x)
                x = self.dropout(x)
        return x


class ADAGCN_GC(nn.Module):
    def __init__(self,use_igcn, encoder_dim, num_gcn_layers,hidden_dims:list,num_classes, device, dropout,path_len,coeff):
        super(ADAGCN_GC, self).__init__()
        self.coeff = coeff
        self.num_gcn_layers = num_gcn_layers
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.device = device
        assert len(self.hidden_dims) == self.num_gcn_layers+1
        self.use_igcn = use_igcn
        self.encoder_dim = encoder_dim
        if self.use_igcn:
            self.encoder = IGCN(num_gcn_layers=num_gcn_layers,hidden_dims=hidden_dims,dropout=dropout,k=path_len,device=self.device)
        else:
            self.encoder = GCN(num_gcn_layers=num_gcn_layers,hidden_dims=hidden_dims,dropout=dropout)

        self.cls_model = nn.Sequential(
            nn.Linear(encoder_dim, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes),
            nn.Sigmoid()
        )
        # TODO: 判别器是否要设计得强一些
        self.discriminator = nn.Sequential(
            nn.Linear(self.encoder_dim, 1),
            nn.Sigmoid()
        )
        self.criterion = nn.BCELoss()

    def forward_critic(self,src_data, tgt_data):
        features_s, edge_index_s, batch_s, labels_s = src_data.x, src_data.edge_index, src_data.batch, src_data.y
        features_t, edge_index_t, batch_t = tgt_data.x, tgt_data.edge_index, tgt_data.batch
        if self.use_igcn:
            x_ds = self.encoder(features_s, edge_index_s,"source")
            x_dt = self.encoder(features_t, edge_index_t,"source")
        else:
            x_ds = self.encoder(features_s, edge_index_s)
            x_dt = self.encoder(features_t, edge_index_t)
        x_ds_batch = global_mean_pool(x_ds, batch_s)
        x_dt_batch = global_mean_pool(x_dt, batch_t)
        critic_ds = self.discriminator(x_ds_batch).reshape(-1)
        critic_dt = self.discriminator(x_dt_batch).reshape(-1)
        gp = gradient_penalty(self.discriminator, x_ds_batch, x_dt_batch, self.device)
        loss_critic = (
                -torch.abs(torch.mean(critic_ds) - torch.mean(critic_dt)) + self.coeff['LAMBDA_GP'] * gp
        )
        return loss_critic

    def forward(self,src_data, tgt_data):
        features_s, edge_index_s, batch_s, labels_s = src_data.x, src_data.edge_index, src_data.batch, src_data.y
        features_t, edge_index_t, batch_t = tgt_data.x, tgt_data.edge_index, tgt_data.batch
        if self.use_igcn:
            x_ds = self.encoder(features_s, edge_index_s,"source")
            x_dt = self.encoder(features_t, edge_index_t,"source")
        else:
            x_ds = self.encoder(features_s, edge_index_s)
            x_dt = self.encoder(features_t, edge_index_t)
        x_ds_batch = global_mean_pool(x_ds, batch_s)
        x_dt_batch = global_mean_pool(x_dt, batch_t)
        # source graph classification loss
        if labels_s.dim() == 1:
            labels_s = to_onehot(labels_s, num_classes=self.num_classes, device=self.device)
        src_logits = self.cls_model(x_ds_batch)
        clf_loss = self.criterion(src_logits, labels_s)
        critic_ds = self.discriminator(x_ds_batch).reshape(-1)
        critic_dt = self.discriminator(x_dt_batch).reshape(-1)
        domain_loss = torch.abs(torch.mean(critic_ds) - torch.mean(critic_dt))
        print(clf_loss, domain_loss)
        return clf_loss + self.coeff['LAMBDA'] * domain_loss, clf_loss, domain_loss

    def inference(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if self.use_igcn:
            x_d = self.encoder(x, edge_index,cache_name='target')
        else:
            x_d = self.encoder(x, edge_index)
        x_d_batch = global_mean_pool(x_d, batch)
        logits = self.cls_model(x_d_batch)
        return logits