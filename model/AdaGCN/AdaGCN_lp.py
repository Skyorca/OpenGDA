import sys
sys.path.append('..')
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from base_model.ppmi_conv import PPMIConv
from torch.autograd import Function
from utils import *

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
            conv_layers.append(PPMIConv(hidden_dims[i], hidden_dims[i + 1],path_len=k,device=device))
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

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class ADAGCN_LP(nn.Module):
    def __init__(self,use_igcn, encoder_dim, num_gcn_layers,hidden_dims:list,device, dropout,path_len,coeff,is_bipart_graph):
        super(ADAGCN_LP, self).__init__()
        self.is_bipart_graph = is_bipart_graph
        self.coeff = coeff
        self.num_gcn_layers = num_gcn_layers
        self.hidden_dims = hidden_dims
        self.device = device
        assert len(self.hidden_dims) == self.num_gcn_layers+1
        self.use_igcn = use_igcn
        self.encoder_dim = encoder_dim
        if self.use_igcn:
            self.encoder = IGCN(num_gcn_layers=num_gcn_layers,hidden_dims=hidden_dims,dropout=dropout,k=path_len,device=self.device)
        else:
            self.encoder = GCN(num_gcn_layers=num_gcn_layers,hidden_dims=hidden_dims,dropout=dropout)

        self.cls_model = nn.Sequential(
            nn.Linear(encoder_dim*2, 1),
            nn.Sigmoid()
        )

        if self.is_bipart_graph:
            # 二部图，两类节点各有自己的域判别器,和连边分类一样都要加sigmoid这样就能用同一套loss
            self.discriminator_u = nn.Sequential(
                nn.Linear(self.encoder_dim, 1),
                nn.Sigmoid()
            )
            self.discriminator_i = nn.Sequential(
                nn.Linear(self.encoder_dim, 1),
                nn.Sigmoid()
            )
        else:
            # TODO: 判别器是否要设计得强一些
            self.discriminator = nn.Sequential(
                nn.Linear(self.encoder_dim, 1),
                nn.Sigmoid()
            )


        self.criterion = nn.BCELoss()

    def forward_critic(self,train_data_s, train_data_t, num_user_ds,num_user_dt,adj_ds,adj_dt, feats_s,feats_t):
        user_s, item_s, labels_s = train_data_s[:, 0], train_data_s[:, 1], train_data_s[:, 2]
        user_t, item_t, labels_t = train_data_t[:, 0], train_data_t[:, 1], train_data_t[:, 2]
        x_ds = self.encoder(feats_s, adj_ds,"source")
        x_dt = self.encoder(feats_t, adj_dt,"target")
        if self.is_bipart_graph:
            user_feats_ds = x_ds[user_s]
            # 商品节点上的下标偏移没加上，之前只加到图的编号中了
            item_feats_ds = x_ds[item_s + num_user_ds]
            user_feats_dt = x_dt[user_t]
            item_feats_dt = x_dt[item_t + num_user_dt]
            critic_user_s = self.discriminator_u(user_feats_ds).reshape(-1)
            critic_user_t = self.discriminator_u(user_feats_dt).reshape(-1)
            gp_u = gradient_penalty(self.discriminator_u, user_feats_ds,user_feats_dt, self.device)
            loss_critic_u = (
                    -torch.abs(torch.mean(critic_user_s) - torch.mean(critic_user_t)) + self.coeff['LAMBDA_GP'] * gp_u
            )
            critic_item_s = self.discriminator_i(item_feats_ds).reshape(-1)
            critic_item_t = self.discriminator_i(item_feats_dt).reshape(-1)
            gp_i = gradient_penalty(self.discriminator_i, item_feats_ds,item_feats_dt, self.device)
            loss_critic_i = (
                    -torch.abs(torch.mean(critic_item_s) - torch.mean(critic_item_t)) + self.coeff['LAMBDA_GP'] * gp_i
            )
            loss_critic = loss_critic_i + loss_critic_u
        else:
            user_feats_ds = x_ds[user_s]
            user_feats_dt = x_dt[user_t]
            item_feats_ds = x_ds[item_s]
            item_feats_dt = x_dt[item_t]
            x_ds_batch = torch.cat([user_feats_ds, item_feats_ds],dim=0)
            x_dt_batch = torch.cat([user_feats_dt, item_feats_dt],dim=0)
            critic_ds = self.discriminator(x_ds_batch).reshape(-1)
            critic_dt = self.discriminator(x_dt_batch).reshape(-1)
            gp = gradient_penalty(self.discriminator, x_ds_batch, x_dt_batch, self.device)
            loss_critic = (
                    -torch.abs(torch.mean(critic_ds) - torch.mean(critic_dt)) + self.coeff['LAMBDA_GP'] * gp
            )
        return loss_critic

    def forward(self,train_data_s, train_data_t, num_user_ds,num_user_dt,adj_ds,adj_dt, feats_s,feats_t):
        self.num_user_ds = num_user_ds
        self.num_user_dt = num_user_dt
        self.adj_dt = adj_dt
        self.feats_t = feats_t
        # label_s/t是0/1表示是正样本还是负样本
        user_s, item_s, labels_s = train_data_s[:, 0], train_data_s[:, 1], train_data_s[:, 2]
        user_t, item_t, labels_t = train_data_t[:, 0], train_data_t[:, 1], train_data_t[:, 2]
        x_ds = self.encoder(feats_s, adj_ds,"source")
        x_dt = self.encoder(feats_t, adj_dt,"target")

        if self.is_bipart_graph:
            # index embeddings
            user_feats_ds = x_ds[user_s]
            item_feats_ds = x_ds[item_s + self.num_user_ds]
            user_feats_dt = x_dt[user_t]
            item_feats_dt = x_dt[item_t + self.num_user_dt]
            # concat for edge rep and predict
            logit_s = self.cls_model(torch.cat([user_feats_ds, item_feats_ds], dim=1))
            logit_t = self.cls_model(torch.cat([user_feats_dt, item_feats_dt], dim=1))
            clf_loss_s = self.criterion(logit_s.view(-1), labels_s.float())
            clf_loss_t = self.criterion(logit_t.view(-1), labels_t.float())
            clf_loss = clf_loss_s + clf_loss_t
            # WGAN domain loss
            critic_user_s = self.discriminator_u(user_feats_ds).reshape(-1)
            critic_user_t = self.discriminator_u(user_feats_dt).reshape(-1)
            loss_critic_u = torch.abs(torch.mean(critic_user_s) - torch.mean(critic_user_t))

            critic_item_s = self.discriminator_i(item_feats_ds).reshape(-1)
            critic_item_t = self.discriminator_i(item_feats_dt).reshape(-1)
            loss_critic_i = torch.abs(torch.mean(critic_item_s) - torch.mean(critic_item_t))

            domain_loss = loss_critic_i+loss_critic_u
        else:
            user_feats_ds = x_ds[user_s]
            user_feats_dt = x_dt[user_t]
            item_feats_ds = x_ds[item_s]
            item_feats_dt = x_dt[item_t]
            logit_s = self.cls_model(torch.cat([user_feats_ds, item_feats_ds], dim=1))
            logit_t = self.cls_model(torch.cat([user_feats_dt, item_feats_dt], dim=1))
            clf_loss_s = self.criterion(logit_s.view(-1), labels_s.float())
            clf_loss_t = self.criterion(logit_t.view(-1), labels_t.float())
            clf_loss = clf_loss_s + clf_loss_t
            # 整体混淆source target在batch中的节点
            x_ds_batch = torch.cat([user_feats_ds, item_feats_ds], dim=0)
            x_dt_batch = torch.cat([user_feats_dt, item_feats_dt], dim=0)
            critic_ds = self.discriminator(x_ds_batch).reshape(-1)
            critic_dt = self.discriminator(x_dt_batch).reshape(-1)
            domain_loss = torch.abs(torch.mean(critic_ds) - torch.mean(critic_dt))
        print(clf_loss, domain_loss)
        return clf_loss+self.coeff['LAMBDA']*domain_loss

    def inference(self, user_idx, item_idx):
        x_dt = self.encoder(self.feats_t, self.adj_dt,"target")
        user_feats_dt = x_dt[user_idx]
        if self.is_bipart_graph:
            item_feats_dt = x_dt[item_idx + self.num_user_dt]
        else:
            item_feats_dt = x_dt[item_idx]
        return self.cls_model(torch.cat([user_feats_dt, item_feats_dt], dim=1))