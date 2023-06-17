import math
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.autograd import Function
import torch.nn as nn
import numpy as np
from torch_geometric.nn import GCNConv

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GRADE_LP(nn.Module):
    def __init__(self, args, num_item_d1, num_user_d1, num_item_d2, num_user_d2, adj_ds=None, adj_dt=None, feats_s=None, feats_t=None):
        super(GRADE_LP, self).__init__()
        self.num_gcn_layers = args.num_gcn_layers
        self.is_bipart_graph = args.is_bipart_graph
        self.device = args.device
        self.num_item_d1 = num_item_d1
        self.num_user_d1 = num_user_d1
        self.num_item_d2 = num_item_d2
        self.num_user_d2 = num_user_d2
        self.adj_ds = adj_ds
        self.adj_dt = adj_dt
        self.V_d1 = torch.tensor(feats_s, requires_grad=False, dtype=torch.float, device=self.device)
        self.V_d2 = torch.tensor(feats_t, requires_grad=False, dtype=torch.float, device=self.device)

        # rewrite
        layers = [feats_s.shape[1]] + args.hidden_dims
        self.gc = [GCNConv(layers[i], layers[i+1]).to(device=self.device) for i in range(self.num_gcn_layers)]

        # 一层FC去做连边二分类
        self.layers = nn.Sequential(
            nn.Linear(sum(args.hidden_dims)*2, 1),
            nn.Sigmoid()
        ).to(device=self.device)

        # 二分类损失
        self.criterion = nn.BCELoss()

        if self.is_bipart_graph:
            # 二部图，两类节点各有自己的域判别器
            self.discriminator_u = nn.Sequential(
                nn.Linear(sum(args.hidden_dims), 1),
                nn.Sigmoid()
            ).to(device=self.device)
            self.discriminator_i = nn.Sequential(
                nn.Linear(sum(args.hidden_dims), 1),
                nn.Sigmoid()
            ).to(device=self.device)
        else:
            self.discriminator = nn.Sequential(
                nn.Linear(sum(args.hidden_dims), 1),
                nn.Sigmoid()
            ).to(device=self.device)

    def forward(self, train_data_s, train_data_t, alpha=1.0):
        """
        这个模型几个点
        1. batch训练模式
        :param train_data_s:
        :param train_data_t:
        :param alpha:
        :return:
        """
        # label_s/t是0/1表示是正样本还是负样本
        user_s, item_s, labels_s = train_data_s[:, 0], train_data_s[:, 1], train_data_s[:, 2]
        user_t, item_t, labels_t = train_data_t[:, 0], train_data_t[:, 1], train_data_t[:, 2]
        x_ds = self.V_d1
        x_dt = self.V_d2

        user_feats_ds = []
        item_feats_ds = []
        user_feats_dt = []
        item_feats_dt = []

        if self.is_bipart_graph:
            for i in range(self.num_gcn_layers):
                x_ds = F.relu(self.gc[i](x_ds, self.adj_ds))
                x_dt = F.relu(self.gc[i](x_dt, self.adj_dt))
                user_feats_ds.append(x_ds[user_s])
                # 商品节点上的下标偏移没加上，之前只加到图的编号中了
                item_feats_ds.append(x_ds[item_s + self.num_user_d1])
                user_feats_dt.append(x_dt[user_t])
                item_feats_dt.append(x_dt[item_t + self.num_user_d2])
            user_feats_ds = torch.cat(user_feats_ds, dim=1)
            item_feats_ds = torch.cat(item_feats_ds, dim=1)
            user_feats_dt = torch.cat(user_feats_dt, dim=1)
            item_feats_dt = torch.cat(item_feats_dt, dim=1)
            logit_s = self.layers(torch.cat([user_feats_ds, item_feats_ds], dim=1))
            logit_t = self.layers(torch.cat([user_feats_dt, item_feats_dt], dim=1))
            loss_s = self.criterion(logit_s.view(-1), labels_s.float())
            loss_t = self.criterion(logit_t.view(-1), labels_t.float())
            loss = loss_s + loss_t

            user_domain_preds = self.discriminator_u(ReverseLayerF.apply(torch.cat([user_feats_ds, user_feats_dt], dim=0), alpha))
            item_domain_preds = self.discriminator_i(ReverseLayerF.apply(torch.cat([item_feats_ds, item_feats_dt], dim=0), alpha))
            domain_labels = np.array([0] * user_feats_ds.shape[0] + [1] * user_feats_dt.shape[0])
            domain_labels = torch.tensor(domain_labels, requires_grad=False, dtype=torch.float, device=self.device)
            domain_loss = self.criterion(item_domain_preds.view(-1), domain_labels) + self.criterion(user_domain_preds.view(-1), domain_labels)

        else:
            for i in range(self.num_gcn_layers):
                x_ds = F.relu(self.gc[i](x_ds, self.adj_ds))
                x_dt = F.relu(self.gc[i](x_dt, self.adj_dt))
                user_feats_ds.append(x_ds[user_s])
                # 商品节点上的下标偏移没加上，之前只加到图的编号中了
                item_feats_ds.append(x_ds[item_s])
                user_feats_dt.append(x_dt[user_t])
                item_feats_dt.append(x_dt[item_t])
            user_feats_ds = torch.cat(user_feats_ds, dim=1)
            item_feats_ds = torch.cat(item_feats_ds, dim=1)
            user_feats_dt = torch.cat(user_feats_dt, dim=1)
            item_feats_dt = torch.cat(item_feats_dt, dim=1)
            logit_s = self.layers(torch.cat([user_feats_ds, item_feats_ds], dim=1))
            logit_t = self.layers(torch.cat([user_feats_dt, item_feats_dt], dim=1))
            loss_s = self.criterion(logit_s.view(-1), labels_s.float())
            loss_t = self.criterion(logit_t.view(-1), labels_t.float())
            loss = loss_s + loss_t

            x_ds_batch = torch.cat([user_feats_ds, item_feats_ds],dim=0)
            x_dt_batch = torch.cat([user_feats_dt, item_feats_dt],dim=0)
            source_domain_preds = self.discriminator(ReverseLayerF.apply(x_ds_batch,alpha))
            target_domain_preds = self.discriminator(ReverseLayerF.apply(x_dt_batch,alpha))
            source_domain_labels = np.array([0] * source_domain_preds.shape[0])
            source_domain_labels = torch.tensor(source_domain_labels, requires_grad=False, dtype=torch.float).to(self.device)
            target_domain_labels = np.array([1] * target_domain_preds.shape[0])
            target_domain_labels = torch.tensor(target_domain_labels, requires_grad=False, dtype=torch.float).to(self.device)
            domain_loss = self.criterion(source_domain_preds.view(-1), source_domain_labels) + self.criterion(target_domain_preds.view(-1), target_domain_labels)

        return loss + domain_loss * 0.1

    def inference(self, user, item):
        x_dt = self.V_d2
        user_feats_dt = []
        item_feats_dt = []
        for i in range(self.num_gcn_layers):
            x_dt = F.relu(self.gc[i](x_dt, self.adj_dt))
            user_feats_dt.append(x_dt[user])
            if self.is_bipart_graph:
                item_feats_dt.append(x_dt[item + self.num_user_d2])
            else:
                item_feats_dt.append(x_dt[item])

        return self.layers(torch.cat([torch.cat(user_feats_dt, dim=1), torch.cat(item_feats_dt, dim=1)], dim=1))
