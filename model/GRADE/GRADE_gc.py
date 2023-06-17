import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Function
import numpy as np
from torch_geometric.nn import GCNConv
from utils import *
from torch_geometric.nn import global_mean_pool


def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)#/len(kernel_val)


def mmd_rbf_noaccelerate(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class GRADE_GC(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, dropout, device, disc="JS",mask=None):
        super(GRADE_GC, self).__init__()
        self.disc = disc
        self.n_classes = n_classes
        self.device = device
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(n_hidden[0], n_hidden[1]))
        for i in range(1,n_layers):
            self.layers.append(GCNConv(n_hidden[i], n_hidden[i+1]))
        # 图分类器有几层？
        self.fc =  nn.Sequential(nn.Linear(n_hidden[-1], 16), nn.ReLU(), nn.Linear(16, n_classes))
        self.dropout = nn.Dropout(p=dropout)

        if disc == "JS":
            self.discriminator = nn.Sequential(
                nn.Linear(sum(n_hidden[1:])+n_classes, 2)
            )
        elif disc == "C":
            self.discriminator = nn.Sequential(
                nn.Linear(n_hidden * n_layers + n_classes * 2, 2)
            )

        self.criterion = nn.BCEWithLogitsLoss()
        self.criterion_c = nn.CrossEntropyLoss()
        self.mask = mask

    def forward(self, src_data, tgt_data, alpha=1.0):
        features_s, edge_index_s, batch_s, labels_s = src_data.x, src_data.edge_index, src_data.batch, src_data.y
        features_t, edge_index_t, batch_t = tgt_data.x, tgt_data.edge_index, tgt_data.batch
        s_f = []
        t_f = []
        for i, layer in enumerate(self.layers):
            features_s = self.dropout(features_s)
            features_t = self.dropout(features_t)
            features_s = layer(features_s, edge_index_s)
            features_t = layer(features_t, edge_index_t)
            features_s_batch = global_mean_pool(features_s, batch_s)
            features_t_batch = global_mean_pool(features_t, batch_t)
            s_f.append(features_s_batch)
            t_f.append(features_t_batch)
        features_s_batch = self.fc(features_s_batch)
        features_t_batch = self.fc(features_t_batch)
        s_f.append(features_s_batch)
        t_f.append(features_t_batch)
        logit_s = features_s_batch
        if labels_s.dim()==1:
            labels_s = to_onehot(labels_s,num_classes=self.n_classes,device=self.device)

        class_loss = self.criterion(logit_s, labels_s)

        domain_loss = 0.
        s_f = torch.cat(s_f, dim=1)
        t_f = torch.cat(t_f, dim=1)
        # JS散度是用对抗思想达到的
        if self.disc == "JS":
            domain_preds = self.discriminator(ReverseLayerF.apply(torch.cat([s_f, t_f], dim=0), alpha))
            domain_labels = np.array([0] * s_f.shape[0] + [1] * t_f.shape[0])
            domain_labels = torch.tensor(domain_labels, requires_grad=False, dtype=torch.long, device=features_s.device)
            domain_loss = self.criterion_c(domain_preds, domain_labels)
        # MMD 只是随便地用了两个网络中shape较小者的节点数量参与运算
        elif self.disc == "MMD":
            mind = min(s_f.shape[0], t_f.shape[0])
            domain_loss = mmd_rbf_noaccelerate(s_f[:mind], t_f[:mind])
        elif self.disc == "C":
            ratio = 8
            s_l_f = torch.cat([s_f, ratio * self.one_hot_embedding(labels_s)], dim=1)
            t_l_f = torch.cat([t_f, ratio * F.softmax(features_t, dim=1)], dim=1)
            domain_preds = self.discriminator(ReverseLayerF.apply(torch.cat([s_l_f, t_l_f], dim=0), alpha))
            domain_labels = np.array([0] * features_s.shape[0] + [1] * features_t.shape[0])
            domain_labels = torch.tensor(domain_labels, requires_grad=False, dtype=torch.long, device=features_s.device)
            domain_loss = self.criterion(domain_preds, domain_labels)
        else:
            domain_loss = 0.
        # NOTE domain loss原始权重是0.01，会造成domain loss非常大
        loss = class_loss + domain_loss * 0.01
        return loss, class_loss, domain_loss

    def inference(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
        x_batch = F.dropout(global_mean_pool(x, batch), p=0.1, training=self.training)
        logits = F.sigmoid(self.fc(x_batch))
        return logits


