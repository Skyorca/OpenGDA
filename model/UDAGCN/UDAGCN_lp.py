
import sys
sys.path.append("..")
from base_model.cached_gcn_conv import CachedGCNConv
from base_model.ppmi_conv import PPMIConv
import numpy as np
import torch
import torch.functional as F
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function


class GNN(torch.nn.Module):
    def __init__(self, num_gcn_layers,hidden_dims, base_model=None, type="gcn",device='cpu', **kwargs):
        super(GNN, self).__init__()
        if base_model is None:
            weights = [None, None]
            biases = [None, None]
        else:
            # 重要：这里PPMIGCN和普通的GCN是共享参数的
            weights = [conv_layer.weight for conv_layer in base_model.conv_layers]
            biases = [conv_layer.bias for conv_layer in base_model.conv_layers]


        self.dropout_layers = [nn.Dropout(0.1) for _ in weights]
        self.type = type

        model_cls = PPMIConv if type == "ppmi" else CachedGCNConv

        conv_layers = []
        for i in range(num_gcn_layers):
            conv_layers.append(model_cls(hidden_dims[i], hidden_dims[i+1],
                     weight=weights[i],
                     bias=biases[i],
                     device=device,
                      **kwargs))
        self.conv_layers = nn.ModuleList(conv_layers)


    def forward(self, x, edge_index, cache_name):
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x, edge_index, cache_name)
            if i < len(self.conv_layers) - 1:
                x = F.relu(x)
                x = self.dropout_layers[i](x)
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

class Attention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.dense_weight = nn.Linear(in_channels, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs):
        stacked = torch.stack(inputs, dim=1)
        weights = F.softmax(self.dense_weight(stacked), dim=1)
        outputs = torch.sum(stacked * weights, dim=1)
        return outputs



class UDAGCN_LP(nn.Module):
    def __init__(self,use_UDAGCN,encoder_dim, num_gcn_layers,hidden_dims:list,device, is_bipart_graph, path_len=3):
        super(UDAGCN_LP, self).__init__()
        self.is_bipart_graph = is_bipart_graph
        self.num_gcn_layers = num_gcn_layers
        self.hidden_dims = hidden_dims
        self.device = device
        assert len(self.hidden_dims) == self.num_gcn_layers+1
        self.use_UDAGCN = use_UDAGCN
        self.encoder_dim = encoder_dim

        self.encoder = GNN(type="gcn",num_gcn_layers=num_gcn_layers,hidden_dims=hidden_dims)
        # TODO 重要参数是PPMI的K
        if self.use_UDAGCN:
            self.ppmi_encoder = GNN(base_model=self.encoder, type="ppmi", path_len=path_len,num_gcn_layers=num_gcn_layers,hidden_dims=hidden_dims,device=self.device)

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
        self.att_model = Attention(self.encoder_dim)
        self.criterion = nn.BCELoss()

    def gcn_encode(self, x, edge_index, cache_name, mask=None):
        encoded_output = self.encoder(x, edge_index, cache_name)
        if mask is not None:
            encoded_output = encoded_output[mask]
        return encoded_output

    def ppmi_encode(self, x, edge_index, cache_name, mask=None):
        encoded_output = self.ppmi_encoder(x, edge_index, cache_name)
        if mask is not None:
            encoded_output = encoded_output[mask]
        return encoded_output

    def encode(self, x, edge_index, cache_name, mask=None):
        gcn_output = self.gcn_encode(x, edge_index, cache_name, mask)
        if self.use_UDAGCN:
            ppmi_output = self.ppmi_encode(x, edge_index, cache_name, mask)
            outputs = self.att_model([gcn_output, ppmi_output])
            return outputs
        else:
            return gcn_output

    def forward(self, train_data_s, train_data_t, num_user_ds, num_user_dt, adj_ds, adj_dt, feats_s, feats_t,alpha=1.0):
        """
        这个模型几个点
        1. batch训练模式
        :param train_data_s:
        :param train_data_t:
        :param alpha:
        :return:
        """

        self.adj_dt = adj_dt
        self.feats_t = feats_t
        # label_s/t是0/1表示是正样本还是负样本
        user_s, item_s, labels_s = train_data_s[:, 0], train_data_s[:, 1], train_data_s[:, 2]
        user_t, item_t, labels_t = train_data_t[:, 0], train_data_t[:, 1], train_data_t[:, 2]
        x_ds = self.encode(feats_s, adj_ds, "source")
        x_dt = self.encode(feats_t, adj_dt, "target")
        if self.is_bipart_graph:
            self.num_user_ds = num_user_ds
            self.num_user_dt = num_user_dt
            user_feats_ds = x_ds[user_s]
            # 商品节点上的下标偏移没加上，之前只加到图的编号中了
            item_feats_ds = x_ds[item_s + self.num_user_ds]
            user_feats_dt = x_dt[user_t]
            item_feats_dt = x_dt[item_t + self.num_user_dt]
            logit_s = self.cls_model(torch.cat([user_feats_ds, item_feats_ds], dim=1))
            logit_t = self.cls_model(torch.cat([user_feats_dt, item_feats_dt], dim=1))
            clf_loss_s = self.criterion(logit_s.view(-1), labels_s.float())
            clf_loss_t = self.criterion(logit_t.view(-1), labels_t.float())
            clf_loss = clf_loss_s + clf_loss_t

            user_domain_preds = self.discriminator_u(ReverseLayerF.apply(torch.cat([user_feats_ds, user_feats_dt], dim=0), alpha))
            item_domain_preds = self.discriminator_i(ReverseLayerF.apply(torch.cat([item_feats_ds, item_feats_dt], dim=0), alpha))
            domain_labels = np.array([0] * user_feats_ds.shape[0] + [1] * user_feats_dt.shape[0])
            domain_labels = torch.tensor(domain_labels, requires_grad=False, dtype=torch.float).to(self.device)
            domain_loss = self.criterion(item_domain_preds.view(-1), domain_labels) + self.criterion(user_domain_preds.view(-1), domain_labels)
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
            x_ds_batch = torch.cat([user_feats_ds, item_feats_ds],dim=0)
            x_dt_batch = torch.cat([user_feats_dt, item_feats_dt],dim=0)
            source_domain_preds = self.discriminator(ReverseLayerF.apply(x_ds_batch,alpha))
            target_domain_preds = self.discriminator(ReverseLayerF.apply(x_dt_batch,alpha))
            source_domain_labels = np.array([0] * source_domain_preds.shape[0])
            source_domain_labels = torch.tensor(source_domain_labels, requires_grad=False, dtype=torch.float).to(self.device)
            target_domain_labels = np.array([1] * target_domain_preds.shape[0])
            target_domain_labels = torch.tensor(target_domain_labels, requires_grad=False, dtype=torch.float).to(self.device)
            domain_loss = self.criterion(source_domain_preds.view(-1), source_domain_labels) + self.criterion(target_domain_preds.view(-1), target_domain_labels)

        return clf_loss + domain_loss * 0.1

    def inference(self, user_idx, item_idx):
        x_dt = self.encode(self.feats_t, self.adj_dt,cache_name='target')
        user_feats_dt = x_dt[user_idx]
        if self.is_bipart_graph:
            item_feats_dt = x_dt[item_idx + self.num_user_dt]
        else:
            item_feats_dt = x_dt[item_idx]
        return self.cls_model(torch.cat([user_feats_dt, item_feats_dt], dim=1))




