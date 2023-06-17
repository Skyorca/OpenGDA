
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
from torch_geometric.nn import global_mean_pool
from utils import to_onehot


class GNN(torch.nn.Module):
    def __init__(self, num_gcn_layers,hidden_dims, base_model=None, type="gcn",device='cpu', **kwargs):
        super(GNN, self).__init__()
        if base_model is None:
            weights = [None]*num_gcn_layers
            biases = [None]*num_gcn_layers
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
                     cached=False,
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

class UDAGCN_GC(nn.Module):
    def __init__(self,use_UDAGCN,encoder_dim, num_gcn_layers,hidden_dims:list,num_classes, device,coeff,path_len=3):
        super(UDAGCN_GC, self).__init__()
        self.num_gcn_layers = num_gcn_layers
        self.hidden_dims = hidden_dims
        self.num_classes = num_classes
        self.device = device
        self.coeff = coeff
        assert len(self.hidden_dims) == self.num_gcn_layers+1
        self.use_UDAGCN = use_UDAGCN
        self.encoder_dim = encoder_dim
        self.encoder = GNN(type="gcn",num_gcn_layers=num_gcn_layers,hidden_dims=hidden_dims)
        # TODO 重要参数是PPMI的K
        if self.use_UDAGCN:
            self.ppmi_encoder = GNN(base_model=self.encoder, type="ppmi", path_len=path_len,num_gcn_layers=num_gcn_layers,hidden_dims=hidden_dims,device=self.device)

        self.cls_model = nn.Sequential(
            nn.Linear(self.encoder_dim, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes),
            nn.Sigmoid()
        )

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

    def forward(self, src_data, tgt_data, alpha=1.0):
        """
        这个模型几个点
        1. batch训练模式
        :param alpha:
        :return:
        """
        src_x, src_edge_index, src_batch, src_y = src_data.x,src_data.edge_index,src_data.batch,src_data.y
        tgt_x, tgt_edge_index, tgt_batch = tgt_data.x, tgt_data.edge_index,tgt_data.batch
        x_ds = self.encode(src_x, src_edge_index, "source")
        x_dt = self.encode(tgt_x, tgt_edge_index, "target")
        x_ds_batch = global_mean_pool(x_ds, src_batch)  # [batch_size, hidden_channels]
        x_dt_batch = global_mean_pool(x_dt, tgt_batch)
        x_ds_batch = F.dropout(x_ds_batch, p=0.1, training=self.training)
        x_dt_batch = F.dropout(x_dt_batch, p=0.1, training=self.training)
        # source graph classification loss
        if src_y.dim()==1:
            src_y = to_onehot(src_y,num_classes=self.num_classes,device=self.device)
        src_logits = self.cls_model(x_ds_batch)
        clf_loss = self.criterion(src_logits, src_y)
        # 整体混淆source target在batch中的节点
        source_domain_preds = self.discriminator(ReverseLayerF.apply(x_ds_batch,alpha))
        target_domain_preds = self.discriminator(ReverseLayerF.apply(x_dt_batch,alpha))
        source_domain_labels = np.array([0] * source_domain_preds.shape[0])
        source_domain_labels = torch.tensor(source_domain_labels, requires_grad=False, dtype=torch.float).to(self.device)
        target_domain_labels = np.array([1] * target_domain_preds.shape[0])
        target_domain_labels = torch.tensor(target_domain_labels, requires_grad=False, dtype=torch.float).to(self.device)
        domain_loss = self.criterion(source_domain_preds.view(-1), source_domain_labels) + self.criterion(target_domain_preds.view(-1), target_domain_labels)

        return clf_loss + domain_loss * self.coeff['domain'], clf_loss, domain_loss

    def inference(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x_d = self.encode(x, edge_index,cache_name='target')
        x_d_batch = F.dropout(global_mean_pool(x_d, batch),p=0.1, training=self.training)
        logits = self.cls_model(x_d_batch)
        return logits




