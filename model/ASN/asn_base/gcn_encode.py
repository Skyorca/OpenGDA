import sys
sys.path.append('../..')
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from base_model.cached_gcn_conv import CachedGCNConv
from base_model.ppmi_conv import PPMIConv
from torch_geometric.nn import GCNConv

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,device,base_model=None, type="gcn",cached=1):
        super(GCN, self).__init__()
        self.cached = cached
        if base_model is None:
            weights = [None, None,None]
            biases = [None, None, None]
        else:
            # 重要：这里PPMIGCN和普通的GCN是共享参数的
            weights = [conv_layer.weight for conv_layer in base_model.conv_layers]
            biases = [conv_layer.bias for conv_layer in base_model.conv_layers]
        if type=="ppmi":
            layer = PPMIConv
        elif cached:
            layer=CachedGCNConv
        else:
            layer=GCNConv
        if cached:
            self.gc1 = layer(nfeat, nhid,weight=weights[0],bias=biases[0],device=device)
            self.gc2 = layer(nhid, nclass,weight=weights[1],bias=biases[1],device=device)
            self.gc3 = layer(nhid, nclass,weight=weights[2],bias=biases[2],device=device)
        else:
            self.gc1 = layer(nfeat, nhid)
            self.gc2 = layer(nhid, nclass)
            self.gc3 = layer(nhid, nclass)

        self.conv_layers = [self.gc1, self.gc2, self.gc3]
        self.dropout = dropout

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj,cache_name):
        if self.cached:
            x = self.gc1(x, adj,cache_name)
        else:
            x = self.gc1(x, adj)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        if self.cached:
            x1 = self.gc2(x, adj,cache_name)
            y1 = self.gc3(x, adj,cache_name)
        else:
            x1 = self.gc2(x, adj)
            y1 = self.gc3(x, adj)
        # return F.dropout(F.relu(x), self.dropout, training=self.training),F.log_softmax(x, dim=1)
        z = self.reparameterize(x1,y1)
        return z,x1,y1

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

class GCNModelVAE(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,device, base_model=None, type="gcn",cached=1):
        super(GCNModelVAE, self).__init__()
        self.cached = cached
        if base_model is None:
            weights = [None, None,None]
            biases = [None, None, None]
        else:
            # 重要：这里PPMIGCN和普通的GCN是共享参数的
            weights = [conv_layer.weight for conv_layer in base_model.conv_layers]
            biases = [conv_layer.bias for conv_layer in base_model.conv_layers]
        if type=="ppmi":
            layer = PPMIConv
        elif cached:
            layer=CachedGCNConv
        else:
            layer=GCNConv
        if cached:
            self.gc1 = layer(nfeat, nhid,weight=weights[0],bias=biases[0],device=device)
            self.gc2 = layer(nhid, nclass,weight=weights[1],bias=biases[1],device=device)
            self.gc3 = layer(nhid, nclass,weight=weights[2],bias=biases[2],device=device)
        else:
            self.gc1 = layer(nfeat, nhid)
            self.gc2 = layer(nhid, nclass)
            self.gc3 = layer(nhid, nclass)
        self.conv_layers = [self.gc1, self.gc2, self.gc3]

    def encode(self, x, adj,cache_name):
        if self.cached:
            hidden1 = self.gc1(x, adj,cache_name)
            return self.gc2(hidden1, adj,cache_name), self.gc3(hidden1, adj,cache_name)
        else:
            hidden1 = self.gc1(x, adj)
            return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj,cache_name):
        mu, logvar = self.encode(x, adj,cache_name)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z, type='normal',**kwargs):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        if type=='bi':
            # 对角方块置0，二部图
            num_user = kwargs['num_user']
            adj[:num_user,:num_user] = 0.
            adj[num_user:,num_user:] = 0.
        return adj
