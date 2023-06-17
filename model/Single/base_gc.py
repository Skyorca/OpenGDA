import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool


class GCN(nn.Module):
    def __init__(self, input_feat, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_feat, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)


    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        x = F.dropout(x, p=0.6, training=self.training)
        return x


class BASE_gc(nn.Module):
    def __init__(self, input_feat, hidden_channels, num_classes):
        super(BASE_gc, self).__init__()
        self.gnn = GCN(input_feat,hidden_channels)
        self.lin = nn.Sequential(Linear(hidden_channels, 16),nn.ReLU(),Linear(16, num_classes))
    def forward(self, src_x, src_edge_index, src_batch):
        src_emb = self.gnn(src_x, src_edge_index, src_batch)
        src_preds = self.lin(src_emb)
        return src_preds