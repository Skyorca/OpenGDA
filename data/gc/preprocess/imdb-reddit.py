from torch_geometric.loader import DataLoader
import torch
from torch_geometric.datasets import TUDataset
import numpy as np
import networkx as nx
import torch_geometric.transforms as T

# TUDataset都是双向边
ds = TUDataset(root='data/TUDataset', name='IMDB-BINARY')
loader = DataLoader(ds, batch_size=1, shuffle=True)

max_dgr = 136
all_feats = []
for idx, d in enumerate(loader):
    edge_index = d.edge_index.numpy()
    a = torch.cat([d.edge_index[0,:], d.edge_index[1,:]])
    a = a.numpy().tolist()
    a = set(a)
    if len(a) != d.num_nodes:
        isolated_nodeids = []
        for n in range(d.num_nodes):
            if n not in a: isolated_nodeids.append(n)
        # 给孤立节点加自环
        self_loops = np.array([[n,n] for n in isolated_nodeids]).reshape(2,-1)
        edge_index = np.hstack([edge_index, self_loops])
    edge_index = edge_index.T.tolist()
    G = nx.from_edgelist(edge_index)
    G.remove_edges_from(nx.selfloop_edges(G))
    dgr = {x[0]: x[1] for x in nx.degree(G)}
    dgr = sorted(dgr.items(), key=lambda x:x[0])
    x_ = np.zeros((d.num_nodes, max_dgr))
    for (nodeid, val) in dgr:
        x_[nodeid][min(val,max_dgr-1)] = 1
    all_feats.append(x_)
x = np.vstack(all_feats)
print(x, x.shape)
x = torch.FloatTensor(x)
torch.save(x, 'REDDIT-BINARY_node_attributes.pt')
