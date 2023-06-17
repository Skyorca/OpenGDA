import numpy as np
import torch
import scipy.sparse as sp
from torch_geometric.utils import negative_sampling

def coo_tensor_to_dense_matrix(edge_index, n_node):
    row = edge_index[0,:].cpu().numpy()
    col = edge_index[1,:].cpu().numpy()
    val = np.ones_like(row)
    coo_t = sp.coo_matrix((val, (row, col)), shape=(n_node, n_node))
    return coo_t.toarray()


def get_loss_inputs(adj,n_node,device):
    # 原始adj就是方阵，节点数为num_user+num_item
    # adj_label = coo_tensor_to_dense_matrix(adj, n_node)
    neg_edge_index = negative_sampling(edge_index=adj,num_nodes=n_node,force_undirected=True).to(device)
    adj_label = torch.cat((torch.ones_like(adj[0,:]).float(), torch.zeros_like(neg_edge_index[0,:]).float()))
    norm = n_node * n_node / float((n_node * n_node - torch.sum(adj_label)) * 2)
    pos_weight0 = float(n_node * n_node - torch.sum(adj_label)) / torch.sum(adj_label)
    pos_weight = torch.ones_like(adj[0,:]).float()*pos_weight0
    return neg_edge_index, adj_label.to(device), norm, pos_weight.to(device)

def to_onehot(label_matrix, num_classes, device):
    identity = torch.eye(num_classes).to(device)
    onehot = torch.index_select(identity, 0, label_matrix)
    return onehot