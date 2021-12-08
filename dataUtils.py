import scipy.io as sio
from sklearn.decomposition import PCA
from torch_geometric.data import InMemoryDataset, Data
import torch
import torch.nn as nn
import scipy.sparse as sp
import numpy as np
import networkx as nx
import random

class CitationDomainData(InMemoryDataset):
    """
    引文数据集加载器：acmv9,citationv1,dblpv7 originally from CDNE
    """
    def __init__(self,root,name,use_pca=True,pca_dim=1000,transform=None,pre_transform=None,pre_filter=None):
        self.name=name
        self.use_pca = use_pca
        self.dim = pca_dim
        super(CitationDomainData, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self):
        return [f'{self.name}.mat']
    @property
    def processed_file_names(self):
        if self.use_pca:
            return [f'data_pca_{self.dim}.pt']
        else:
            return ['data.pt']

    def feature_compression(self,features):
        """Preprcessing of features"""
        features = features.toarray()
        feat = sp.lil_matrix(PCA(n_components=self.dim, random_state=0).fit_transform(features))
        return feat.toarray()

    def process(self):
        net = sio.loadmat(self.raw_dir+"\\"+self.name+".mat")
        features, adj, labels = net['attrb'], net['network'], net['group']
        if not isinstance(features, sp.lil_matrix):
            features = sp.lil_matrix(features)
        # citation networks do not use PCA, but blog networks use
        if self.use_pca:
            features = self.feature_compression(features)
            features = torch.from_numpy(features).to(torch.float)
        else:
            features = torch.from_numpy(features.todense()).to(torch.float)
        if not isinstance(adj,sp.coo_matrix):
            adj = sp.coo_matrix(adj)
        # edge_index应该是COO格式的long型tensor
        indices = np.vstack([adj.row, adj.col])
        edge_index = torch.tensor(indices,dtype=torch.long)
        # 计算ppmi矩阵（虽然以稀疏矩阵格式，但大量元素都有值，因为是概率矩阵）
        A_k = AggTranProbMat(adj, 3)
        PPMI_ = ComputePPMI(A_k)
        n_PPMI_ = MyScaleSimMat(PPMI_)  # row normalized PPMI
        n_PPMI_mx = sp.coo_matrix(n_PPMI_)
        ppmi_edge_index = torch.tensor(np.vstack([n_PPMI_mx.row, n_PPMI_mx.col]),dtype=torch.long)
        ppmi_edge_attr = torch.tensor(n_PPMI_mx.data, dtype=torch.float32)
        # label is float: to support BCEWithLogits loss
        y = torch.from_numpy(np.array(labels)).to(torch.float)
        data_list = []
        graph = Data(x=features, edge_index=edge_index, ppmi_edge_index=ppmi_edge_index, ppmi_edge_attr=ppmi_edge_attr,y=y)
        # train-val-test split
        random_node_indices = np.random.permutation(y.shape[0])
        train_size = int(len(random_node_indices) * 0.7)
        val_size = int(len(random_node_indices) * 0.1)
        train_node_indices = random_node_indices[:train_size]
        val_node_indices = random_node_indices[train_size:train_size+val_size]
        test_node_indices = random_node_indices[train_size+val_size:]
        train_mask = torch.zeros([y.shape[0]], dtype=torch.uint8)
        train_mask[train_node_indices] = 1
        train_mask = train_mask.bool() # mask 应该是bool类型
        val_mask = torch.zeros([y.shape[0]], dtype=torch.uint8)
        val_mask[val_node_indices] = 1
        val_mask = val_mask.bool()
        test_mask = torch.zeros([y.shape[0]], dtype=torch.uint8)
        test_mask[test_node_indices] = 1
        test_mask = test_mask.bool()
        graph.train_mask = train_mask
        graph.val_mask = val_mask
        graph.test_mask = test_mask
        if self.pre_transform is not None:
            graph = self.pre_transform(graph)
        data_list.append(graph)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = np.vstack((sparse_mx.row, sparse_mx.col))
    print(indices)
    values = torch.from_numpy(sparse_mx.data)
    size = indices.shape[1]
    return torch.sparse_coo_tensor(indices, values)


def MyScaleSimMat(W):
    '''L1 row norm of a matrix'''
    rowsum = np.array(np.sum(W, axis=1), dtype=np.float32)
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    W = r_mat_inv.dot(W)
    return W

def AggTranProbMat(G, step):
    '''aggregated K-step transition probality'''
    G = MyScaleSimMat(G)
    G = sp.csc_matrix.toarray(G)
    A_k = G
    A = G
    for k in np.arange(2, step + 1):
        A_k = np.matmul(A_k, G)
        A = A + A_k / k
    return A

def ComputePPMI(A):
    '''compute PPMI, given aggregated K-step transition probality matrix as input'''
    np.fill_diagonal(A, 0)
    A = MyScaleSimMat(A)
    (p, q) = np.shape(A)
    col = np.sum(A, axis=0)
    col[col == 0] = 1
    PPMI = np.log((float(p) * A) / col[None, :])
    IdxNan = np.isnan(PPMI)
    PPMI[IdxNan] = 0
    PPMI[PPMI < 0] = 0
    return PPMI


def disturb_network(adj):
    g = nx.from_scipy_sparse_matrix(adj)
    edges_1 = []
    edges_2 = []
    edges = g.edges
    g_1 = nx.Graph()
    g_2 = nx.Graph()
    g_1.add_nodes_from(g.nodes)
    g_2.add_nodes_from(g.nodes)
    alpha_s = 0.8
    alpha_c = 0.5
    for edge in edges:
        p = np.random.uniform(0,1)
        if p>1-2*alpha_s+alpha_s*alpha_c and p<=1-alpha_s: edges_1.append(edge)
        elif p>1-alpha_s and p<=1-alpha_s*alpha_c: edges_2.append(edge)
        elif p>1-alpha_s*alpha_c:
            edges_1.append(edge)
            edges_2.append(edge)
        else: continue
    g_1.add_edges_from(edges_1)
    g_2.add_edges_from(edges_2)
    adj_1 = sp.coo_matrix(nx.to_scipy_sparse_matrix(g_1))
    adj_2 = sp.coo_matrix(nx.to_scipy_sparse_matrix(g_2))
    return adj_1, adj_2