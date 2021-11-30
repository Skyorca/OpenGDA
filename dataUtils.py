import scipy.io as sio
from sklearn.decomposition import PCA
from torch_geometric.data import InMemoryDataset, Data
import torch
import torch.nn as nn
import scipy.sparse as sp
import numpy as np

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
        indices = np.vstack([adj.row, adj.col])
        edge_index = torch.tensor(indices,dtype=torch.long)
        # label is float: to support BCEWithLogits loss
        y = torch.from_numpy(np.array(labels)).to(torch.float)
        data_list = []
        graph = Data(x=features,edge_index=edge_index,y=y)
        # train-val-test split
        random_node_indices = np.random.permutation(y.shape[0])
        train_size = int(len(random_node_indices) * 0.7)
        val_size = int(len(random_node_indices) * 0.1)
        train_node_indices = random_node_indices[:train_size]
        val_node_indices = random_node_indices[train_size:train_size+val_size]
        test_node_indices = random_node_indices[train_size+val_size:]
        train_mask = torch.zeros([y.shape[0]], dtype=torch.uint8)
        train_mask[train_node_indices] = 1
        val_mask = torch.zeros([y.shape[0]], dtype=torch.uint8)
        val_mask[val_node_indices] = 1
        test_mask = torch.zeros([y.shape[0]], dtype=torch.uint8)
        test_mask[test_node_indices] = 1
        graph.train_mask = train_mask
        graph.val_mask = val_mask
        graph.test_mask = test_mask
        if self.pre_transform is not None:
            graph = self.pre_transform(graph)
        data_list.append(graph)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])