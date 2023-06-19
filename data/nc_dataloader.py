import torch
import numpy as np
import scipy.sparse as sp
import scipy.io as sio
from sklearn.decomposition import PCA
from torch_geometric.data import InMemoryDataset, Data
import pandas as pd
import json
import networkx as nx
from data.utils import *
import os
import sys

class CitationDomainData(InMemoryDataset):
    """
    图数据的dataloader
    引文数据集加载器：acmv9,citationv1,dblpv7  these datasets are originally from CDNE
    """
    def __init__(self,root,name,use_pca=False,pca_dim=1000,transform=None,pre_transform=None,pre_filter=None):
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

class BlogDomainData(InMemoryDataset):
    """
    图数据的dataloader
    博客数据集加载器：blog1 blog2  these datasets are originally from CDNE
    """
    def __init__(self,root,name,use_pca=False,pca_dim=1000,transform=None,pre_transform=None,pre_filter=None):
        self.name=name
        self.use_pca = use_pca
        self.dim = pca_dim
        super(BlogDomainData, self).__init__(root, transform, pre_transform, pre_filter)
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
            features = features.todense().astype(float)
            features = torch.from_numpy(features).to(torch.float)
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

class TwitchDomainData(InMemoryDataset):
    def __init__(self,root,name,transform=None,pre_transform=None,pre_filter=None):
        self.name=name
        self.root = root
        super(TwitchDomainData, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f"musae_{self.name}_features.json", f"musae_{self.name}_edges.csv",f"musae_{self.name}_target.csv"]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def _load_graph(self,graph_path):
        """
        Reading a NetworkX graph.
        :param graph_path: Path to the edge list.
        :return graph: NetworkX object.
        """
        data = pd.read_csv(graph_path)
        edges = data.values.tolist()
        edges = [[int(edge[0]), int(edge[1])] for edge in edges]
        graph = nx.from_edgelist(edges)
        graph.remove_edges_from(nx.selfloop_edges(graph))
        # 根据节点在g.nodes中出现的先后次序把节点重新编号
        node_id_map = {}
        nodes = list(graph.nodes)
        for i in range(len(nodes)):
            node_id_map[nodes[i]] = i
        # to_scipy_sparse_matrix会把节点的id根据其出现的先后顺序，即g.nodes列表，替换掉
        # 如果原节点i在g.nodes排序在第j个，则其出现在邻接矩阵中的id是j
        # 注意导出时会导出undirected graph，就是一条边会被算2次，所以边数double
        adj = nx.to_scipy_sparse_matrix(graph,format="coo")
        return adj,graph.number_of_nodes(), graph.number_of_edges(), node_id_map

    def _load_features(self, features_path, node_id_map):
        """
        Reading the features from disk.
        :param features_path: Location of feature JSON.
        :return features: Feature hash table.
        """
        features = json.load(open(features_path))
        features = {node_id_map[int(k)]: [int(val) for val in v] for k, v in features.items()}
        return features

    def _load_labels(self,target_path,node_id_map):
        data = pd.read_csv(target_path)[["mature","new_id"]]
        data.sort_values(by=['new_id'],inplace=True)
        data = data[~data.duplicated(subset="new_id")]  # 去掉重复的行
        pos_idx_ = data[data['mature']==True]['new_id'].tolist()
        neg_idx_ = data[data['mature']==False]['new_id'].tolist()
        pos_idx = [node_id_map[n] for n in pos_idx_]
        neg_idx = [node_id_map[n] for n in neg_idx_]
        y = np.zeros((len(pos_idx)+len(neg_idx),2))
        y[pos_idx] = np.array([0,1])
        y[neg_idx] = np.array([1,0])
        return torch.FloatTensor(y)

    def _create_onehot_embedding(self, features, n_nodes):
        x = np.zeros((n_nodes,3170))
        for row_idx, col_idxes in features.items():
            x[row_idx, col_idxes] = np.ones_like(col_idxes)
        return torch.FloatTensor(x)

    def process(self):
        adj,n_nodes,n_edges, node_id_map = self._load_graph(self.root+"\\"+f"raw/musae_{self.name}_edges.csv")
        # edge_index应该是COO格式的long型tensor
        indices = np.vstack([adj.row, adj.col])
        edge_index = torch.tensor(indices,dtype=torch.long)
        # 计算ppmi矩阵（虽然以稀疏矩阵格式，但大量元素都有值，因为是概率矩阵）
        print('computing PPMI')
        A_k = AggTranProbMat(adj, 3)
        PPMI_ = ComputePPMI(A_k)
        n_PPMI_ = MyScaleSimMat(PPMI_)  # row normalized PPMI
        n_PPMI_mx = sp.coo_matrix(n_PPMI_)
        ppmi_edge_index = torch.tensor(np.vstack([n_PPMI_mx.row, n_PPMI_mx.col]),dtype=torch.long)
        ppmi_edge_attr = torch.tensor(n_PPMI_mx.data, dtype=torch.float32)

        raw_features = self._load_features(self.root+"\\"+f"raw/musae_{self.name}_features.json", node_id_map)
        features = self._create_onehot_embedding(raw_features, n_nodes)
        labels = self._load_labels(self.root+"\\"+f"raw/musae_{self.name}_target.csv", node_id_map)
        data_list = []
        graph = Data(x=features,edge_index=edge_index,ppmi_edge_index=ppmi_edge_index, ppmi_edge_attr=ppmi_edge_attr,y=labels)
        data_list.append(graph)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class AirPortDomainData(InMemoryDataset):
    def __init__(self,root,name,transform=None,pre_transform=None,pre_filter=None):
        self.name=name
        self.root = root
        self.kmax = 1
        self.feat_dim = 8
        self.num_class = 4
        super(AirPortDomainData, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self):
        return [f"{self.name}-airports.edgelist", f"labels-{self.name}-airports.txt"]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def generate_core_view(self, path, kmax=1):
        """
        :return: k-core-views of original graph
        """
        g = nx.read_edgelist(path, nodetype=int, create_using=nx.DiGraph())
        mapping = {}
        r_mapping = {}
        nodes = list(g.nodes)
        n_node = g.number_of_nodes()
        for idx in range(len(nodes)):
            mapping[nodes[idx]] = idx
            r_mapping[idx] = nodes[idx]
        g = nx.relabel_nodes(g, mapping=mapping)
        g.remove_edges_from(nx.selfloop_edges(g))
        views = []
        sparse_adj = []
        for i in range(1, kmax + 1):
            core = nx.k_core(g, k=i)
            # NOTE : nx to scipy sparse matrix要求node ordering，比较麻烦，不如直接拿节点映射后的ID
            row = np.array([x[0] for x in core.edges])
            col = np.array([x[1] for x in core.edges])
            # NOTE 为什么必须要输入双向边才有好的效果, 而且与sparse matrix结果不一致
            indices = np.hstack([np.vstack([row, col]), np.vstack([col, row])])
            self_loops = np.vstack([np.array(core.nodes), np.array(core.nodes)])
            indices = np.hstack([indices, self_loops])
            # print(indices.shape)
            edge_index = torch.tensor(indices, dtype=torch.long)
            views.append(edge_index)
            sp_adj = sp.coo_matrix((np.ones_like(row),(row,col)),shape=(n_node, n_node))
            sparse_adj.append(sp_adj)
        # note 只取kcore=1即原图
        edge_index = views[0]
        sp_adj = sparse_adj[0]
        return g, edge_index, r_mapping, sp_adj

    def generate_feture(self, graph, max_degree):
        features = torch.zeros([graph.number_of_nodes(), max_degree])
        # nodeID是0开始的数组编号，因此可以这么写
        for i in range(graph.number_of_nodes()):
            try:
                features[i][min(graph.degree[i], max_degree - 1)] = 1
            except:
                features[i][0] = 1
        return features

    def generate_label(self,path, r_mapping):
        labels = dict()
        with open(path) as IN:
            IN.readline()
            for line in IN:
                tmp = line.strip().split(' ')
                labels[int(tmp[0])] = int(tmp[1])
        y = []
        for idx, nodeid in r_mapping.items():
            y.append(labels[nodeid])
        y = torch.tensor(y)
        identity = torch.eye(self.num_class)
        onehot = torch.index_select(identity, 0, y)
        return onehot

    def process(self):
        graph, edge_index, r_mapping, sp_adj = self.generate_core_view(self.root+"\\"+f'raw/{self.name}-airports.edgelist', kmax=self.kmax)
        print('computing PPMI')
        A_k = AggTranProbMat(sp_adj, 3)
        PPMI_ = ComputePPMI(A_k)
        n_PPMI_ = MyScaleSimMat(PPMI_)  # row normalized PPMI
        n_PPMI_mx = sp.coo_matrix(n_PPMI_)
        ppmi_edge_index = torch.tensor(np.vstack([n_PPMI_mx.row, n_PPMI_mx.col]),dtype=torch.long)
        ppmi_edge_attr = torch.tensor(n_PPMI_mx.data, dtype=torch.float32)
        features = self.generate_feture(graph, self.feat_dim)
        labels = self.generate_label(self.root+"\\"+f"raw/labels-{self.name}-airports.txt", r_mapping)
        data_list = []
        graph = Data(x=features,edge_index=edge_index,ppmi_edge_index=ppmi_edge_index, ppmi_edge_attr=ppmi_edge_attr,y=labels)
        data_list.append(graph)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def dataloader_nc(model_name, dataset_name, collection_name, device="cpu"):
    if dataset_name=="citation":
        dataloader = CitationDomainData
    elif dataset_name=='blog':
        dataloader = BlogDomainData
    elif dataset_name=="twitch":
        dataloader = TwitchDomainData
    elif dataset_name=='airport':
        dataloader = AirPortDomainData
    else:
        raise NotImplementedError
    dataset = dataloader(root=f"../../data/nc/{dataset_name}/{collection_name}", name=collection_name)
    if model_name == 'pyg':
        return dataset