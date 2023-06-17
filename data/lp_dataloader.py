from torch_geometric.data import InMemoryDataset, Data
import scipy.io as sio
import pickle
import networkx as nx
from data.utils import *
import random
from torch_geometric.utils import negative_sampling

class Amazon1DomainData(InMemoryDataset):
    """
    图数据的dataloader
    Non-IID论文提供的amazon review的子集
    """
    def __init__(self,root,name,transform=None,pre_transform=None,pre_filter=None):
        self.root = root
        self.name=name
        super(Amazon1DomainData, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self):
        """
        注意这是已经处理好的数据
        :return:
        """
        return [f'{self.name}_train.mat',f"{self.name}_test.pkl"]
    @property
    def processed_file_names(self):
            return ['data.pt']
    def process(self):
        with open(self.root+'/raw/'+self.raw_file_names[1], 'rb') as f1:
            test_data = pickle.load(f1)
        test_dict, test_input_dict, num_user, num_item = test_data["test_dict"], test_data["test_input_dict"], test_data["num_user"], test_data["num_item"]
        train_ds = sio.loadmat(self.root+'/raw/'+self.raw_file_names[0])
        # data是个元素是三元组的列表，每个元素[u,i,0/1]表示是正样本还是负样本
        train_data, adj, feats = train_ds["data"], train_ds["adj"], train_ds["feats"]
        x = torch.FloatTensor(feats)
        edge_index = sparse_mx_to_pyg_coo_tensor(adj)
        train_data = torch.LongTensor(train_data)
        data_list = []
        graph = Data(x=x,edge_index=edge_index,data=train_data, num_user=num_user,num_item=num_item,test_dict=test_dict,test_input_dict=test_input_dict)
        data_list.append(graph)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class CitationDomainData(InMemoryDataset):
    def __init__(self,root,name,transform=None,pre_transform=None,pre_filter=None):
        self.root = root
        self.name=name
        super(CitationDomainData, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self):
        """
        注意这是已经处理好的数据,不是原始数据
        :return:
        """
        return [f'{self.name}_train.mat',f"{self.name}_test.pkl"]
    @property
    def processed_file_names(self):
            return ['data.pt']

    def process(self):
        with open(self.root+'/raw/'+self.raw_file_names[1], 'rb') as f1:
            test_data = pickle.load(f1)
        test_dict, test_input_dict, num_node = test_data["test_dict"], test_data["test_input_dict"], test_data["num_node"]
        train_ds = sio.loadmat(self.root+'/raw/'+self.raw_file_names[0])
        # data是个元素是三元组的列表，每个元素[u,i,0/1]表示是正样本还是负样本
        train_data, adj, feats = train_ds["data"], train_ds["adj"], train_ds["feats"]
        x = torch.FloatTensor(feats)
        edge_index = sparse_mx_to_pyg_coo_tensor(adj)
        train_data = torch.LongTensor(train_data)
        data_list = []
        graph = Data(x=x,edge_index=edge_index,data=train_data, num_node=num_node,test_dict=test_dict,test_input_dict=test_input_dict)
        data_list.append(graph)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

from torch_geometric.utils import negative_sampling

class PPIDomainData(InMemoryDataset):
    def __init__(self,root,name,transform=None,pre_transform=None,pre_filter=None):
        self.root = root
        self.name=name
        self.species_idx_dict = {'human':'9606', 'yeast':'4932', 'mouse':'10090', 'fruit_fly':'7227', 'zebrafish':'7955', 'nematode':'6239'}
        super(PPIDomainData, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        """
        注意这是PPI的原始数据
        :return:
        """
        return [self.species_idx_dict[self.name]+'.protein.sequences.v11.5.fa',self.species_idx_dict[self.name]+'.protein.links.full.v11.5.txt']
    @property
    def processed_file_names(self):
            return ['data.pt']


    def process(self,  train_ratio=0.5):  # species in {human, yeast, mouse, fruit_fly, zebrafish, nematode}
        species_idx_dict = {'human': '9606', 'yeast': '4932', 'mouse': '10090', 'fruit_fly': '7227', 'zebrafish': '7955',
                            'nematode': '6239'}

        with open(self.raw_file_names[0], 'r') as f:
            data = f.read().split('\n')[:-1]

        # protein sequence mapping
        prot_seq_dict = {}
        prot_id, seq = '', ''
        for d in data:
            if '>' in d:
                if not prot_id == '':
                    seq.upper()
                    prot_seq_dict[prot_id] = seq
                prot_id = d.split('.')[1]
                seq = ''
            else:
                seq += d
        # get node, edge
        # edge_attr_list
        # label_list  #edge*2的标签矩阵，确实都是0 1， 但是边可能是多label，loss还是BCE，评估时要注意
        node_list, edge_list, edge_attr_list, label_list = [], [], [], []
        label_mask = []
        with open(self.raw_file_names[1], 'r') as f:
            data = f.read().split('\n')[1:-1]
        for d in data:
            d = d.split()
            # neighbor / fusion / co-occurence
            edge_attr = np.array([float(d[2]), float(d[4]), float(d[5])], dtype=np.float32) / 1000
            # edge_attr[edge_attr<0.4] = 0， 边有两个标签
            label = np.zeros(2, dtype=np.float32)
            # 7 co-expression
            if int(d[7]) >= 700: label[0] = 1
            # 9 experimental
            if int(d[9]) >= 700: label[1] = 1
            # 跳过对实验没有物理意义的边
            if ((edge_attr >= 0.4).sum() == 0) and (label.sum() == 0): continue

            prot1, prot2 = d[0].split('.')[1], d[1].split('.')[1]
            node_list += [prot1, prot2]
            edge_list.append([prot1, prot2])
            edge_attr_list.append(edge_attr)
            label_list.append(label)
            # label mask？为什么加mask
            label_mask.append(1) if label.sum() > 0 else label_mask.append(0)
        # 表示边的mask ['val','val',...,.,'train']
        label_train_val = ['val' if n < len(label_mask) * (1 - train_ratio) else 'train' for n in range(len(label_mask))]
        # 随机性必须有，后续label_train_val2才能保证训练集标签不被刷掉
        random.shuffle(label_train_val)
        # label_train_val_2 把一些放在【0,0】上的val train换成了Unk
        label_train_val_2, count = [], 0
        for l in label_mask:
            if l == 0: label_train_val_2.append('unknown')
            # 为什么要这么做？是不是进一步减少了训练集数量？
            if l == 1: label_train_val_2.append(label_train_val[count]); count += 1
        print(label_train_val, label_train_val_2)
        # build graph
        node_list = list(set(node_list))  # drop duplicate
        G = nx.Graph()
        # node - idx映射。很关键
        prot_idx_dict = {node: n for n, node in enumerate(node_list)}
        for node in node_list:
            G.add_node(prot_idx_dict[node], protein_id=node, sequence=prot_seq_dict[node])  # node有两个属性：protein-id和sequence
        # add edge  train_val属性包括Unk, val, train三类
        for n, (prot1, prot2) in enumerate(edge_list):
            G.add_edge(prot_idx_dict[prot1], prot_idx_dict[prot2], edge_attr=edge_attr_list[n], label=label_list[n],
                       train_val=label_train_val_2[n])

        # build input for GDA models
        # train data
        # x 必须是列表，因为是字符串类型的
        self.x = [n[1]['sequence'] for n in G.nodes(data=True)]  # feats: sequence
        adj = torch.tensor([[n1, n2] for n1, n2 in G.edges if G.edges[n1, n2]['train_val'] == 'train'],
                           dtype=torch.int64).t()
        adj_attr = torch.cat(
            [torch.from_numpy(e[2]['edge_attr']).view(1, -1) for e in G.edges(data=True) if e[2]['train_val'] == 'train'],
            dim=0)
        # train data没有GRADE提供的负采样，不过也不需要，相当于是连边分类任务
        train_data = [[x[0], x[1], x[2]['label'].tolist()] for x in G.edges(data=True) if x[2]['train_val'] == 'train']

        # test data
        # 至少有一个标签的边
        edge_index_label = torch.tensor([[n1, n2] for n1, n2 in G.edges if not G.edges[n1, n2]['train_val'] == 'unknown'],
                                        dtype=torch.int64).t()
        # val边
        edge_index_label_val = torch.tensor([[n1, n2] for n1, n2 in G.edges if G.edges[n1, n2]['train_val'] == 'val'],
                                            dtype=torch.int64).t()
        # 对至少有一个标签的边负采样， 19倍负采样，val的时候比例是5%，切合论文
        edge_index_label_neg = negative_sampling(edge_index=edge_index_label, num_nodes=len(G),
                                                             num_neg_samples=edge_index_label_val.shape[
                                                                                 1] * 19).t().numpy().tolist()
        for n1, n2 in edge_index_label_neg:
            if (n1, n2) in G.edges():
                G.edges[n1, n2]['train_val'] = 'val'
            else:
                G.add_edge(n1, n2, edge_attr=np.zeros(3, dtype=np.float32), label=np.zeros(2, dtype=np.float32),
                           train_val='val')
        test_adj = torch.tensor([[n1, n2] for n1, n2 in G.edges if G.edges[n1, n2]['train_val'] == 'val'],
                                dtype=torch.int64).t()
        test_adj_attr = torch.cat(
            [torch.from_numpy(e[2]['edge_attr']).view(1, -1) for e in G.edges(data=True) if e[2]['train_val'] == 'val'],
            dim=0)
        test_adj_label = torch.cat(
            [torch.from_numpy(x[2]['label']).view(1, -1) for x in G.edges(data=True) if x[2]['train_val'] == 'val'], dim=0)

        data_list = []
        graph = Data(edge_index=adj,data=train_data, edge_attr=adj_attr, num_node=len(G.nodes),test_edge_index=test_adj, test_edge_attr = test_adj_attr, test_edge_label=test_adj_label)
        data_list.append(graph)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

        # print(self.name, 'node number', len(G.nodes()), 'edge number', len(G.edges()), 'coexpression',
        #       int(label_list[:, 0].sum()), 'experiments', int(label_list[:, 1].sum()))

def dataloader_lp(model_name, dataset_name, collection_name, device="cpu"):

    # register dataset name
    if 'amazon1' in dataset_name:
        dl = Amazon1DomainData
    elif 'citation' in dataset_name:
        dl = CitationDomainData
    elif 'ppi' in dataset_name:
        dl = PPIDomainData
    dataset = dl(root=f"../../data/lp/{dataset_name}/{collection_name}", name=collection_name)
    if model_name=='pyg':
        return dataset


