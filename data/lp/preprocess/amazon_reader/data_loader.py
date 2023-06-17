from reader.base import Dataset
import numpy as np
import pandas as pd
import scipy.sparse as sp
from multiprocessing import Pool
import copy


class data_loader(Dataset):
    """
    加载已经处理好的user-item交互csv
    """
    def __init__(self, path, name, target, print_summary=False):
        super(data_loader, self).__init__(path, name, target, print_summary)

    def get_train(self, neg_num=1):
        processor_num = 4
        # 通过多线程，构建好正样本与负样本集合
        try:
            with Pool(processor_num) as pool:
                nargs = [(user, item, self.train_neg_dict[user]) for user, item in self.train_data]
                res_list = pool.map(_add_negtive, nargs)
        except:
            pool.close()
        # res_list = [[(u1,v1,1),(u1,v2,0),...],[1POS+4NEG],[]]表示负采样的结果，负采样率为4
        out = []
        # batch_n是一个五元素的列表
        # out把res_list中每个五元组的界限打破，形成一个大列表
        for batch_n in res_list:
            out += batch_n

        adj, feats = self.construct_g()
        return out, adj, feats

    def construct_g(self):
        # 构建二部图需要训练集和测试集中的全部用户和商品列表
        num_nodes = self.num_user + self.num_item
        edges = np.array(copy.deepcopy(self.train_data))
        # 因为原始的用户和商品的下标都是0开始的，为了区分，把商品的下标提高到num_user开始
        edges[:, 1] += self.num_user
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(num_nodes, num_nodes),
                            dtype=np.float32)
        # 邻接矩阵对称化
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        # 因为做了归一化，所以有边权
        adj, feats = self.normalize_adj(adj + sp.eye(adj.shape[0]))
        return adj, feats

    def normalize_adj(self, adj):
        """如果是正常的gcn的输入，不需要处理，只需要对称化就行"""
        adj = sp.coo_matrix(adj)
        rowsum = np.array(adj.sum(1))
        feats = self.one_hot_embedding((rowsum-1).astype(int).flatten())
        return adj, feats.astype(np.float)
        # d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        # d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        # d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        # return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo(), feats.astype(np.float)

    def one_hot_embedding(self, degree, num_classes=5000):
        """只对用户节点用度的one-hot做特征"""
        y = np.eye(num_classes)
        return y[degree]


# @staticmethod
def _add_negtive(args):
    user, item, neg_dict = args
    # 1 表示正样本 0 表示负样本
    neg_pair = [[user, item, 1]]

    neg_num = 4
    neg_sample_list = np.random.choice(neg_dict, neg_num, replace=False).tolist()
    for neg_sample in neg_sample_list:
        neg_pair.append([user, neg_sample, 0])

    return neg_pair
