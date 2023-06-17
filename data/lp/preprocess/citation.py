import numpy as np
import random
import torch
import scipy.sparse as sp
from multiprocessing import Pool
import copy
import pickle
import scipy.io as sio


# 保持编号一致
def load_edge(file_name, train_ratio=0.4):
    train_edge_dict = {}
    train_data  = []
    test_edge_dict = {}
    test_data = []
    with open(file_name,'r') as f:
        cont = f.read().splitlines()
        for edge in cont:
            src, dst = int(edge.split(',')[0]), int(edge.split(',')[1])
            if np.random.randint(0,100)<train_ratio*100:
                if src not in train_edge_dict:
                    train_edge_dict[src] = [dst]
                else:
                    train_edge_dict[src].append(dst)
                train_data.append([src, dst])
            else:
                if src not in test_edge_dict:
                    test_edge_dict[src] = [dst]
                else:
                    test_edge_dict[src].append(dst)
                test_data.append([src, dst])
    return train_edge_dict, train_data, test_edge_dict, test_data


def generate_neg(train_dict, test_dict, node_set):
    """
    会有测试集的节点不在训练集中的情况
    :param train_dict:
    :param test_dict:
    :param node_set:
    :return:
    """
    train_actual_dict, test_actual_dict = train_dict, test_dict
    train_neg_dict = {}  # 训练集的负采样全集
    test_input_dict = {}  # 包含原始正样本和采样的负样本的测试集
    # 针对训练集 测试集中的所有用户构建每个用户的全部负样本集合
    for n in train_actual_dict.keys():
        train_neg_dict[n] = list(node_set - set(train_actual_dict[n]))
    # 因为可能会出现测试集中的节点不在训练集中的情况，所以需要对测试集单独构建负样本词典
    test_neg_dict = {}
    for n in test_actual_dict.keys():
        if n not in train_actual_dict.keys():
            test_neg_dict[n] = list(node_set - set(test_actual_dict[n]))
        else:
            test_neg_dict[n] = list(node_set - set(test_actual_dict[n]) - set(train_actual_dict[n]))

    # 往测试集中填充负样本,
    for n in test_actual_dict.keys():
        random.shuffle(test_neg_dict[n])
        test_input_dict[n] = list(
            set(test_neg_dict[n][:len(test_actual_dict[n]) * 100] + test_actual_dict[n]))
        random.shuffle(test_input_dict[n])
    return test_input_dict, train_neg_dict

def load_feat(docs_path):
    f = open(docs_path, 'rb')
    content_list = []
    for line in f.readlines():
        line = str(line, encoding="utf-8")
        content_list.append(line.split(","))
    x = np.array(content_list, dtype=float)  # 这里要注意返回的一定不能是tensor而是array
    return x, set(range(x.shape[0]))

def get_train(train_neg_dict, train_data, num_nodes):
    processor_num = 4
    # 通过多线程，构建好正样本与负样本集合
    try:
        with Pool(processor_num) as pool:
            nargs = [(src, dst, train_neg_dict[src]) for src, dst in train_data]
            res_list = pool.map(_add_negtive, nargs)
    except:
        pool.close()
    # res_list = [[(u1,v1,1),(u1,v2,0),...],[1POS+4NEG],[]]表示负采样的结果，负采样率为4
    out = []
    # batch_n是一个五元素的列表
    # out把res_list中每个五元组的界限打破，形成一个大列表
    for batch_n in res_list:
        out += batch_n

    adj= construct_g(num_nodes, train_data)
    return out, adj

def construct_g(num_nodes, train_data):
    # 构建二部图需要训练集和测试集中的全部用户和商品列表
    edges = np.array(copy.deepcopy(train_data))
    print('edges shape', edges.shape)
    # 因为原始的用户和商品的下标都是0开始的，为了区分，把商品的下标提高到num_user开始
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(num_nodes, num_nodes),
                        dtype=np.float32)
    # 邻接矩阵对称化
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # 因为做了归一化，所以有边权
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj

def normalize_adj(adj):
    """如果是正常的gcn的输入，不需要处理，只需要对称化就行"""
    adj = sp.coo_matrix(adj)
    return adj

def _add_negtive(args):
    src, dst, neg_dict = args
    # 1 表示正样本 0 表示负样本
    neg_pair = [[src, dst, 1]]

    neg_num = 4
    neg_sample_list = np.random.choice(neg_dict, neg_num, replace=False).tolist()
    for neg_sample in neg_sample_list:
        neg_pair.append([src, neg_sample, 0])

    return neg_pair

if __name__ == '__main__':
    domain = 'acm'
    train_edge_dict, train_data, test_edge_dict, test_data = load_edge(f'data/{domain}/raw/{domain}_edgelist.txt')
    print(len(train_data), len(test_data))
    feat_s, node_set = load_feat(f'data/{domain}/raw/{domain}_docs.txt')
    test_input_dict, train_neg_dict = generate_neg(train_edge_dict,test_edge_dict,node_set)
    data, adj = get_train(train_neg_dict, train_data,len(node_set))
    test = {"test_dict": test_edge_dict, "test_input_dict": test_input_dict, "num_node": feat_s.shape[0]}
    with open(f"{domain}_test.pkl", 'wb') as f1:
        pickle.dump(test, f1)
    print(f'saved {domain} test data')
    # 存储源域训练集
    train_ds = {"data": data, "adj": adj, "feats": feat_s}
    sio.savemat(f"{domain}_train.mat", train_ds)
    print(f'saved {domain} train data')

