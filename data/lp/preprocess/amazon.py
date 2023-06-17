from __future__ import print_function
import argparse
import torch
import numpy as np
from amazon_reader.data_loader import data_loader
import scipy.io as sio
import os
import pickle

# Command setting
parser = argparse.ArgumentParser(description='Cross-Domain Recommendation')
parser.add_argument('--data_dir', type=str, default='data/nonoverlapping/Book_Movie', help='domain path')
parser.add_argument('--domain', type=str, default='music', help='domain')
args = parser.parse_args()


if __name__ == '__main__':
    args.device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() else "cpu")
    domain = args.domain
    dataset = data_loader(args.data_dir, args.domain, target=False, print_summary=True)
    # 存储源域测试集
    test_dict_ds, test_input_dict_ds, num_user_ds, num_item_ds = dataset.get_data()
    test_ds = {"test_dict":test_dict_ds, "test_input_dict":test_input_dict_ds, "num_user":num_user_ds, "num_item":num_item_ds}
    with open(f"{args.domain}_test.pkl",'wb') as f1:
        pickle.dump(test_ds, f1)
    print(f'saved source domain test data')
    # 存储源域训练集
    data_ds, adj_s, feats_s = dataset.get_train()
    train_ds = {"data":data_ds, "adj":adj_s,"feats":feats_s}
    sio.savemat(f"{args.s_domain}_train.mat",train_ds)
    print(f'saved source domain train data')
