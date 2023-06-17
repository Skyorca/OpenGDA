import os
import pandas as pd
import random


def load_file(file_name):
    """
    data: [[v1,v2],[v1,v3],...]边表
    actual_dict: {v1:[v2,v3]} keys是用户values是商品列表
    :param file_name:
    :return:
    """
    df = pd.read_csv(file_name)
    actual_dict = {}
    # 同一个user购买的多个商品，按照user聚合起来
    for user, sf in df.groupby("users"):
        actual_dict[user] = list(sf["items"])

    data = df[["users", "items"]].to_numpy(dtype=int).tolist()
    return data, actual_dict, set(df["users"]), set(df["items"])


class Dataset(object):
    def __init__(self, path, name, target=True, print_summary=False):
        if target:
            self.train_path = os.path.join(path, "{}_test.csv".format(name))
            self.test_path = os.path.join(path, "{}_train.csv".format(name))
        else:
            self.train_path = os.path.join(path, "{}_train.csv".format(name))
            self.test_path = os.path.join(path, "{}_test.csv".format(name))
        self.print_summary = print_summary
        self.initialize()

    def initialize(self):
        self.train_data, self.train_dict, train_user_set, train_item_set = load_file(self.train_path)
        self.test_data, self.test_dict, test_user_set, test_item_set = load_file(self.test_path)

        # assert (test_user_set.issubset(train_user_set))
        # assert (test_item_set.issubset(train_item_set))
        # 通过union操作取所有的train test的user/items的全集
        self.user_set = train_user_set.union(test_user_set)
        self.item_set = train_item_set.union(test_item_set)
        self.num_user = len(self.user_set)
        self.num_item = len(self.item_set)
        self.train_size = len(self.train_data)
        self.test_size = len(self.test_data)

        self.test_input_dict, self.train_neg_dict = self.get_dicts()

        if self.print_summary:
            print("Train size:", self.train_size)
            print("Test size:", self.test_size)
            print("Number of user:", self.num_user)
            print("Number of item:", self.num_item)
            print("Data Density: {:3f}%".format(100 * self.train_size / (self.num_user * self.num_item)))

    def get_dicts(self):
        """

        :return:
        """
        train_actual_dict, test_actual_dict = self.train_dict, self.test_dict
        train_neg_dict = {}   # 训练集的负采样全集
        test_input_dict = {}  # 包含原始正样本和采样的负样本的测试集
        random.seed(0)
        # 针对训练集 测试集中的所有用户构建每个用户的全部负样本集合
        for user in list(self.user_set):
            train_neg_dict[user] = list(self.item_set - set(train_actual_dict[user]))

        # 往测试集中填充负样本，测试集中每个用户有K个交互商品构成正样本对时，就填充100K个负样本
        for user in test_actual_dict.keys():
            # test_input_dict[user] = train_neg_dict[user]
            # train_neg_dict[user] = list(set(train_neg_dict[user]) - set(test_actual_dict[user]))
            # train_neg_dict[user] = list(set(train_neg_dict[user]) - set(test_actual_dict[user]))
            random.shuffle(train_neg_dict[user])
            test_input_dict[user] = list(set(train_neg_dict[user][:len(test_actual_dict[user])*100] + test_actual_dict[user]))
            random.shuffle(test_input_dict[user])
        return test_input_dict, train_neg_dict

    def get_train(self):
        print("No training data are generated.")
        return None

    def get_data(self):
        """
        test_dict是测试集上的user-itemList pair
        test_input_dict是user-itemList pair,这里的itemList除了包括本身user购买过的item还包括100个负样本。这些共同构成了
        推荐的潜在候选集。将来评估时从这里选topK然后去评价
        问题：
        :return:
        """
        return self.test_dict, self.test_input_dict, self.num_user, self.num_item
