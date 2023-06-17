import numpy as np
import torch

def precision_at_k(r, k):
    assert k >= 1
    r = np.asarray(r)[:k]
    return np.mean(r)


def average_precision(r):
    r = np.asarray(r)
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    if not out:
        return 0.
    return np.mean(out)


def mean_average_precision(rs):
    return np.mean([average_precision(r) for r in rs])


def dcg_at_k(r, k, method=1):
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=1):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def recall_at_k(r, k, all_pos_num):
    r = np.asfarray(r)[:k]
    return np.sum(r) / all_pos_num


def F1(pre, rec):
    if pre + rec > 0:
        return (2.0 * pre * rec) / (pre + rec)
    else:
        return 0.


def MRR(r):
    rf = np.asarray(r).nonzero()[0]
    if rf.size == 0:
        return 0
    else:
        return 1/(rf[0]+1)

def evaluate_test_at_k(model, test_dict, test_input_dict, k=10):
    with torch.no_grad():
        relevant = 0
        selected = 0
        hit = 0
        mrr = 0
        ndcg = 0
        n = 0
        for user in test_dict.keys():
            items = torch.tensor(test_input_dict[user])
            users = torch.tensor([user] * len(items))
            output = model.inference(users, items)
            indices = torch.argsort(output, dim=0, descending=True)[0:k].tolist()
            pred = []
            for idx in indices:
                pred.append(items[idx])
            actual = test_dict[user]
            # print(pred)
            # print(actual)
            reward = 0
            for item in pred:
                if item in actual:
                    reward += 1

            n += reward
            relevant += len(actual)
            selected += len(pred)
            if reward > 0:
                hit += 1

                r = []
                for i in pred:
                    if i in actual:
                        r.append(1)
                    else:
                        r.append(0)
                rf = np.asarray(r).nonzero()[0]
                mrr += 1 / (rf[0] + 1)
                ndcg += ndcg_at_k(r, k)

        print("HIT RATIO@{}: {:.4f}".format(k, hit / len(test_dict.keys())))
        print("MRR@{}: {:.4f}".format(k, mrr / len(test_dict.keys())))
        print("NDCG@{}: {:.4f}".format(k, ndcg / len(test_dict.keys())))
        print("PRECISION@{}: {:.4f}".format(k, n / selected))
        print("RECALL@{}: {:.4f}".format(k, n / relevant))
        print()

def to_onehot(label_matrix, num_classes, device):
    identity = torch.eye(num_classes).to(device)
    onehot = torch.index_select(identity, 0, label_matrix)
    return onehot
