import numpy as np
import torch
from sklearn.metrics import f1_score


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
        return {"hits": hit / len(test_dict.keys()), "mrr":mrr / len(test_dict.keys()), "ndcg":ndcg / len(test_dict.keys())}


def multilabel_f1(**kwargs):
    """
    Ref  github/ACDNE (pytorch ver)
    y_pred: prob  y_true 0/1
    """
    y_true = kwargs['y_true']
    y_pred = kwargs['y_pred']
    def _predict(y_tru, y_pre):
        top_k_list = y_tru.sum(1)
        prediction = []
        for i in range(y_tru.shape[0]):
            pred_i = torch.zeros(y_tru.shape[1])
            pred_i[torch.argsort(y_pre[i, :])[-int(top_k_list[i].item()):]] = 1
            prediction.append(torch.reshape(pred_i, (1, -1)))
        prediction = torch.cat(prediction, dim=0)
        return prediction.to(torch.int32)
    results = {}
    predictions = _predict(y_true, y_pred)
    averages = ["micro", "macro"]
    for average in averages:
        results[average] = f1_score(y_true.cpu().numpy(), predictions.cpu().numpy(), average=average)
    return results['micro'], results['macro']


def multilabel_acc(**kwargs):
    y_true = kwargs['y_true']
    y_pred = kwargs['y_pred']
    top_k_list = torch.sum(y_true,dim=1, dtype=int)
    prediction = []
    for i in range(y_true.shape[0]):
        pred_i = torch.zeros(y_true.shape[1])
        pred_i[torch.argsort(y_pred[i,:])[-top_k_list[i]:]]=1
        prediction.append(pred_i.reshape(1,-1))
    prediction = torch.vstack(prediction)
    c = 0
    for i in range(y_true.shape[0]):
        if torch.sum(y_true[i,:]@prediction[i,:].T)>0: c+= 1
    return c/y_true.shape[0]


def f1(**kwargs):
    y_true = kwargs['y_true']
    y_pred = kwargs['y_pred']
    results = {}
    averages = ["micro", "macro"]
    for average in averages:
        results[average] = f1_score(y_true=torch.argmax(y_true,1).cpu().numpy(), y_pred=torch.argmax(y_pred,1).cpu().numpy(), average=average)
    return results['micro'], results['macro']