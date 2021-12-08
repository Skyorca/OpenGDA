import torch
from torch_geometric.nn import Node2Vec
from modelUtils import *
from dataUtils import *
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score
import os

def f1_scores(y_pred, y_true):
    """ y_pred: prob  y_true 0/1 """
    def predict(y_tru, y_pre):
        top_k_list = y_tru.sum(1)
        prediction = []
        for i in range(y_tru.shape[0]):
            pred_i = torch.zeros(y_tru.shape[1])
            pred_i[torch.argsort(y_pre[i, :])[-int(top_k_list[i].item()):]] = 1
            prediction.append(torch.reshape(pred_i, (1, -1)))
        prediction = torch.cat(prediction, dim=0)
        return prediction.to(torch.int32)
    results = {}
    predictions = predict(y_true, y_pred)
    averages = ["micro", "macro"]
    for average in averages:
        results[average] = f1_score(y_true.cpu().numpy(), predictions.cpu().numpy(), average=average)
    return results

def train(model, loader, optimizer):
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

@torch.no_grad()
def test(model, data):
    model.eval()
    z = model()
    z = z.cpu()
    x_train, y_train = z[data.train_mask], data.y[data.train_mask]
    x_test, y_test   = z[data.test_mask], data.y[data.test_mask]
    clf = MultiOutputClassifier(estimator=LogisticRegression())
    clf = clf.fit(x_train, y_train)
    pred = torch.tensor(clf.predict(x_test))
    results = f1_scores(pred, y_test)
    return z, results

if __name__ =="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    datasets = ["acmv9","citationv1","dblpv7"]
    for i in range(len(datasets)):
        name = datasets[i]
        print(name)
        src = CitationDomainData(f"data/{name}",name=name,use_pca=False)
        src_data = src[0]
        model = Node2Vec(src_data.ppmi_edge_index, embedding_dim=128, walk_length=20,
                         context_size=10, walks_per_node=10,
                         num_negative_samples=1, p=1, q=0.5, sparse=True).to(device)

        loader = model.loader(batch_size=128, shuffle=True, num_workers=0)
        optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)
        best_micro = 0.
        for epoch in range(1, 151):
            loss = train(model, loader, optimizer)
            z, results = test(model,src_data)
            if epoch%10==0:
                print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, MicroF1: {results["micro"]:.4f}')
                if results["micro"]>best_micro:
                    best_micro = results["micro"]
                    os.makedirs("embedding",exist_ok=True)
                    torch.save(z,f"embedding/node2vec_ppmi_{name}.pt")

