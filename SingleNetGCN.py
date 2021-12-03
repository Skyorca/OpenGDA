import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')
import copy
from dataUtils import *
from modelUtils import *

class PPMIGCNClassifier(nn.Module):
    def __init__(self,input_dim,output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        super(PPMIGCNClassifier, self).__init__()
        self.h1 = 128
        self.h2 = 16
        self.conv1 = GCNConv(self.input_dim, self.h1)
        self.conv2 = GCNConv(self.h1, self.h2)
        self.fc = nn.Linear(self.h2, self.output_dim)
        self.prelu = nn.PReLU()
    def forward(self,data):
        x, edge_index, edge_attr = data.x, data.ppmi_edge_index, data.ppmi_edge_attr
        feat1 = F.dropout(self.prelu(self.conv1(x, edge_index,edge_weight=edge_attr)))
        feat2 = F.dropout(self.prelu(self.conv2(feat1, edge_index, edge_weight=edge_attr)))
        logits = self.fc(feat2)
        return logits

class GCNClassifier(nn.Module):
    def __init__(self,input_dim,output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        super(GCNClassifier, self).__init__()
        self.h1 = 128
        self.h2 = 16
        self.conv1 = GCNConv(self.input_dim, self.h1)
        self.conv2 = GCNConv(self.h1, self.h2)
        self.fc = nn.Linear(self.h2, self.output_dim)
        self.prelu = nn.PReLU()
    def forward(self,data):
        x, edge_index = data.x, data.ppmi_edge_index
        feat1 = F.dropout(self.prelu(self.conv1(x, edge_index)))
        feat2 = F.dropout(self.prelu(self.conv2(feat1, edge_index)))
        logits = self.fc(feat2)
        return logits

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

def test(inp,net):
    net.eval()
    with torch.no_grad():
        test_out = net(inp)[inp.test_mask]
    pred = F.sigmoid(test_out)
    result = f1_scores(pred,inp.y[inp.test_mask])
    return result

def net_pro_loss(emb, edge_index, edge_attr):
    """
       Ref  github/ACDNE (pytorch ver)
       保持网络结构的损失
       emb  节点表达矩阵   a 图的邻接矩阵/ppmi矩阵等COO格式的tensor
    """
    a = torch.sparse_coo_tensor(indices=edge_index, values=edge_attr)
    r = torch.sum(emb*emb, 1)
    r = torch.reshape(r, (-1, 1))
    dis = r-2*torch.matmul(emb, emb.T)+r.T
    return torch.mean(torch.sum(torch.sparse.mm(a,dis), 1))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
datasets = ["dblpv7","acmv9","citationv1"]
for i in range(len(datasets)):
    name = datasets[i]
    print(name)
    src = CitationDomainData(f"data/{name}",name=name,use_pca=False)
    src_data = src[0].to(device)
    inp_dim = src_data.x.shape[1]
    out_dim = src_data.y.shape[1]
    model = GCNClassifier(inp_dim, out_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=5e-4)
    model.zero_grad()
    model.train()
    running_loss = 0.
    best_macro = 0.
    best_micro = 0.
    #best_gcn_wts = copy.deepcopy(model.state_dict())
    for epoch in range(500):
        optimizer.zero_grad()
        out = model(src_data)
        train_mask = src_data.train_mask+src_data.val_mask
        loss = criterion(out[train_mask], src_data.y[train_mask])
        pro_loss = net_pro_loss(out, src_data.ppmi_edge_index, src_data.ppmi_edge_attr)
        running_loss += (loss+1e-6*pro_loss)
        loss.backward()
        optimizer.step()
        if epoch%10==0 and epoch>0:
            test_res = test(src_data, model)
            print(f"EPOCH:{epoch}, Loss:{running_loss/10}, Source: MacroF1={test_res['macro']}, MicroF1={test_res['micro']}")
            if test_res['macro']>best_macro:
                best_macro = test_res['macro']
                best_gcn_wts = copy.deepcopy(model.state_dict())
            if test_res['micro']>best_micro:
                best_micro = test_res['micro']
            running_loss = 0.
    #torch.save(best_gcn_wts, f"checkpoint/{src_name}-{tgt_name}-gcn.pt")
    with open('重要的原始数据/single-gcn-results.txt', 'a+', encoding='utf8') as f:
        f.write(name+','+'macro '+str(best_macro)+','+'micro '+str(best_micro)+"\n")