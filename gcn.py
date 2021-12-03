from modelUtils import *
from dataUtils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')
import copy


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
        x, edge_index = data.x, data.edge_index
        feat1 = F.dropout(self.prelu(self.conv1(x, edge_index)))
        feat2 = F.dropout(self.prelu(self.conv2(feat1, edge_index)))
        logits = self.fc(feat2)
        return logits


def f1_scores(y_pred, y_true):
    """
    Ref  github/ACDNE (pytorch ver)
    y_pred: prob  y_true 0/1
    """
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

def test_tgt(inp, net):
    net.eval()
    with torch.no_grad():
        test_out = net(inp)
    pred = F.sigmoid(test_out)
    result = f1_scores(pred,inp.y)
    return result



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
datasets = ["acmv9","citationv1","dblpv7"]
for i in range(len(datasets)):
    for j in range(i+1,len(datasets)):
        dual_ds = [(i,j),(j,i)]
        for ds in dual_ds:
            src_name = datasets[ds[0]]
            tgt_name = datasets[ds[1]]
            print(src_name, tgt_name)
            src = CitationDomainData(f"data/{src_name}",name=src_name,use_pca=False)
            tgt = CitationDomainData(f"data/{tgt_name}",name=tgt_name,use_pca=False)
            src_data = src[0].to(device)
            tgt_data = tgt[0].to(device)
            inp_dim = src_data.x.shape[1]
            out_dim = src_data.y.shape[1]
            # if use adj
            #model = GCNClassifier(inp_dim, out_dim).to(device)
            #if use ppmi
            model = PPMIGCNClassifier(inp_dim, out_dim).to(device)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.02, weight_decay=5e-4)
            model.zero_grad()
            model.train()
            running_loss = 0.
            best_macro = 0.
            best_micro = 0.
            best_gcn_wts = copy.deepcopy(model.state_dict())
            for epoch in range(200):
                optimizer.zero_grad()
                out = model(src_data)
                train_mask = src_data.train_mask+src_data.val_mask
                clf_loss = criterion(out[train_mask], src_data.y[train_mask])
                target_probs = F.softmax(out[tgt_data.train_mask], dim=-1)
                target_probs = torch.clamp(target_probs, min=1e-9, max=1.0)
                entropy_loss = torch.mean(torch.sum(-target_probs * torch.log(target_probs), dim=-1))
                loss = clf_loss + (epoch/200)*0.01*entropy_loss
                #loss = clf_loss
                running_loss += loss
                loss.backward()
                optimizer.step()
                if epoch%10==0 and epoch>0:
                    test_res = test(src_data, model)
                    tgt_res = test_tgt(tgt_data, model)
                    print(f"EPOCH:{epoch}, Loss:{running_loss/10}, Source: MacroF1={test_res['macro']}, MicroF1={test_res['micro']}, Target: MacroF1={tgt_res['macro']}, MicroF1={tgt_res['micro']}")
                    if tgt_res['macro']>best_macro:
                        best_macro = tgt_res['macro']
                        best_gcn_wts = copy.deepcopy(model.state_dict())
                    if tgt_res['micro']>best_micro:
                        best_micro = tgt_res['micro']
                    running_loss = 0.
            #torch.save(best_gcn_wts, f"checkpoint/{src_name}-{tgt_name}-PPMI-gcn.pt")
            with open('重要的原始数据/gcn-PPMI-results.txt', 'a+', encoding='utf8') as f:
                f.write(src_name+'-'+tgt_name+','+'macro '+str(best_macro)+','+'micro '+str(best_micro)+"\n")