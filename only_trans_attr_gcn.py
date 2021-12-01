from modelUtils import *
from dataUtils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore',category=Warning)
import copy

class FE(nn.Module):
    def __init__(self,input_dim,output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        super(FE,self).__init__()
        self.h1 = 512
        self.fc1 = nn.Linear(input_dim,self.h1)
        self.bn1 = nn.BatchNorm1d(self.h1)
        self.fc2 = nn.Linear(self.h1,self.output_dim)
        self.bn2 = nn.BatchNorm1d(self.output_dim)

    def forward(self,inp):
        feat1 =  F.dropout(F.relu(self.bn1(self.fc1(inp))))  # 128-dim feature
        feat2 = F.relu(self.bn2(self.fc2(feat1)))
        return feat1,feat2

class GCNClassifier(nn.Module):
    def __init__(self,input_dim,output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        super(GCNClassifier, self).__init__()
        self.h1 = 512
        self.h2 = 128
        self.conv1 = GCNConv(self.input_dim, self.h1)
        self.conv2 = GCNConv(self.h1, self.h2)
        self.fc = nn.Linear(self.h2, self.output_dim)
        self.prelu = nn.PReLU()
    def forward(self,x, edge_index):
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

def test(x,inp,net):
    net.eval()
    with torch.no_grad():
        test_out = net(x,inp.edge_index)[inp.test_mask]
    pred = F.sigmoid(test_out)
    result = f1_scores(pred,inp.y[inp.test_mask])
    return result

def test_tgt(x,inp, net):
    net.eval()
    with torch.no_grad():
        test_out = net(x,inp.edge_index)
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
            # add pre-computed features
            fe = FE(input_dim=src_data.x.shape[1],output_dim=128)
            fe.load_state_dict(torch.load(f'checkpoint/{src_name}-{tgt_name}-fe-Attr-MMD.pt'))
            fe = fe.to(device)
            fe.eval()
            with torch.no_grad():
                _, src_feature_trans = fe(src_data.x)
                _, tgt_feature_trans = fe(tgt_data.x)
            src_feature_trans = src_feature_trans.to(device)
            tgt_feature_trans = tgt_feature_trans.to(device)
            inp_dim = src_feature_trans.shape[1]
            out_dim = src_data.y.shape[1]
            model = GCNClassifier(inp_dim, out_dim).to(device)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
            model.zero_grad()
            model.train()
            running_loss = 0.
            best_macro = 0.
            best_micro = 0.
            best_gcn_wts = copy.deepcopy(model.state_dict())
            for epoch in range(200):
                optimizer.zero_grad()
                out = model(src_feature_trans, src_data.edge_index)
                loss = criterion(out[src_data.train_mask], src_data.y[src_data.train_mask])
                running_loss += loss
                loss.backward()
                optimizer.step()
                if epoch%10==0 and epoch>0:
                    test_res = test(src_feature_trans,src_data, model)
                    tgt_res = test_tgt(tgt_feature_trans,tgt_data, model)
                    print(f"EPOCH:{epoch}, Loss:{running_loss/10}, Source: MacroF1={test_res['macro']}, MicroF1={test_res['micro']}, Target: MacroF1={tgt_res['macro']}, MicroF1={tgt_res['micro']}")
                    if tgt_res['macro']>best_macro:
                        best_macro = tgt_res['macro']
                        best_gcn_wts = copy.deepcopy(model.state_dict())
                    if tgt_res['micro']>best_micro:
                        best_micro = tgt_res['micro']
                    running_loss = 0.
            #torch.save(best_gcn_wts, f"checkpoint/{src_name}-{tgt_name}-gcn.pt")
            with open('gcn-only-trans-attr-results.txt','a+',encoding='utf8') as f:
                f.write(src_name+'-'+tgt_name+','+'macro '+str(best_macro)+','+'micro '+str(best_micro)+"\n")