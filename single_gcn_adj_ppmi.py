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
import itertools

class PPMIGCN(nn.Module):
    def __init__(self,input_dim,output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        super(PPMIGCN, self).__init__()
        self.h1 = 512
        self.h2 = 128
        self.conv1 = GCNConv(self.input_dim, self.h1)
        self.conv2 = GCNConv(self.h1, self.h2)
        self.prelu = nn.PReLU()
    def forward(self,data):
        x, edge_index, edge_attr = data.x, data.ppmi_edge_index, data.ppmi_edge_attr
        feat1 = F.dropout(self.prelu(self.conv1(x, edge_index,edge_weight=edge_attr)))
        feat2 = F.dropout(self.prelu(self.conv2(feat1, edge_index, edge_weight=edge_attr)))
        return feat2


class GCN(nn.Module):
    def __init__(self,input_dim,output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        super(GCN, self).__init__()
        self.h1 = 512
        self.h2 = 128
        self.conv1 = GCNConv(self.input_dim, self.h1)
        self.conv2 = GCNConv(self.h1, self.h2)
        self.prelu = nn.PReLU()
    def forward(self,data):
        x, edge_index = data.x, data.edge_index
        feat1 = F.dropout(self.prelu(self.conv1(x, edge_index)))
        feat2 = F.dropout(self.prelu(self.conv2(feat1, edge_index)))
        return feat2

class Attention(nn.Module):
    """
    该种实现方法用在UDAGCN、ASN中，但并没有像论文中那样把原始输入建模进来
    """
    def __init__(self, in_channels):
        super().__init__()
        self.dense_weight = nn.Linear(in_channels, 1)
        self.dropout = nn.Dropout(0.1)
    def forward(self, inputs):
        # [bs,2,128]
        stacked = torch.stack(inputs, dim=1)
        # [bs,2,1]
        weights = F.softmax(self.dense_weight(stacked), dim=1)  # 因为这里是三维张量，所以dim=1实际上就是2维情况的dim=0
        # [bs,128]
        outputs = torch.sum(stacked * weights, dim=1)
        return outputs

class Classifier(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(in_channels,out_channels)
    def forward(self, x):
        logits = self.fc(x)
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

def test(inp,gcn, ppmi_gcn, attn, clf):
    gcn.eval()
    ppmi_gcn.eval()
    attn.eval()
    clf.eval()
    with torch.no_grad():
        l_o = gcn(inp)[inp.test_mask]
        g_o = ppmi_gcn(inp)[inp.test_mask]
        attn_o = attn([l_o,g_o])
        test_out = clf(attn_o)
    pred = F.sigmoid(test_out)
    result = f1_scores(pred,inp.y[inp.test_mask])
    return result

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
datasets = ["acmv9","citationv1","dblpv7"]
for i in range(len(datasets)):
    src_name = datasets[i]
    print(src_name)
    src = CitationDomainData(f"data/{src_name}",name=src_name,use_pca=False)
    src_data = src[0].to(device)
    inp_dim = src_data.x.shape[1]
    out_dim = src_data.y.shape[1]
    gcn = GCN(inp_dim, 128).to(device)
    ppmi_gcn = PPMIGCN(inp_dim, 128).to(device)
    attn = Attention(128).to(device)
    clf = Classifier(128,out_dim).to(device)
    models = [gcn, ppmi_gcn, attn, clf]
    criterion = nn.BCEWithLogitsLoss()
    params = itertools.chain(*[model.parameters() for model in models])
    optimizer = torch.optim.Adam(params, lr=0.01, weight_decay=5e-4)
    gcn.zero_grad()
    ppmi_gcn.zero_grad()
    attn.zero_grad()
    clf.zero_grad()
    gcn.train()
    ppmi_gcn.train()
    attn.train()
    clf.train()
    running_loss = 0.
    best_macro = 0.
    best_micro = 0.
    for epoch in range(300):
        optimizer.zero_grad()
        l_out = gcn(src_data)
        g_out = ppmi_gcn(src_data)
        attn_out = attn([l_out,g_out])
        out = clf(attn_out)
        clf_loss = criterion(out[src_data.train_mask], src_data.y[src_data.train_mask])
        loss = clf_loss
        running_loss += loss
        loss.backward()
        optimizer.step()
        if epoch%10==0 and epoch>0:
            test_res = test(src_data, gcn, ppmi_gcn, attn, clf)
            print(f"EPOCH:{epoch}, Loss:{running_loss/10}, Source: MacroF1={test_res['macro']}, MicroF1={test_res['micro']}")
            if test_res['macro']>best_macro:
                best_macro = test_res['macro']
                #best_gcn_wts = copy.deepcopy(model.state_dict())
            if test_res['micro']>best_micro:
                best_micro = test_res['micro']
            running_loss = 0.
    #torch.save(best_gcn_wts, f"checkpoint/{src_name}-{tgt_name}-PPMI-gcn.pt")
    #with open('重要的原始数据/gcn-PPMI-results.txt', 'a+', encoding='utf8') as f:
    #    f.write(src_name+'-'+tgt_name+','+'macro '+str(best_macro)+','+'micro '+str(best_micro)+"\n")