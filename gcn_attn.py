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
        self.h1 = 128
        self.h2 = 16
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
        self.h1 = 128
        self.h2 = 16
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


class Attention2(nn.Module):
    """ 实现论文中描述的注意力机制  实验表明f = Q^T*K 比归一化要效果好"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dense_weight = nn.Linear(in_channels, out_channels)  # change Zo to the shape of Zl、Zg
        self.dropout = nn.Dropout(0.1)
    def forward(self, inputs):
        """inputs = [Zo, Zl, Zg]"""
        Z_o = inputs[0]
        Z_l = inputs[1]
        Z_g = inputs[2]
        Z_o_trans = self.dense_weight(Z_o)
        att_l = torch.sum(torch.matmul(Z_o_trans, torch.t(Z_l)),dim=1)
        att_g = torch.sum(torch.matmul(Z_o_trans, torch.t(Z_g)),dim=1)
        att = torch.hstack([att_l.view(-1,1), att_g.view(-1,1)])
        att = F.softmax(att, dim=1)
        outputs = Z_l*(att[:,0].view(-1,1))+Z_g*(att[:,1].view(-1,1))
        return outputs

class Classifier(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(in_channels,out_channels)
    def forward(self, x):
        logits = self.fc(x)
        return logits

def encode_data(data,gcn,ppmi_gcn,attn,clf,attn_mode):
    """ 数据从输入到输出的映射
        data: src/tgt 是Data类
    """
    l_out = gcn(data)
    g_out = ppmi_gcn(data)
    if attn_mode==1:
        attn_out = attn([l_out, g_out])
    else:
        attn_out = attn([data.x, l_out, g_out])
    out = clf(attn_out)
    return out


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

def test(inp,gcn, ppmi_gcn, attn, clf, mask, attn_mode):
    gcn.eval()
    ppmi_gcn.eval()
    attn.eval()
    clf.eval()
    with torch.no_grad():
        l_o = gcn(inp)[mask]
        g_o = ppmi_gcn(inp)[mask]
        if attn_mode==1:
            attn_o = attn([l_o,g_o])
        else:
            attn_o = attn([inp.x[mask],l_o,g_o])
        test_out = clf(attn_o)
    pred = F.sigmoid(test_out)
    result = f1_scores(pred,inp.y[mask])
    return result

def test_tgt(inp,gcn, ppmi_gcn, attn, clf,attn_mode):
    gcn.eval()
    ppmi_gcn.eval()
    attn.eval()
    clf.eval()
    with torch.no_grad():
        l_o = gcn(inp)
        g_o = ppmi_gcn(inp)
        if attn_mode==1:
            attn_o = attn([l_o,g_o])
        elif attn_mode==2:
            attn_o = attn([inp.x,l_o,g_o])
        else:
            pass
        test_out = clf(attn_o)
    pred = F.sigmoid(test_out)
    result = f1_scores(pred,inp.y)
    return result


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
### model mode selection ###
attn_mode = 2
use_entropy_loss = True
############################
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
            gcn = GCN(inp_dim, 16).to(device)
            ppmi_gcn = PPMIGCN(inp_dim, 16).to(device)
            if attn_mode==1:
                attn = Attention(16).to(device)
            else:
                attn = Attention2(inp_dim,16).to(device)
            clf = Classifier(16,out_dim).to(device)
            models = [gcn, ppmi_gcn, attn, clf]
            criterion = nn.BCEWithLogitsLoss()
            params = itertools.chain(*[model.parameters() for model in models])
            optimizer = torch.optim.Adam(params, lr=0.02, weight_decay=5e-4)
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
            for epoch in range(200):
                optimizer.zero_grad()
                src_out = encode_data(src_data,gcn,ppmi_gcn,attn,clf,attn_mode)
                src_train_mask = src_data.train_mask+src_data.val_mask
                clf_loss = criterion(src_out[src_train_mask], src_data.y[src_train_mask])
                loss = clf_loss
                if use_entropy_loss:
                    tgt_train_mask = tgt_data.train_mask + tgt_data.val_mask
                    tgt_out = encode_data(tgt_data,gcn,ppmi_gcn,attn,clf,attn_mode)
                    target_probs = F.softmax(tgt_out[tgt_train_mask], dim=-1)
                    target_probs = torch.clamp(target_probs, min=1e-9, max=1.0)
                    entropy_loss = torch.mean(torch.sum(-target_probs * torch.log(target_probs), dim=-1))
                    loss += (epoch/200)*0.01*entropy_loss
                running_loss += loss
                loss.backward()
                optimizer.step()
                if epoch%10==0 and epoch>0:
                    test_res = test(src_data, gcn, ppmi_gcn, attn, clf, mask=src_data.test_mask, attn_mode=attn_mode)
                    tgt_res = test_tgt(tgt_data, gcn, ppmi_gcn, attn, clf, attn_mode=attn_mode)
                    print(f"EPOCH:{epoch}, Loss:{running_loss/10}, Source: MacroF1={test_res['macro']}, MicroF1={test_res['micro']}, Target: MacroF1={tgt_res['macro']}, MicroF1={tgt_res['micro']}")
                    if tgt_res['macro']>best_macro:
                        best_macro = tgt_res['macro']
                        #best_gcn_wts = copy.deepcopy(model.state_dict())
                    if tgt_res['micro']>best_micro:
                        best_micro = tgt_res['micro']
                    running_loss = 0.
            #torch.save(best_gcn_wts, f"checkpoint/{src_name}-{tgt_name}-PPMI-gcn.pt")
            with open('重要的原始数据/gcn-ATTN-results.txt', 'a+', encoding='utf8') as f:
                f.write(src_name+'-'+tgt_name+','+'macro '+str(best_macro)+','+'micro '+str(best_micro)+"\n")