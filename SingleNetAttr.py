import scipy.io as sio
from sklearn.decomposition import PCA
from sklearn.metrics import precision_score,f1_score
from torch_geometric.data import InMemoryDataset, Data
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')
from torch.utils.tensorboard import SummaryWriter
from modelUtils import *
from dataUtils import *

class CitationDomainData(InMemoryDataset):
    """
    引文数据集加载器：acmv9,citationv1,dblpv7 originally from CDNE
    """
    def __init__(self,root,name,use_pca,pca_dim=800,transform=None,pre_transform=None,pre_filter=None):
        self.name=name
        self.use_pca = use_pca
        self.dim = pca_dim
        super(CitationDomainData, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
    @property
    def raw_file_names(self):
        return [f'{self.name}.mat']
    @property
    def processed_file_names(self):
        if self.use_pca:
            return [f'data_pca_{self.dim}.pt']
        else:
            return ['data.pt']

    def feature_compression(self,features):
        """Preprcessing of features"""
        features = features.toarray()
        feat = sp.lil_matrix(PCA(n_components=self.dim, random_state=0).fit_transform(features))
        return feat.toarray()

    def process(self):
        net = sio.loadmat(self.raw_dir+"\\"+self.name+".mat")
        features, adj, labels = net['attrb'], net['network'], net['group']
        if not isinstance(features, sp.lil_matrix):
            features = sp.lil_matrix(features)
        # citation networks do not use PCA, but blog networks use
        if self.use_pca:
            features = self.feature_compression(features)
            features = torch.from_numpy(features).to(torch.float)
        else:
            features = torch.from_numpy(features.todense()).to(torch.float)
        if not isinstance(adj,sp.coo_matrix):
            adj = sp.coo_matrix(adj)
        # label is float: to support BCEWithLogits loss
        y = torch.from_numpy(np.array(labels)).to(torch.float)
        data_list = []
        graph = Data(x=features,edge_index=adj,y=y)
        # train-val-test split
        random_node_indices = np.random.permutation(y.shape[0])
        train_size = int(len(random_node_indices) * 0.7)
        val_size = int(len(random_node_indices) * 0.1)
        train_node_indices = random_node_indices[:train_size]
        val_node_indices = random_node_indices[train_size:train_size+val_size]
        test_node_indices = random_node_indices[train_size+val_size:]
        train_mask = torch.zeros([y.shape[0]], dtype=torch.uint8)
        train_mask[train_node_indices] = 1
        val_mask = torch.zeros([y.shape[0]], dtype=torch.uint8)
        val_mask[val_node_indices] = 1
        test_mask = torch.zeros([y.shape[0]], dtype=torch.uint8)
        test_mask[test_node_indices] = 1
        graph.train_mask = train_mask
        graph.val_mask = val_mask
        graph.test_mask = test_mask
        if self.pre_transform is not None:
            graph = self.pre_transform(graph)
        data_list.append(graph)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class FeatureExtractor(nn.Module):
    def __init__(self,input_dim,output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        super(FeatureExtractor,self).__init__()
        self.h1 = 512
        self.h2 = 128
        self.extract_feature = nn.Sequential(
            nn.Linear(input_dim,self.h1),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(self.h1,self.h2),
            nn.ReLU()
        )
        self.output = nn.Linear(self.h2,self.output_dim)

    def forward(self,input):
        feature = self.extract_feature(input)  # 128-dim feature
        logits = self.output(feature)
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

def val(source_data, net,criterion):
    """ get best multilabel classification threshold"""
    net.eval()
    with torch.no_grad():
        val_inp = source_data.x[source_data.val_mask]
        val_label = source_data.y[source_data.val_mask]
        val_out = net(val_inp)
    pred = F.sigmoid(val_out)
    val_loss = criterion(pred,val_label)
    result = f1_scores(pred,val_label)
    return val_loss, result

def test(source_data,net):
    net.eval()
    with torch.no_grad():
        test_inp = source_data.x[source_data.test_mask]
        test_label = source_data.y[source_data.test_mask]
        test_out = net(test_inp)
    pred = F.sigmoid(test_out)
    result = f1_scores(pred,test_label)
    return result

def train_test_loop(source_data, net, criterion, init_lr, writer):
    # clear model before each training
    net.zero_grad()
    source_train_data = source_data.x[source_data.train_mask]
    source_train_label = source_data.y[source_data.train_mask]
    source_train_num = source_train_data.shape[0]
    time = 0
    running_loss = 0.
    running_macro = []
    running_micro = []
    minibatch_times = int(source_train_num/batch_size)+1
    early_stopping = EarlyStopping(patience=5,verbose=True)
    stop = False
    for epoch in range(epoches):
        if stop: break
        # mini-batch
        for start in range(0,source_train_num,batch_size):
            net.train()
            end = min(source_train_num, start+batch_size)
            src_input = source_train_data[start:end,:]
            src_label = source_train_label[start:end,:]
            src_output = net(src_input)
            loss = criterion(src_output,src_label)
            lr = init_lr/(1+10*epoch)**0.75
            #optimizer = torch.optim.SGD(net.parameters(),lr, 0.9,weight_decay=1e-3/2)
            optimizer = torch.optim.Adam(net.parameters(),lr,weight_decay=5e-3/2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            time += 1
            running_loss += loss.item()
            if time%50==0 and time>0:
                val_loss, val_result = val(source_data,net,criterion)
                early_stopping(val_loss,net)
                if early_stopping.early_stop:
                    print('Early Stopping..')
                    stop=True
                    break
                test_result = test(source_data,net)
                writer.add_scalar("loss",running_loss/50,epoch*minibatch_times+time)
                writer.add_scalar('val_macrof1',val_result['macro'],epoch*minibatch_times+time)
                writer.add_scalar('test_macrof1',test_result['macro'],epoch*minibatch_times+time)
                running_macro.append(test_result['macro'])
                running_micro.append(test_result['micro'])
                #print(f"Loss:{running_loss/50}, Step:{epoch*minibatch_times+time}, Val-MacroF1:{val_result['macro']}, Val-MicroF1:{val_result['micro']},Test-MacroF1:{test_result['macro']}, Test-MicroF1:{test_result['micro']}")
                running_loss = 0.
    print(f"best score: macroF1:{max(running_macro)}, microF1:{max(running_micro)}")
    return max(running_macro),max(running_micro)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
random.seed(200)
np.random.seed(200)
epoches = 100
init_lr = 0.02
batch_size = 100
criterion = nn.BCEWithLogitsLoss()
data_name = "acmv9"
for dim in range(500,1300,100):
    data = CitationDomainData(f"data/{data_name}",name=data_name,use_pca=True,pca_dim=dim)
    # 因为只有一个图，所以0号索引就是我们的图
    inp = data[0]
    inp = inp.to(device)
    net = FeatureExtractor(input_dim=inp.x.shape[1],output_dim=inp.y.shape[1])
    net = net.to(device)
    writer = SummaryWriter('runs/MMD')
    ma,mi = train_test_loop(inp,net,criterion,init_lr,writer)
    with open(f"{data_name}-pca-search.txt",'a',encoding='utf8') as f:
        f.write(str(dim)+','+str(ma)+','+str(mi)+'\n')