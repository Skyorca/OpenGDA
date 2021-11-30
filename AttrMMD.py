import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from modelUtils import *
from dataUtils import *
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

class Classifier(nn.Module):
    def __init__(self,input_dim,output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        super(Classifier, self).__init__()
        self.fc3 = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, inp):
        logits = self.fc3(inp)
        return logits

def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.1)
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias.data is not None:
            torch.nn.init.constant_(m.bias.data, val=0.1)

def test(tgt_inp,net,clf):
    net.eval()
    with torch.no_grad():
        test_inp = tgt_inp.x
        test_label = tgt_inp.y
        _,test_feat2 = net(test_inp)
        test_out = clf(test_feat2)
    pred = torch.sigmoid(test_out)
    result = f1_scores(pred,test_label)
    return result

def train_test_loop(src_inp,src_name, tgt_inp, tgt_name, net,clf,criterion,batch_size,init_lr):
    # clear model before each training
    net.zero_grad()
    # use all labeled source data to train
    src_train_data = src_inp.x
    src_train_label = src_inp.y
    src_train_num = src_train_data.shape[0]
    tgt_train_data = tgt_inp.x
    tgt_train_num = tgt_train_data.shape[0]
    time = 0
    running_clf_loss = 0.
    running_mmd_loss = 0.
    best_macroF1 =  0.
    best_microF1 = 0.
    best_fe_wts = copy.deepcopy(net.state_dict())
    best_clf_wts = copy.deepcopy(clf.state_dict())
    minibatch_times = int(src_train_num/batch_size)+1
    for epoch in range(epoches):
        # mini-batch
        for start in range(0,src_train_num,batch_size):
            net.train()
            end = min(src_train_num, start+batch_size)
            src_input = src_train_data[start:end,:]
            src_label = src_train_label[start:end,:]
            src_feat1, src_feat2 = net(src_input)
            src_output = clf(src_feat2)
            clf_loss = criterion['clf'](src_output,src_label)
            mmd_loss = torch.tensor(0.)
            # 如果不满足，说明tgt domain已经没有数据了，不存在mmd loss
            if start<tgt_train_num-1:
                end_ = min(tgt_train_num, start+batch_size)
                tgt_input = tgt_train_data[start:end_,:]
                tgt_feat1,tgt_feat2 = net(tgt_input)
                # 计算src tgt domain的最小数据批量，保证MMD计算时二者数据维度相同
                real_size = min((end-start),(end_-start))
                if real_size==batch_size:
                    mmd_loss = 0.5*(criterion['mmd'](src_feat1,tgt_feat1)+criterion['mmd'](src_feat2,tgt_feat2))
                else:
                    print(real_size)
                    src_random_indices = np.random.permutation(end-start)[:real_size]
                    tgt_random_indices = np.random.permutation(end_-start)[:real_size]
                    mmd_loss = 0.5*(criterion['mmd'](src_feat1[src_random_indices],tgt_feat1[tgt_random_indices])+criterion['mmd'](src_feat2[src_random_indices],tgt_feat2[tgt_random_indices]))
            lr = init_lr/(1+10*epoch)**0.75
            #optimizer = torch.optim.SGD(net.parameters(),lr, 0.9,weight_decay=1e-3/2)
            optimizer = torch.optim.Adam(net.parameters(),lr,weight_decay=5e-3/2)
            optimizer.zero_grad()
            loss = clf_loss+mmd_loss
            loss.backward()
            optimizer.step()
            time += 1
            running_clf_loss += clf_loss.item()
            running_mmd_loss += mmd_loss.item()
            if time%20==0 and time>0:
                train_result = test(src_inp,net,clf)
                test_result = test(tgt_inp,net,clf)
                print(f"CLF Loss:{running_clf_loss/20}, MMD Loss:{running_mmd_loss/20},Step:{epoch*minibatch_times+time}, TrainMacroF1:{train_result['macro']}, TrainMicroF1:{train_result['micro']},TestMacroF1:{test_result['macro']}, TestMicroF1:{test_result['micro']}")
                running_clf_loss = 0.
                running_mmd_loss = 0.
                if test_result['macro']>best_macroF1:
                    best_macroF1 = test_result['macro']
                    best_fe_wts = copy.deepcopy(net.state_dict())
                    best_clf_wts = copy.deepcopy(clf.state_dict())
                if test_result['micro']>best_microF1:
                    best_microF1 = test_result['micro']
    torch.save(best_fe_wts, f'checkpoint/{src_name}-{tgt_name}-fe-Attr-MMD.pt')
    torch.save(best_clf_wts, f'checkpoint/{src_name}-{tgt_name}-clf-Attr-MMD.pt')
    return best_macroF1, best_microF1


seed = np.random.randint(1,1000)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
epoches = 30
lr = 0.03
batch_size = 100
clf_criterion = nn.BCEWithLogitsLoss()
sigmas = [
    1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
    1e3, 1e4, 1e5, 1e6
]
kernels = []
for sigma in sigmas:
    kernels.append(GaussianKernel(sigma=sigma))
mmd_criterion = MultipleKernelMaximumMeanDiscrepancy(kernels=kernels)
criterion = {"clf":clf_criterion,"mmd":mmd_criterion}

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
            source_data = src[0]
            source_data = source_data.to(device)
            target_data = tgt[0]
            target_data = target_data.to(device)
            print(source_data.x.shape[0],target_data.x.shape[0])
            net = FE(input_dim=source_data.x.shape[1],output_dim=128)
            net.apply(weight_init)
            net = net.to(device)
            clf = Classifier(input_dim=128,output_dim=source_data.y.shape[1])
            clf.apply(weight_init)
            clf = clf.to(device)
            best_macroF1, best_microF1 = train_test_loop(source_data, src_name, target_data, tgt_name, net, clf, criterion,batch_size,init_lr=lr)
            with open("attr-mmd-results.txt",'a+',encoding='utf8') as f:
                f.write(src_name+'-'+tgt_name+','+'macro '+str(best_macroF1)+','+'micro '+str(best_microF1)+"\n")
