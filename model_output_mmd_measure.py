import torch
import torch.nn
from modelUtils import *
from dataUtils import *
import torch.nn.functional as F

####### model ############
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
            src_data = src[0]
            tgt_data = tgt[0]
            # add pre-computed features
            fe = FE(input_dim=src_data.x.shape[1],output_dim=128)
            fe.load_state_dict(torch.load(f'checkpoint/{src_name}-{tgt_name}-fe-Attr-MMD.pt'))
            fe.eval()
            with torch.no_grad():
                _, src_feature_trans = fe(src_data.x)
                _, tgt_feature_trans = fe(tgt_data.x)
            size = min(src_feature_trans.shape[0],tgt_feature_trans.shape[0])
            random_indices = np.random.permutation(size)
            x = src_feature_trans[random_indices]
            y = tgt_feature_trans[random_indices]
            sigmas = [
                1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
                1e3, 1e4, 1e5, 1e6
            ]
            kernels = []
            for sigma in sigmas:
                kernels.append(GaussianKernel(sigma=sigma))
            attr_mmd = 0.
            mmd2 = MultipleKernelMaximumMeanDiscrepancy(kernels=kernels)
            bs = 64
            for start in range(0,size,bs):
                end = min(size-1,start+bs)
                attr_mmd += mmd2(x[start:end],y[start:end])
            print(src_name, tgt_name, attr_mmd)