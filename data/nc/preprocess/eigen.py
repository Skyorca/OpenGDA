import sys
sys.path.append('../../..')
import scipy.sparse as sp
import scipy.io as sio
import numpy as np
import torch
from sklearn.decomposition import TruncatedSVD
from data.dataloader import dataloader

# load dataset as COO edge index tensor
dataset = 'airport'
name = 'europe'
# Note: You may modify `nc_dataloader.py` line 372 to  dataset = dataloader(root=f"../../../data/nc/{dataset_name}/{collection_name}", name=collection_name) for using this file
# use other relative positions
src = dataloader(task_type="nc",model_name="pyg", dataset_name=dataset, collection_name=name)[0]
edge_index = src.edge_index
n_node = src.x.shape[0]

row = edge_index[0,:].numpy()
col = edge_index[1,:].numpy()
adj = sp.coo_matrix((np.ones_like(row),(row,col)), shape=(n_node,n_node))
adj = sp.csr_matrix(adj)
adj.setdiag(-adj.sum(axis=1))
adj = -adj
svd = TruncatedSVD(n_components=100, n_iter=20, random_state=42)
svd.fit(adj)
eival = torch.tensor(svd.explained_variance_ ** 0.5, dtype=torch.float32).to('cuda')
eivec = torch.tensor(svd.components_, dtype=torch.float32).to('cuda')
torch.save(eivec, f"{name}-eivec.pt")
torch.save(eival, f"{name}-eival.pt")
