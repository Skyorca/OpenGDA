from dataUtils import *
import networkx as nx
from grakel import Graph
from grakel.kernels import WeisfeilerLehman, VertexHistogram, GraphletSampling, RandomWalk

datasets = ["acmv9","citationv1","dblpv7"]
g1 = CitationDomainData(f"data/acmv9",name="acmv9",use_pca=False)
g2 = CitationDomainData(f"data/citationv1",name="citationv1",use_pca=False)
g3 = CitationDomainData(f"data/dblpv7",name="dblpv7",use_pca=False)
g1_ = nx.from_scipy_sparse_matrix(g1[0].edge_index)
g1_label = {pair[0]:pair[1] for pair in g1_.degree()}
g2_ = nx.from_scipy_sparse_matrix(g2[0].edge_index)
g2_label = {pair[0]:pair[1] for pair in g2_.degree()}
g3_ = nx.from_scipy_sparse_matrix(g3[0].edge_index)
g3_label = {pair[0]:pair[1] for pair in g3_.degree()}
inp1 = Graph(g1[0].edge_index,node_labels=g1_label)
inp2 = Graph(g2[0].edge_index,node_labels=g2_label)
inp3 = Graph(g3[0].edge_index,node_labels=g3_label)
gk = WeisfeilerLehman(n_iter=10, base_graph_kernel=VertexHistogram, normalize=True)
sim = gk.fit_transform([inp1, inp2, inp3])
print(sim)
"""
[[1.         0.94741758 0.93084741]
 [0.94741758 1.         0.98474622]
 [0.93084741 0.98474622 1.        ]]
"""
gk2 = RandomWalk(n_jobs=4,normalize=True)
sim2 = gk2.fit_transform([inp1, inp2, inp3])
print(sim2)



