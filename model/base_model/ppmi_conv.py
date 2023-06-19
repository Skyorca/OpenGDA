import sys
sys.path.append('..')
from collections import Counter
import torch
from torch_scatter import scatter_add
from torch_geometric.utils import add_remaining_self_loops
import numpy as np
from tqdm import tqdm
from base_model.cached_gcn_conv import CachedGCNConv


class PPMIConv(CachedGCNConv):

    def __init__(self, in_channels, out_channels,
                 weight=None, bias=None, improved=False, use_bias=True,
                 path_len=3,device='cpu',
                 **kwargs):
        super().__init__(in_channels, out_channels, weight, bias, improved, use_bias, **kwargs)
        self.path_len = path_len
        self.device = device


    def norm(self, edge_index, num_nodes, edge_weight=None, improved=False,
             dtype=None):

        adj_dict = {}  # {n0:(n1,n2,...,n3), n1:(n2,n3)} 相当于是邻接表

        def add_edge(a, b):
            if a in adj_dict:
                neighbors = adj_dict[a]
            else:
                neighbors = set()
                adj_dict[a] = neighbors
            if b not in neighbors:
                neighbors.add(b)

        cpu_device = torch.device("cpu")
        gpu_device = torch.device("cuda")
        for a, b in edge_index.t().detach().to('cpu').numpy():
            a = int(a)
            b = int(b)
            add_edge(a, b)
            add_edge(b, a)

        adj_dict = {a: list(neighbors) for a, neighbors in adj_dict.items()}  # 邻接表中set转换成list

        def sample_neighbor(a):
            """随机采样一个邻居"""
            neighbors = adj_dict[a]
            random_index = np.random.randint(0, len(neighbors))
            return neighbors[random_index]


        # word_counter = Counter()
        walk_counters = {}  # 重复40次：在邻接表上，从每个节点a进行长度为path_len的随机游走，记录a与路径上的节点的共现次数

        def norm(counter):
            s = sum(counter.values())
            new_counter = Counter()
            for a, count in counter.items():
                new_counter[a] = counter[a] / s
            return new_counter
        sample_times = 40
        print('sample times: ', sample_times)
        for _ in range(sample_times):
            for a in adj_dict:
                current_a = a
                current_path_len = np.random.randint(1, self.path_len + 1)
                for _ in range(current_path_len):
                    b = sample_neighbor(current_a)
                    if a in walk_counters:
                        walk_counter = walk_counters[a]
                    else:
                        walk_counter = Counter()
                        walk_counters[a] = walk_counter

                    walk_counter[b] += 1

                    current_a = b
        # 对每个节点归一化：共现次数-->共现概率
        normed_walk_counters = {a: norm(walk_counter) for a, walk_counter in walk_counters.items()}

        # 每个节点的出现概率总和 Fi,: or F:,j
        prob_sums = Counter()

        for a, normed_walk_counter in normed_walk_counters.items():
            for b, prob in normed_walk_counter.items():
                prob_sums[b] += prob

        ppmis = {}
        # prob是Fij ij节点共现概率; prob_sums[b]是F:j b节点出现总概率; \sum_i \sum_j Fij是len(prob_sums);
        for a, normed_walk_counter in normed_walk_counters.items():
            for b, prob in normed_walk_counter.items():
                ppmi = np.log(prob / prob_sums[b] * len(prob_sums) / self.path_len)
                ppmis[(a, b)] = ppmi

        new_edge_index = []
        edge_weight = []
        for (a, b), ppmi in ppmis.items():
            new_edge_index.append([a, b])
            edge_weight.append(ppmi)

        # PPMI-GCN生成的edge_index是在gpu上
        edge_index = torch.tensor(new_edge_index).t().to(self.device)
        edge_weight = torch.tensor(edge_weight).to(self.device)

        fill_value = 1. if not improved else 2.
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)

        row, col = edge_index
        deg = scatter_add(edge_weight, row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        # Note: modify here, set nan to 0
        deg_inv_sqrt[deg_inv_sqrt != deg_inv_sqrt] = 0
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        return edge_index, (deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]).type(torch.float32)





