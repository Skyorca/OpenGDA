import torch
import numpy as np
import os
import os.path as osp
import shutil
from typing import Callable, List, Optional
from data.utils import read_tu_data
from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T

class OurTUDataset(InMemoryDataset):
    """
    from pyg TUDataset with modifications mainly adding node features (x)
    """

    url = 'https://www.chrsmrrs.com/graphkerneldatasets'
    cleaned_url = ('https://raw.githubusercontent.com/nd7141/'
                   'graph_datasets/master/datasets')

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None,
                 use_node_attr: bool = True, use_edge_attr: bool = False,
                 cleaned: bool = False):
        self.name = name
        self.cleaned = cleaned
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])
        if self.data.x is not None and not use_node_attr:
            num_node_attributes = self.num_node_attributes
            self.data.x = self.data.x[:, num_node_attributes:]
        if self.data.edge_attr is not None and not use_edge_attr:
            num_edge_attributes = self.num_edge_attributes
            self.data.edge_attr = self.data.edge_attr[:, num_edge_attributes:]

    @property
    def raw_dir(self) -> str:
        return self.root+'/'+self.name+'/raw'

    @property
    def processed_dir(self) -> str:
        return self.root+'/'+self.name+'/processed'

    @property
    def num_node_labels(self) -> int:
        if self.data.x is None:
            return 0
        for i in range(self.data.x.size(1)):
            x = self.data.x[:, i:]
            if ((x == 0) | (x == 1)).all() and (x.sum(dim=1) == 1).all():
                return self.data.x.size(1) - i
        return 0

    @property
    def num_node_attributes(self) -> int:
        if self.data.x is None:
            return 0
        return self.data.x.size(1) - self.num_node_labels

    @property
    def num_edge_labels(self) -> int:
        if self.data.edge_attr is None:
            return 0
        for i in range(self.data.edge_attr.size(1)):
            if self.data.edge_attr[:, i:].sum() == self.data.edge_attr.size(0):
                return self.data.edge_attr.size(1) - i
        return 0

    @property
    def num_edge_attributes(self) -> int:
        if self.data.edge_attr is None:
            return 0
        return self.data.edge_attr.size(1) - self.num_edge_labels

    @property
    def raw_file_names(self) -> List[str]:
        names = ['A', 'graph_indicator','graph_labels']
        return [f'{self.name}_{name}.txt' for name in names] + [f'{self.name}_node_attributes.pt']

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'


    def process(self):
        self.data, self.slices = read_tu_data(self.raw_dir, self.name)

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        # if self.pre_transform is not None:
        #     data_list = [self.get(idx) for idx in range(len(self))]
        #     data_list = [self.pre_transform(data) for data in data_list]
        #     self.data, self.slices = self.collate(data_list)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}({len(self)})'


def dataloader_gc(model_name, dataset_name, collection_name, device="cpu"):
    # register dataset name
    if 'BINARY' in collection_name:
        dl = OurTUDataset
    elif 'Letter' in collection_name:
        dl = TUDataset
    dataset = dl(root=f"../../data/gc/{dataset_name}", name=collection_name, use_node_attr=True)
    if model_name=='pyg':
        return dataset