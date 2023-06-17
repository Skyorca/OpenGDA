### Instructions to data resource for cross-network graph classification task

Folder `\preprocess` provides codes for preprocessing raw data. 
#### IMDB-REDDIT Dataset
This dataset originates from TUDataset, so please refer to that for more details.

As the original datasets provide no node features, we use one-hot node degree vector as node feature, and
set the maximum degree as 135 for both IMDB and REDDIT (135 is the maximum degree in IMDB), leading to a 136-dim
feature vector. In practice, you should generate a file entitled `IMDB-BINARY_node_attributes.pt` and put it with other raw files together in the folder: `\data\gc\TUDataset\IMDB-BINARY\raw`. Pre-processing codes are
provided in path `\data\gc\preprocess`

Why do not directly use `OneHotTransform` provided by PyG: The maximum degree for REDDIT is 3000+, thus use 135 will report error.
We hope two domains share the same feature space, and thus set the nodes in REDDIT which degree is larger than 135 to 135 manually.


#### IMDB-REDDIT Dataset
This dataset originates from TUDataset, so please refer to that for more details.

Generally, we use the original node features provided by the dataset, so no pre-processing work is needed. The PyG will
automatically download raw data if you do not have the raw data.