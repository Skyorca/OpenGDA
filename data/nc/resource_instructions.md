### Instructions to data resource for cross-network node classification task

Folder `\preprocess` provides codes for preprocessing raw data. 

#### Folder \eigen
==[Note: May be removed to other path in the future version]==
Folder `\eigen` stores the pre-processed eigen vectors (e.g.: `brazil-eivec.pt`) and values (e.g.: `brazil-eival.pt`) for each dataset, which are specially prepared 
for 'SpecReg' model to load. The eigen vectors and eigen values are pre-computed and stored to accelerate the training of SpecReg. 
	The code to generate eigenval/eigenvec, i.e., `eigen.py`, is provided in `\preprocess` 

#### Dataset Citation
This dataset originates from ACDNE[1], so please refer to that for more details. It contains 3 graphs: acmv9,dblpv7,citationv1. Each graph is a domain.

Each domain contains one raw data file, namely `acmv9.mat` (acmv9,dblpv7,citationv1), so put it under the path `\data\nc\citation\acmv9\raw`,
then it will be loaded and processed by `CitationDomainData` class defined in `\data\nc_dataloader.py`

About node feature& node label: no extra process should be done.

#### Dataset Twitch
This data originates from MUSAE[2], so please refer to that for more details. It contains 6 graphs: DE, ENGB, ES, FR, PTBR and RU. Each graph is a domain.

Each domain contains three raw data files, namely `musae_DE_edges.csv`, `musae_DE_features.json`, `musae_DE_target.csv` (DE, ENGB, ES, FR, PTBR and RU), so put them under the path `\data\nc\twitch\DE\raw`,
then they will be loaded and processed by `TwitchDomainData` class defined in `\data\nc_dataloader.py`

About node feature& node label: no extra process should be done.

#### Dataset Blog
This dataset originates from ACDNE[1], so please refer to that for more details. It contains 2 graphs: blog1, blog2. Each graph is a domain.

Each domain contains one raw data file, namely `blog1.mat` (blog1,blog2), so put it under the path `\data\nc\blog\blog1\raw`,
then it will be loaded and processed by `BlogDomainData` class defined in `\data\nc_dataloader.py`

About node feature& node label: no extra process should be done.

#### Dataset Airport
This dataset originates from GRADE, so please refer to that for more details. It contains 3 graphs: brazil, usa, europe. Each graph is a domain.

Each domain contains two raw data files, namely `brazil-airports.edgelist` and `labels-brazil-airports.txt`, so put them under the path `\data\nc\airport\brazil\raw`,
then they will be loaded and processed by `AirportDomainData` class defined in `\data\nc_dataloader.py`

About node feature& node label: Following GRADE, we use one-hot degree vector as node feature with maximum degree as 7, leading to a 8-dim feat vector.


#### Ref.
[1] Adversarial Deep Network Embedding for Cross-network Node Classification
[2] Multi-scale Attributed Node Embedding