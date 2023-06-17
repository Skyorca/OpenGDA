### Instructions to data resource for cross-network link prediction task

Folder `\preprocess` provides codes for preprocessing raw data. 
#### amazon1 Dataset
This dataset originates from Amazon Review dataset provided by GRADE, so please refer to that for more details. But we need to pre-process the raw data, in order
to efficiently use them for training models. In the contrary, GRADE takes a lot of time building data from the very beginning at every running, which is too
time-consuming.

Step1-Find Data: You should get the data from the source code&data provided by GRADE, which lies in the path `\recommendation\data`,
or find a copy in the link we provided.

Step2-Preprocess: Basically we re-organize and rewrite the original code for generating the desired raw data. You should put the `data` folder (downloaded above)
, `amazon-reader` folder and `amazon.py` file in the same directory. Run `amazon.py` to generate two pre-processed files: book_train.mat and book_test.pkl for book domain
(book, movie, cd and music are potential domain names for both overlapping and non-overlapping settings). 

Example: `python amazon.py --data_dir data/nonoverlapping/Book_Movie  --domain book`

Step3: Back to OpenGDA. Assume you have pre-processed 'book' domain in non-overlapping settings, then turn to `\data\lp\amazon1\nonoverlapping\book\raw` and put `book_train.mat` and `book_test.pkl`
here. These pre-processed data will be loaded and processed by `Amazon1DomainData` class defined in `\data\lp_dataloader.py`.

Congratulations! You have completed the most complicated procedure for preparing data.

#### citation-small Dataset
This dataset is provided by UDAGCN, so please refer to that for more details. The original dataset is used for cross-network node classification,
here we use it for link prediction.

Step1-Find Data: You should get the data from the source code&data provided by UDAGCN, which lies in the path `\data`, or find a copy in the link we provided.

Step2-Preprocess: Basically we split training and testing for each domain. Put `data` folder you downloaed above and `citation.py` in the same directory. Run citation.py
to get two files: acm_train.mat and acm_test.pkl for acm domain. (acm dblp)

Step3:Back to OpenGDA. Assume you have pre-processed 'acm' domain , then turn to `\data\lp\citation-small\book\raw` and put `book_train.mat` and `book_test.pkl`
here. These pre-processed data will be loaded and processed by `CitationDomainData` class defined in `\data\lp_dataloader.py`.

#### ppi dataset

[This dataset needs further improvement, and may be not fully available currently]

This dataset is provided by SpecReg, so please refer to that for more details.

Step1: download the original datasets following instructions provided by SpecReg papers. For each species in {human, yeast, mouse, fruit_fly, zebrafish, nematode},
there should be two files. Take human as an example, there should be `9606.protein.links.full.v11.5.txt` and `9606.protein.sequences.v11.5.fa` two files, indicating
protein-protein interactions and protein sequences, seperately.

Step2: Back to OpenGDA. Put the two files you have downloaded above into folder `\data\lp\ppi\human\raw`. These data will be loaded and processed by `PPIDomainData` class defined in `\data\lp_dataloader.py`.

