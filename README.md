### :star2:OpenGDA: Graph Domain Adaptation Benchmark for Cross-network Learning :star2:

:seedling: :seedling: Currently we are building this project with more models and datasets.:seedling::seedling:  

#### What is OpenGDA?

OpenGDA is a benchmark which integrates 1) datasets for evaluating diverse cross-network learning tasks and 2) state-of-the-art graph domain adaptation models. 

- **cross-network learning task:**  To alleviate the lack of high-quality labels and the sparse graph structure, reseachers build cross-network learning task by introducing relevent source graphs to transfer labeling and structural knowledge to target graphs. The goal of cross-network learning task is improving task performance on target graphs by transferring knowledge from source graphs.
- **graph domain adaptation:** Researchers improve domain adaptation techinique by taking the properties of structured graph data into account.

#### Why establish OpenGDA?

Currently, there mainly exist two limitations in evaluating graph domain adaptation models. 

- **Task-limitation:**  GDA models are primarily tested for the specific cross-network node classification task, leaving tasks at edge-level and graph-level largely under-explored.
- **Scenario-limitation:** GDA models are primarily tested in limited scenarios, such as social networks or citation networks, lacking validation ofmodel’s capability in richer scenarios. 

As comprehensively assessing models could enhance model practicality in real-world applications, we propose a benchmark, known as OpenGDA. **It provides abundant pre-processed and unified datasets for different types of tasks (node, edge, graph). They originate from diverse scenarios, covering web information systems, urban systems and natural systems. Furthermore, it integrates state-of-the-art models with standardized and end-to-end pipelines.** Overall, OpenGDA provides a user-friendly, scalable and reproducible benchmark for evaluating graph domain adaptation models.

#### How to use OpenGDA?

##### Requirements

###### About gpu environment

We have tested under multiple environments and here we list some of them:

- cuda 10.2, pytorch 1.8.1+cu102, torch-geometric 2.0.2
- cuda 11.3, pytorch1.10.2+cu113, torch-geometric 2.0.3

###### packages

We mainly require `pytorch` as neural network framework and `torch-geometric` as GNN framework.

Other related packages you can find in the `requirements.txt`

##### WorkFlow

##### Node-level tasks

:fireworks:Currently we provide **Airport** dataset as it is relatively small, other datasets please refer to their original studies, and we will provide a copy on cloud drive asap.

:bangbang: For instructions to data resources, please refer to [node-level](https://github.com/Skyorca/OpenGDA/blob/master/data/nc/resource_instructions.md)

:sun_with_face:To run a model, like ASN, you need to change your path to `\model\ASN`, and run it with:

`python start_nc.py --dataset_name airport --src_name usa  --tgt_name brazil --cuda 0`

For command line args, please refer to `start_nc.py` for more details.

##### Edge-level tasks

:bangbang: For instructions to data resources, please refer to [edge-level](https://github.com/Skyorca/OpenGDA/blob/master/data/lp/resource%20instructions.md)

##### Graph-level tasks

:fireworks:Currently we provide **LetterHigh-LetterLow**  dataset as it is relatively small, other datasets please refer to their original studies, and we will provide a copy on cloud drive asap.

:bangbang: For instructions to data resources, please refer to [graph-level](https://github.com/Skyorca/OpenGDA/blob/master/data/gc/resource%20instructions.md)

:sun_with_face:To run a model, like GRADE, you need to change your path to `\model\GRADE`, and run it with:

`python start_gc.py --dataset_name TUDataset --src_name Letter-high  --tgt_name Letter-low --cuda 0`

For command line args, please refer to `start_gc.py` for more details.