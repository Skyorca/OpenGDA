### OpenGDA: Graph Domain Adaptation Benchmark for Cross-network Learning

#### What is OpenGDA?

OpenGDA is a benchmark which integrates 1) datasets for evaluating diverse cross-network learning tasks and 2) state-of-the-art graph domain adaptation models. 

- **cross-network learning task:**  To alleviate the lack of high-quality labels and the sparse graph structure, reseachers build cross-network learning task by introducing relevent source graphs to transfer labeling and structural knowledge to target graphs. The goal of cross-network learning task is improving task performance on target graphs by transferring knowledge from source graphs.
- **graph domain adaptation:** Researchers improve domain adaptation techinique by taking the properties of structured graph data into account.

#### Why establish OpenGDA?

Currently, there mainly exist two limitations in evaluating graph domain adaptation models. 

- **Task-limitation:**  GDA models are primarily tested for the specific cross-network node classification task, leaving tasks at edge-level and graph-level largely under-explored.
- **Scenario-limitation:** GDA models are primarily tested in limited scenarios, such as social networks or citation networks, lacking validation ofmodel’s capability in richer scenarios. 

As comprehensively assessing models could enhance model practicality in real-world applications, we propose a benchmark, known as OpenGDA. ==It provides abundant pre-processed and unified datasets for different types of tasks (node, edge, graph). They originate from diverse scenarios, covering web information systems, urban systems and natural systems. Furthermore, it integrates state-of-the-art models with standardized and end-to-end pipelines.== Overall, OpenGDA provides a user-friendly, scalable and reproducible benchmark for evaluating graph domain adaptation models.

#### How to use OpenGDA?

##### Requirements

##### WorkFlow

##### Node-level tasks

##### Edge-level tasks

##### Graph-level tasks

