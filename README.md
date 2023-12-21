# CompositionalPremiseRetrieval


### Requirements
- torch
- sentence-transformers
- info-nce-pytorch
- miosqp

See `environment.yml` for more details

### Dataset

The raw data comes from the *EntailmentBank* dataset created by Dalvi et al. (https://arxiv.org/abs/2104.08661, https://allenai.org/data/entailmentbank), which contains natural language Q&A problems. We processed and extracted the corresponding tree structure out of each problem to build our training samples.

### Methods

- `checkpoints/` and `data/` are not shown in this repo
- `result/` contains the qualitative and quantitative evaluation results
- `dataloader/` contains scripts for data processing and inspection
- `evaluation/` contains scripts that evaluate the performance metrics

The static baselines, neural baselines, and the compositional methods each differs in data processing and performance evaluation.

- `train*.py` contains training scripts for baselines and compositional models
- `index.py` creates indices out of a neural model in order to perform retrieval tasks
- `eval_vector_norm.py` plots a histogram of model outputs
- `util/` contains the **compositional losses** and utilities specific for training compositional models. Note that in order to run compositional training, the SentenceTransformer library needs to be modified by using a function in sbert_mod.py
- `miosqp/`contains an approximate DP solver
