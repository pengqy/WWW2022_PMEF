The source code and data for the paper: **Towards a Multi-view Attentive Matching for Personalized Expert Finding**.



## Requirements

- python >=3.6
- torch == 1.7.0
- numpy == 1.19.5
- tokenizers == 0.8.1rc2
- sklearn == 0.20.0
- Cuda Version == 11.1
- fire



## Basic

The introduction of the dirs and files:

- `main.py`: the entrance of the project
- `utils.py`: the ranking metrics for evaluating the model performance
- `data/`: the dir to save the preprocessed data (History dataset)
- `config/`: the parameter config 
- `models/`: the model architecture
- `checkpoints/`: the dir for saving the best model on validation dataset



## Usage

We provide the code and a dataset `History` for evaluation.

Run the `main.py` for training, validation and testing; the script is:

```sh
python3 main.py run
```

A sample output:

```
*************************************************
user config:
dataset => History
a_num => 471
q_num => 4904
tag_num => 811
*************************************************
train data: 79060; test data: 9440; dev data: 9580
start training....
2022-01-27 08:16:04  Epoch 0: train data: loss:0.1761.
dev data results
mean_mrr: 0.5248; P@1: 0.2985; P@3: 0.7349; ndcg@10: 0.6189;
2022-01-27 08:29:59  Epoch 1: train data: loss:0.1654.
dev data results
mean_mrr: 0.6193; P@1: 0.4384; P@3: 0.7808; ndcg@10: 0.6901;
2022-01-27 08:43:57  Epoch 2: train data: loss:0.1552.
dev data results
mean_mrr: 0.5824; P@1: 0.4071; P@3: 0.6952; ndcg@10: 0.6627;
2022-01-27 08:57:52  Epoch 3: train data: loss:0.1516.
dev data results
mean_mrr: 0.6185; P@1: 0.4447; P@3: 0.7495; ndcg@10: 0.6928;
2022-01-27 09:11:46  Epoch 4: train data: loss:0.1510.
dev data results
mean_mrr: 0.6112; P@1: 0.4384; P@3: 0.7349; ndcg@10: 0.6841;
****************************************************************************************************
test data results
mean_mrr: 0.5578; P@1: 0.3677; P@3: 0.7151; ndcg@10: 0.6401;
****************************************************************************************************
```
