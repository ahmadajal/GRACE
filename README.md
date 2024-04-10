# GRACE
In this work we present GRACE, a post-hoc interpretability technique specifically designed for single layer GNNs. GRACE is highly scalable and offers rapid processing capabilities for large graph applications. We show both analytically and empirically the relationship between GRACE and conventional, interpretable user-to-user and item-to-item collaborative filtering strategies




## Datasets
We run our experiment on four datasets: __Gowalla__, __Yelp2018__, __Amazon-Book__ and __Tomplay__

|     Dataset   |   # Users  | # Items| # Interactions | Density |
|---------------|------------|--------|----------------|---------|
|    Gowalla    | 29,858     | 40,981 | 1,027,370      | 0.00084 |
|   Yelp2018    | 31,668     | 38,018 | 1,561,406      | 0.00130 |
|  Amazon-Book  | 52,643     | 91,599 | 2,984,108      | 0.00062 |
|  Tomplay      | 35,028     | 33,397 | 7,510,000      | 0.00642 |

In particular __Gowalla__, __Yelp2018__, __Amazon-Book__ are three widely used dataset for the recommendation task used for example in the NGCF and LightGCN paper.



## The Recommendation Task

### Comparison of different recommendation approaches.
Each model is trained on the same training set and finetuned with the same validation set. The shown results are on the test set. (70% training, 10% validation, and 20% test set). The following table summarizes the results for each recommender model and data processing approach:

|            model           | recall@25 | precision@25 | HR@10   | NDCG@10 |
|:--------------------------:|:---------:|:------------:|:-------:|:-------:|
|    matrix factorization    |    0.14   |     0.07     | 0.93    | 0.74    |
|  Factorization Machine     |    0.337  |     0.127    | 0.967   | 0.864   |
|  Neural Factorization Machine| 0.332   |    0.130     | 0.943   | 0.847   |
| Neural Collaborative Filtering (NCF) | 0.33 | 0.13    | 0.97    | 0.87    |
| TiSASRec                   |    0.226  |     0.152    | 0.954   | 0.863   |
| Wide & Deep                |    0.319  |     0.131    | 0.959   | 0.861   |
| Deep & Cross               |    0.333  |     0.134    | 0.964   | 0.869   |
| GAT                        |    0.332  |     0.126    | 0.999   | 0.85    |
|            NGCF            |    0.334  |     0.135    | 0.973   | 0.863   |
|          LightGCN          |    0.378  |     0.165    |         |         |


### Train your model. 

A model can be trained using the following command:
```bash
python RecSys/nn/run_train.py /path/to/config/file.yaml
```

The config files gather all informations about the models, its hyperparameters, the training parameters and the dataset on wich the model should be trained.

The following is a description of what can be put in the config.

Some configs are given in the folder: ```RecSys/config/```

#### 1. Experiment
- The key `name` is the name of the experiment. Metrics over training and model parameters will be saved under __RecSys/config/res/`name`/__
- The key `dataset` is to choose between the available datasets in the folder __data__.
In order to be consider, a dataset must have at least three files: __data/`dataset`/split/interactions_train.csv__, __data/`dataset`/split/interactions_val.csv__ and __data/`dataset`/split/interactions_test.csv__. Each of these files is a csv files having at least two columns representing interactions:
    - `u`: user id
    - `i`: item id
    - `t`: (optional) timestamp
- The optional key `seed` is an int that cand be useful for reproducing results.

#### 2. Choose your model
- The key `model` allows to choose between the different implemented model:
    - Matrix Factorization: __MatrixFactorization__
    - User to user collaborative filtering using Pearson correlation: __User2UserCosineSimilarity__ (name is not well chosen as the Pearson correlation is used)
    - Item to item collaborative filtering using Pearson correlation: __Item2ItemCosineSimilarity__ (idem)
    - NCF: __NCF__
    - NGCF: __NGCF__
    - LightGCN: __LightGCN_simple__

- For models that uses an embedding matrix to represent users and items, the key `embedding_type` allows to introduce entity resolution.
    - __IdEmbedding__: Each user/items is mapped to a different embedding.
    - __IdEmbeddingPlusNameEmbedding__: (only for the Tomplay dataset) The encoding of song title is added to item embeddings.
    - __UserFeaturesPlusNameEmbedding__: (only for the Tomplay dataset) Users level and instrument are encoded and summed. The encoding of song title is added to item embeddings.
    - __UserFeaturesAndIdEmbeddingPlusNameEmbedding__: (only for the Tomplay dataset) Users level, instrument and id are encoded and summed. The encoding of song title is added to item embeddings.

- The key `embedding_dim` allows to choose the embedding dimension.
- For model that suits this config, `num_layers` allows to choose the number of layers. 

#### 3. Graph preprocessing
We implemented different preprocessing of the user-item interaction graph that had significant effect on the models performance.

- The key `add_edges`, if used, connects items with similar features by ading a node that is connected to them.
    - __add_nothing__: By default no connection is added.
    - __connect_same_musics__: Add a between each group of items having the same features. The features that determine the merging are chosen using the key `merge_on` (list of strings).

- The key `edge_weights` turns the user-item interaction graph into a weighted graph.
    - __ones__: All interactions have the same weights: the weight of (u,i) is 0 if u have never interacted with i, 1 otherwise.
    - __num_interactions__: Weight of edge (u,i) is the number of times u interacted with i.
    - __exp_time_since_last_inter__: Weights decrease exponentially as the timestamp difference with last user interactions increases. The key `edge_weighting_gamma` controls how fast the weights decrease.
    - __exp_time__: Weights decrease exponentially as the timestamp difference with last overall interaction increases. The key `edge_weighting_gamma` controls how fast the weights decrease.

#### 4. Training parameters
- The key `loss` can be __BPR__ as for BPR loss or __CE__ as for cross entropy loss.
- The keys `lr`, `epochs`, `batch_size` are the standard training parameters.


### Test your model

After being trained with the previous instructions, a model can simply be tested using the following command:
```bash
python RecSys/nn/run_eval.py /path/to/results/folder/
```

You can perform a ttest between the metrics of two model with the command:
```bash
python ttest.py /path/to/results/of/model1/ /path/to/results/of/model2/
```


### Interpretability

We implemented three different method for explaining GNNs:
 - SensitivityAnalysis : it measures the impact of a particular change in the input on the prediction. We used the local gradient of the model with respect to the nodes features to quantify sensitivity. 
 - GNNExplainer : it is an approach designed to explain GNN-based models. Its primary objective is to identify crucial graph structures by maximizing the mutual information between the GNN’s predictions and the distribution of subgraphs derived from the input.
 - GRACE (our work) : it leverages LightGCN uses a linear aggregation function and the (linear) dot product to compute user-item pair scores.


 The model implementations can be found in `RecSYS/interpretability/models`

 Then scripts under `RecSys/interpretability` are used to compare the different methods.
