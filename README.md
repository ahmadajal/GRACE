# GRACE
In this work we present GRACE, a post-hoc interpretability technique specifically designed for single layer GNNs. GRACE is highly scalable and offers rapid processing capabilities for large graph applications. We show both analytically and empirically the relationship between GRACE and conventional, interpretable user-to-user and item-to-item collaborative filtering strategies

This directory contains the files used when performing experiments about GNNs interpretability.

## Models
We benchmarked 3 models: GNNExplainer, Sensitivity Analysis and GRACE (ours)

The three are implemented in the `__models/__` directory. The way they are implemented, models explain a cerain prediction (a user-item interactions) by giving the existing links in the user-item graph that explain the prediciton the most. And they return a score that is interpretable.


## Metrics
We used two metrics to compare the models, i.e., the COMP score and the Model Fidelity. The COMP score metric for each model is implemented in the `__COMP_SCORE__` directory. As for GNNExplainer needs times to explain predicitons, only the first 100 users and 1000 items per user were used. 

## Datasets
We run our experiment on four datasets: __Gowalla__, __Yelp2018__, __Amazon-Book__ and __Tomplay__

|     Dataset   |   # Users  | # Items| # Interactions | Density |
|---------------|------------|--------|----------------|---------|
|    Gowalla    | 29,858     | 40,981 | 1,027,370      | 0.00084 |
|   Yelp2018    | 31,668     | 38,018 | 1,561,406      | 0.00130 |
|  Amazon-Book  | 52,643     | 91,599 | 2,984,108      | 0.00062 |
|  Tomplay      | 35,028     | 33,397 | 7,510,000      | 0.00642 |


Results of the paper can be found in the notebook __interpretable_gnn_figures.ipynb__.

## Other folders in the repository

### RecSys
This folder contains the implementation of the lightGCN model and the user-user and item-item collaborative filtering models used in this paper. It also contains config files for training and some utility funcitons.

### scores
This folder contains the implemenation of the user and items contributions for the LighGCN and the user2user and item2item CF methods.