This directory contains the files used when performing experiments about GNNs interpretability.

## Models
We benchmarked 3 models: GNNExplainer, Sensitivity Analysis and GRACE (ours)

The three are implemented in the __models/__ directory. The way they are implemented, models explain a cerain prediction (a user-item interactions) by giving the existing links in the user-item graph that explain the prediciton the most. And they return a score that is interpretable.


## Metrics
Metrics were computed using the files in this folder. As for GNNExplainer needs times to explain predicitons, only the first 100 users and 1000 items per user were used. The two metrics computed were the COMP score and the Model Fidelity.

Results can be found in the notebook __interpretable_gnn_figures.ipynb__
