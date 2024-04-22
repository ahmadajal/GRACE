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

## Example: Running GRACE for a user-item pair

You can use the following code snippet to run GRACE on the model prediction for a user-item pair. The explainability is in terms of the top-k most influential users and items for the model prediction. Additionally, the `one_forward` method returns the score of the model when the top users and items are present and when they are not present.

```
import os
import torch

from RecSys.utils.config import get_config, load_everything_from_exp, Experiment
from RecSys.nn.models.LightGCN import LightGCN_simple
from models.grace import GRACE

# Device
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

PATH = <PATH-to-the-trained-model-folder>
RES_PATH = os.path.join(PATH, "results.yaml")
exp = get_config(RES_PATH)
exp = Experiment(exp["Config"])
MODEL_PATH = os.path.join(PATH, "trained_models", "best_val_Rec@25_e.pt")
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = os.path.join(PATH, "trained_models", "best_val_Rec@25_ne.pt")

# Load data
datas, model = load_everything_from_exp(exp, DEVICE, test=False)
(train_graph, test_graph, train_ds, test_ds) = datas
(model, optimizer, scheduler, loss_fn) = model
assert isinstance(model, LightGCN_simple), "Model must be LightGCN_simple"
assert model.nb_layers == 1, "Model must be LightGCN with 1 layer"
model.load_state_dict(torch.load(MODEL_PATH, DEVICE))
model.eval()
model = model.to(DEVICE)

grace_model = GRACE(model, train_graph, 10, 10)

# You can select any random user-item pair, preferebly a positive pair from the train/test data.
u = 0
i = 1
new_y, y_wo_top, top_users, top_items = grace_model.one_forward(train_graph, u, i+train_graph.num_users)
```