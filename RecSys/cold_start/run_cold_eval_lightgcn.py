import os
import sys
import numpy as np
import torch
import torch.utils.data
from torch_geometric.seed import seed_everything

from RecSys.nn.training import test_fn
from RecSys.cold_start.evaluate import final_evaluation
from RecSys.utils.config import Experiment, get_config, get_default_config, load_everything_from_exp_cold

import RecSys.cold_start.models.lightgcn as cold_start_lightgcn
from RecSys.nn.models.LightGCN import LightGCN_simple

from RecSys.metrics import LossMetric, Accuracy_K, Recall_K


# Import config
if len(sys.argv) == 1:
    exp = Experiment(get_default_config())
else:
    try:
        if len(sys.argv) == 3:
            exp = Experiment(get_config(sys.argv[1]), res_path=sys.argv[2])
        else:
            exp = Experiment(get_config(sys.argv[1]))
    except Exception:
        raise ValueError("Argument must be a path to a config")

# Seed
if "seed" not in exp.config:
    seed = torch.randint(0, 1 << 32, (1,)).item()
else:
    seed = exp.config["seed"]
seed_everything(seed)  # type: ignore
exp.config["seed"] = seed


print("#"*8, "Experiment", "#"*8)
print(exp)
print("#"*8, "##########", "#"*8)

# Device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
val_device = "cpu"  # device

# Load data
datas, model = load_everything_from_exp_cold(exp, device)
(cold_data, train_graph, val_graph, train_ds, val_ds) = datas
(model, optimizer, scheduler, loss_fn) = model
train_user_id = np.load("data/tomplay/users_split/users_train.npy")
train_user_id = torch.tensor(train_user_id, dtype=torch.long).to(val_device)
assert isinstance(model, LightGCN_simple)

# Compute matrix needed for metrics
val_ds.compute_adj_matrix()
# val_ds.compute_first_test_interactions()

# Metrics
train_metrics = [
    LossMetric().to(device),
]
val_metrics = [
    Accuracy_K(25, exclude=False),
    Recall_K(25, exclude=False),
]
for m in val_metrics:
    m.name = "val_" + m.name

# Load best_model
model.load_state_dict(torch.load(os.path.join(exp.model_path, "best_val_Rec@25_ne.pt"), map_location=val_device))
test_fn(
    model=model.to(device),
    graph=train_graph.to(device),
    test_ds=val_ds,
    device=val_device,
    epoch=1,
    metrics=val_metrics,
    exp=None,
    user_subset=train_user_id
)

models_class = [
    # Baselines
    # cold_start_baselines.RandomPrediction,
    # cold_start_baselines.MostVisitedItems,
    # cold_start_baselines.MostVisitedItemsByInstrument,
    # cold_start_baselines.MostVisitedItemsByInstrumentLevel,

    # LightGCN wo first embedding
    # cold_start_lightgcn.LightGCN_wo_first_embedding,
    # cold_start_lightgcn.LightGCN_wo_first_embedding_merged,
    # cold_start_lightgcn.LightGCN_wo_first_embedding_keep_normalization,
    cold_start_lightgcn.LightGCN_wo_first_embedding_no_normalization,

    # LightGCN with first embedding mean instr
    # cold_start_lightgcn.LightGCN_first_embedding_mean_instr,
    # cold_start_lightgcn.LightGCN_first_embedding_mean_instr_merged,
    # cold_start_lightgcn.LightGCN_first_embedding_mean_instr_keep_normalization,

    # LightGCN with first embedding mean instr level
    # cold_start_lightgcn.LightGCN_first_embedding_mean_instr_level,
    # cold_start_lightgcn.LightGCN_first_embedding_mean_instr_level_merged,
    # cold_start_lightgcn.LightGCN_first_embedding_mean_instr_level_keep_normalization,
    cold_start_lightgcn.LightGCN_first_embedding_mean_instr_level_no_normalization

    # cold_start_lightgcn.LightGCN_first_embedding_mean_instr_level_test
]
models_name = [
    # "RandomPrediction",
    # "MostVisitedItems",
    # "MostVisitedItemsByInstrument",
    # "MostVisitedItemsByInstrumentLevel",

    # "LightGCN_wo_first_embedding",
    # "LightGCN_wo_first_embedding_merged",
    # "LightGCN_wo_first_embedding_keep_normalization",
    "LightGCN_wo_first_embedding_no_normalization",

    # "LightGCN_first_embedding_mean_instr",
    # "LightGCN_first_embedding_mean_instr_merged",
    # "LightGCN_first_embedding_mean_instr_keep_normalization"

    # "LightGCN_first_embedding_mean_instr_level",
    # "LightGCN_first_embedding_mean_instr_level_merged",
    # "LightGCN_first_embedding_mean_instr_level_keep_normalization"
    "LightGCN_first_embedding_mean_instr_level_no_normalization",

    # "LightGCN_test"
]
models_args = [
    # [],
    # [],
    # [],
    # [],

    [model],
    [model],
    [model],

    # [model],
    # [model],
    # [model],

    [model],
    [model],
    [model],
]

for m_class, m_name, m_args in zip(models_class, models_name, models_args):
    print(m_name)
    m = m_class(*m_args)
    final_evaluation(
        model=m.to(val_device),
        data=cold_data.to(val_device),
        metrics=val_metrics,
        nb_inter_range=[1, 5, 10, 15, 20, 30, 40, 50, 60, 100],  # range(1, 101),
        res_path=f"RecSys/cold_start/res_512_last_25/{m_name}.pkl",
        device=val_device,
    )
