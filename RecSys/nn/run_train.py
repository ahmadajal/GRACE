"""Training script for LightGCN

args:
    config_path (str): Path to the config file. (default: RecSys/config/default_config.yaml)
    res_path (str): Path to the directory to store results. (default: RecSys/config/res/)
"""
import sys
from time import time
import torch
import torch.utils.data
from torch_geometric.seed import seed_everything

from RecSys.nn.training import train_fn, test_fn
from RecSys.utils.config import Experiment, get_config, get_default_config, load_everything_from_exp, get_results

from RecSys.metrics import HITS_K, MRR_K, LossMetric, Accuracy_K, Recall_K, NDCG_K


# Import config
if len(sys.argv) == 1:
    exp = Experiment(get_default_config())
else:
    try:
        if len(sys.argv) == 3:
            exp = Experiment(get_config(sys.argv[1]), res_path=sys.argv[2])
        else:
            exp = Experiment(get_config(sys.argv[1]))
    except Exception as exc:
        raise ValueError("Argument must be a path to a config") from exc

# Seed
if "seed" not in exp.config:
    seed = int(torch.randint(0, 1 << 32, (1,)).item())
else:
    seed = int(exp.config["seed"])
seed_everything(seed=seed)
exp.config["seed"] = seed


print("#"*8, "Experiment", "#"*8)
print(exp)
print("#"*8, "##########", "#"*8)

# Hyperparameters
EPOCH_PER_VAL = 10
EPOCH_PER_LR_DECAY = 20

EPOCHS = exp["epochs"]
BATCH_SIZE = exp["batch_size"]

# Device
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
VAL_DEVICE = "cpu"
print("Working with device:", DEVICE)

# Load data
datas, model = load_everything_from_exp(exp, DEVICE)
(graph, val_graph, train_ds, val_ds) = datas
(model, optimizer, scheduler, loss_fn) = model
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

print("Training graph has:")
print(f"\t{graph.directed_edge_index.shape[1]} edges")
print(f"\t{graph.num_users} users")
print(f"\t{graph.num_items} items")

# Compute matrix needed for metrics
val_ds.compute_adj_matrix()
# val_ds.compute_first_test_interactions()

# Metrics
train_metrics = [
    LossMetric().to(DEVICE),
]
val_metrics = [
    # Accuracy_K(10, exclude=True),
    # Accuracy_K(25, exclude=True),
    # Accuracy_K(100, exclude=True),

    # Recall_K(10, exclude=True),
    # Recall_K(25, exclude=True),
    # Recall_K(100, exclude=True),

    # MRR_K(10, exclude=True),
    # MRR_K(25, exclude=True),
    # MRR_K(100, exclude=True),

    # NDCG_K(10, exclude=True),
    # NDCG_K(25, exclude=True),
    # NDCG_K(100, exclude=True),

    # HITS_K(10, exclude=True),
    # HITS_K(25, exclude=True),
    # HITS_K(100, exclude=True),

    Accuracy_K(10, exclude=False),
    Accuracy_K(25, exclude=False),
    # Accuracy_K(100, exclude=False),

    Recall_K(10, exclude=False),
    Recall_K(25, exclude=False),
    # Recall_K(100, exclude=False),

    # MRR_K(10, exclude=False),
    # MRR_K(25, exclude=False),
    # MRR_K(100, exclude=False),

    NDCG_K(10, exclude=False),
    NDCG_K(25, exclude=False),
    # NDCG_K(100, exclude=False),

    HITS_K(10, exclude=False),
    HITS_K(25, exclude=False),
    # HITS_K(100, exclude=False),
]
for m in val_metrics:
    m.name = "val_" + m.name

# Training routine
for epoch in range(1, EPOCHS+1, EPOCH_PER_VAL):
    top = time()
    train_fn(
        model=model,
        graph=graph,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch_per_lr_decay=EPOCH_PER_LR_DECAY,
        dataloader=train_dl,
        loss_fn=loss_fn,
        epoch_begin=epoch,
        nepochs=EPOCH_PER_VAL,
        metrics=train_metrics,
        exp=exp
    )
    print(f"Time for {EPOCH_PER_LR_DECAY} epochs: {time()-top:.2f} seconds")

    test_fn(
        model=model,
        graph=graph,
        test_ds=val_ds,
        device=VAL_DEVICE,
        epoch=epoch+EPOCH_PER_VAL-1,
        metrics=val_metrics,
        exp=exp
    )

    exp.save_model(model)

    # Early stopping
    if "early_stopping" in exp.config:
        early_stopping = exp["early_stopping"]
        results = get_results(exp)
        if len(results[early_stopping][0]) > 1:
            if results[early_stopping][1][-1] < results[early_stopping][1][-2]:
                print("Early stopping")
                exit()
