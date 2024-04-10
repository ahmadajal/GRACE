import sys
import numpy as np
import torch
import torch.utils.data
from torch_geometric.seed import seed_everything

from RecSys.nn.training import train_fn, test_fn
from RecSys.utils.config import Experiment, get_config, get_default_config, load_everything_from_exp_cold, get_results

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

# Hyperparameters
EPOCH_PER_VAL = 10
EPOCH_PER_LR_DECAY = 20

EPOCHS = exp["epochs"]
BATCH_SIZE = exp["batch_size"]

# Device
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
val_device = "cpu"

# Load data
datas, model = load_everything_from_exp_cold(exp, device)
(cold_data, train_graph, val_graph, train_ds, val_ds) = datas
(model, optimizer, scheduler, loss_fn) = model
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
train_user_id = np.load("data/users_split/simple/users_train.npy")
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

# Training routine
for epoch in range(1, EPOCHS+1, EPOCH_PER_VAL):
    train_fn(
        model=model.to(device),
        graph=train_graph.to(device),
        optimizer=optimizer,
        scheduler=scheduler,
        epoch_per_lr_decay=EPOCH_PER_LR_DECAY,
        dataloader=train_dl,
        loss_fn=loss_fn,
        epoch_begin=epoch,
        nepochs=EPOCH_PER_VAL,
        metrics=train_metrics,  # type: ignore
        exp=exp
    )

    test_fn(
        model=model.to(device),
        graph=train_graph.to(device),
        test_ds=val_ds,
        device=val_device,
        epoch=epoch+EPOCH_PER_VAL-1,
        metrics=val_metrics,
        exp=exp,
        user_subset=train_user_id
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
