"""Script to compute the scores of the LightGCN model.

args:
    model_directory (str): Path to the saved model. (default: RecSys/LightGCN/config/res/default_config/)
"""
import os
import sys
import torch
import numpy as np
from RecSys.nn.models.LightGCN import LightGCN_simple
from RecSys.utils.config import get_config, load_everything_from_exp, Experiment


# Device
device = "cpu"

try:
    PATH = sys.argv[1]
    RES_PATH = os.path.join(PATH, "results.yaml")
    exp = get_config(RES_PATH)
    exp = Experiment(exp["Config"])
    MODEL_PATH = os.path.join(PATH, "trained_models", "best_val_Rec@25_e.pt")
    if not os.path.exists(MODEL_PATH):
        MODEL_PATH = os.path.join(PATH, "trained_models", "best_val_Rec@25_ne.pt")
    if not os.path.exists(MODEL_PATH):
        raise ValueError("Model not found")
except IndexError:
    print("Usage: python compute_lightgcn_scores.py <model_dir_path>")
    sys.exit(1)

print("#"*20)
print(exp)
print("#"*20)

# Load data
datas, model = load_everything_from_exp(exp, device, test=False)
(train_graph, test_graph, train_ds, test_ds) = datas
(model, optimizer, scheduler, loss_fn) = model
assert isinstance(model, LightGCN_simple)
model.load_state_dict(torch.load(MODEL_PATH, device))
model.eval()
model = model.to(device)

# Compute the scores
with torch.no_grad():  # no need to compute gradients
    e0 = model.embedding.get_all(train_graph).to(device)
    e1 = model.layers[0](e0, train_graph.edge_index.to(device))
    num_users = train_graph.num_users
    num_items = train_graph.num_items

    s0 = e0[:num_users] @ e0[num_users:].T
    s01 = e0[:num_users] @ e1[num_users:].T
    s10 = e1[:num_users] @ e0[num_users:].T
    s1 = e1[:num_users] @ e1[num_users:].T

    # scores = torch.stack([s0, s01, s10, s1], dim=0)
    # scores = scores.cpu().numpy()

# Save scores
np.save(f"./{exp['dataset']}/s0.npy", s0.cpu().numpy())
np.save(f"./{exp['dataset']}/s01.npy", s01.cpu().numpy())
np.save(f"./{exp['dataset']}/s10.npy", s10.cpu().numpy())
np.save(f"./{exp['dataset']}/s1.npy", s1.cpu().numpy())
