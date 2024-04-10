"""Script to compute the scores of the interpretable models.

args:
    model_directory (str): Path to the saved model. (default: RecSys/LightGCN/config/res/default_config/)
"""
import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from RecSys.GNN.models.LightGCN import LightGCN_simple
from RecSys.interpretability.models.interpretable_gnn import TopKUsers_LightGCN, TopKItems_LightGCN
from RecSys.utils.config import get_config, load_everything_from_exp, Experiment


# Device
device = "cuda:1"

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
datas, model = load_everything_from_exp(exp, device, test=True)
(train_graph, test_graph, train_ds, test_ds) = datas
(gnn, optimizer, scheduler, loss_fn) = model
assert isinstance(gnn, LightGCN_simple)
gnn.load_state_dict(torch.load(MODEL_PATH, device))
gnn.eval()
gnn = gnn.to(device)


k = int(sys.argv[2])

# model = TopKUsers_LightGCN(gnn, train_graph, k=k)
model = TopKItems_LightGCN(gnn, train_graph, k=k)
num_users, num_items = train_graph.num_users, train_graph.num_items

all_scores = np.zeros((num_users, num_items))

# Compute the scores
bs = 1
with torch.no_grad():  # no need to compute gradients
    item_id = torch.arange(num_items).to(device) + num_users
    for u in tqdm(range(0, num_users, bs)):
        user_ids = torch.arange(u, min(u+bs, num_users), dtype=torch.long, device=device).tile((num_items, 1)).T.flatten()
        item_ids = item_id.repeat(min(bs, num_users-u))
        scores_u = model(train_graph, user_ids, item_ids)
        for uu, p in enumerate(torch.split(scores_u, num_items)):
            all_scores[u+uu] = p.cpu().numpy()

# Save the scores
np.save(f"RecSys/interpretability/scores/top{k}items.npy", all_scores)
