"""Test script

args:
    model_directory (str): Path to the saved model. (default: RecSys/config/res/default_config/)
"""
import os
import sys
import pickle
import torch
from RecSys.metrics import Accuracy_K, Recall_K, NDCG_K, NDCG_K_first_bootstrap, HITS_K, HITS_K_first, HITS_K_first_bootstrap
from RecSys.utils.config import get_config, load_everything_from_exp, Experiment
from RecSys.nn.training import test_fn

# Device
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

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
    print("Usage: python test.py <model_dir_path>")
    sys.exit(1)

print("#"*20)
print(exp)
print("#"*20)

# Load data
datas, model = load_everything_from_exp(exp, DEVICE, test=True)
(train_graph, test_graph, train_ds, test_ds) = datas
(model, optimizer, scheduler, loss_fn) = model
model.load_state_dict(torch.load(MODEL_PATH, DEVICE))
model.eval()
model = model.to(DEVICE)

test_ds.compute_adj_matrix()
# test_ds.compute_first_test_interactions()
test_ds.train_adj_matrix = test_ds.train_adj_matrix.to(DEVICE)
test_ds.val_adj_matrix = test_ds.val_adj_matrix.to(DEVICE)


# Metrics
metrics = [
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

    # Accuracy_K(10, exclude=False),
    Accuracy_K(25, exclude=False),
    # Accuracy_K(100, exclude=False),

    # Recall_K(10, exclude=False),
    Recall_K(25, exclude=False),
    # Recall_K(100, exclude=False),

    # MRR_K(10, exclude=False),
    # MRR_K(25, exclude=False),
    # MRR_K(100, exclude=False),

    # NDCG_K(10, exclude=False),
    # NDCG_K(10, exclude=False),
    # NDCG_K_first_bootstrap(10, num_rep=30),
    # NDCG_K(100, exclude=False),

    # HITS_K(10, exclude=False),
    # HITS_K_first(10),
    # HITS_K_first_bootstrap(10, num_rep=30),
    # HITS_K(25, exclude=False),
    # HITS_K(100, exclude=False),

    # HITS_K_first_bootstrap(10, num_rep=30),
    # HITS_K_first(10),
]

test_fn(
    model=model,
    graph=train_graph.to(DEVICE),
    test_ds=test_ds,
    device=DEVICE,
    epoch=1,
    metrics=metrics,
    exp=None
)

# Save metrics
with open(os.path.join(PATH, "test_results.pkl"), "wb") as f:
    pickle.dump(metrics, f, protocol=2)
