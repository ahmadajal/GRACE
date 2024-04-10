""" Compute adj matrix.

args:
    model_directory (str): Path to the saved model. (default: RecSys/NCF/config/res/default_config/)
"""
import os
import sys
import numpy as np
import dask.dataframe as dd
from tqdm import tqdm
from itertools import product
from RecSys.utils.config import get_config, load_everything_from_exp, Experiment


# Device
device = "cuda:1"

try:
    PATH = sys.argv[1]
    RES_PATH = os.path.join(PATH, "results.yaml")
    exp = get_config(RES_PATH, False)
    exp = Experiment(exp["Config"])
except IndexError:
    print("Usage: python test.py <model_dir_path>")
    sys.exit(1)

print("#"*20)
print(exp)
print("#"*20)

# Load data
datas, model = load_everything_from_exp(exp, device, test=True)
(train_graph, test_graph, train_ds, test_ds) = datas
(model, optimizer, scheduler, loss_fn) = model
test_ds.compute_adj_matrix()
num_users = train_graph.num_users
num_items = train_graph.num_items
del model, optimizer, scheduler, loss_fn, train_graph, test_graph, train_ds
val_adj_matrix = test_ds.val_adj_matrix.cpu().numpy()
val_adj_matrix = val_adj_matrix.astype(np.float32)

np.save("RecSys/interpretability/scores/test_adj_matrix.npy", val_adj_matrix)
