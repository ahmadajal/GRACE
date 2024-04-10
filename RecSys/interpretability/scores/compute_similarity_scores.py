""" Compute similarity scores for all users and items in the test set.

args:
    model_directory (str): Path to the saved model. (default: RecSys/NCF/config/res/default_config/)
"""
import os
import sys
import torch
import numpy as np
from RecSys.NN.models.similarity import Item2ItemCosineSimilarity, User2UserCosineSimilarity
from RecSys.utils.config import get_config, load_everything_from_exp, Experiment

# Device
device = "cuda:1"

try:
    PATH = sys.argv[1]
    RES_PATH = os.path.join(PATH, "results.yaml")
    exp = get_config(RES_PATH, False)
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
datas, model = load_everything_from_exp(exp, device, test=True)
(train_graph, test_graph, train_ds, test_ds) = datas
(model, optimizer, scheduler, loss_fn) = model
assert isinstance(model, Item2ItemCosineSimilarity) or isinstance(model, User2UserCosineSimilarity)
model.load_state_dict(torch.load(MODEL_PATH, device))
model.eval()
model = model.to(device)


with torch.no_grad():
    # Compute scores for all users and items
    scores = model.scores.cpu().numpy()
    scores = scores.astype("float16")

# Save scores
np.save(os.path.join("RecSys/interpretability/scores", f"""{exp.config["name"]}.npy"""), scores)
