""" Script to compute the COMP score for a given model and dataset. """
import os
import sys
import pickle
import torch
import numpy as np
from tqdm import tqdm
from RecSys.utils.config import get_config, load_everything_from_exp, Experiment
from RecSys.nn.models.LightGCN import LightGCN_simple
from models.grace import GRACE, GRACEAbsolute
from models.gnnexplainer import GNNExplainer, sigmoid
from models.sensitivity_analysis import SensitivityAnalysis


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
    print("Usage: python compute_comp.py <model_dir_path>")
    sys.exit(1)

print("#"*20)
print(exp)
print("#"*20)

# Load data
datas, model = load_everything_from_exp(exp, DEVICE, test=False)
(train_graph, test_graph, train_ds, test_ds) = datas
(model, optimizer, scheduler, loss_fn) = model
assert isinstance(model, LightGCN_simple), "Model must be LightGCN_simple"
assert model.nb_layers == 1, "Model must be LightGCN with 1 layer"
model.load_state_dict(torch.load(MODEL_PATH, DEVICE))
model.eval()
model = model.to(DEVICE)

test_ds.compute_adj_matrix()


NB_USER = 100
NB_ITEM = 300
TOP_KU_RANGE = [25]
TOP_KI_RANGE = [25]


def make_empty_list_mat(num_row, num_col):
    """ Make a matrix filled with empty lists. """
    mat = []
    for _ in range(num_row):
        mat.append([])
        for _ in range(num_col):
            mat[-1].append([])
    return mat


models = {
    "GRACE": GRACE(model, train_graph, 0, 0),
    # "GRACEAbsolute": GRACEAbsolute(model, train_graph, 0, 0),
    "GNNExplainer": GNNExplainer(model, 0, 0),
    "SensitivityAnalysis": SensitivityAnalysis(model, 0, 0)
}

s = np.load(f"/home/ahmad/GRACE/scores/{exp['dataset']}/s0.npy")\
    + np.load(f"/home/ahmad/GRACE/scores/{exp['dataset']}/s01.npy")\
    + np.load(f"/home/ahmad/GRACE/scores/{exp['dataset']}/s10.npy")\
    + np.load(f"/home/ahmad/GRACE/scores/{exp['dataset']}/s1.npy")

# comp = {"GRACE": [], "GRACEAbsolute": [], "GNNExplainer": [], "SensitivityAnalysis": []}
# mf = {"GRACE": np.zeros((4, 4)), "GRACEAbsolute": np.zeros((4, 4)), "GNNExplainer": np.zeros((4, 4)), "SensitivityAnalysis": np.zeros((4, 4))}
# top_users, top_items = sensitivity_analysis_top(train_graph)

recalls = {
    "full_model": [],
    "GRACE": make_empty_list_mat(len(TOP_KU_RANGE), len(TOP_KI_RANGE)),
        # "GRACEAbsolute": make_empty_list_mat(4, 4),
    "GNNExplainer": make_empty_list_mat(len(TOP_KU_RANGE), len(TOP_KI_RANGE)),
    "SensitivityAnalysis": make_empty_list_mat(len(TOP_KU_RANGE), len(TOP_KI_RANGE))
}

precisions = {
    "full_model": [],
    "GRACE": make_empty_list_mat(len(TOP_KU_RANGE), len(TOP_KI_RANGE)),
        # "GRACEAbsolute": make_empty_list_mat(4, 4),
    "GNNExplainer": make_empty_list_mat(len(TOP_KU_RANGE), len(TOP_KI_RANGE)),
    "SensitivityAnalysis": make_empty_list_mat(len(TOP_KU_RANGE), len(TOP_KI_RANGE))
}
np.random.seed(exp['seed'])
selected_users = np.random.choice(train_graph.num_users, NB_USER, replace=False)
for u_ind, u in enumerate(tqdm(selected_users)):
    u_new_scores = {
        "GRACE": make_empty_list_mat(len(TOP_KU_RANGE), len(TOP_KI_RANGE)),
        # "GRACEAbsolute": make_empty_list_mat(4, 4),
    "GNNExplainer": make_empty_list_mat(len(TOP_KU_RANGE), len(TOP_KI_RANGE)),
    "SensitivityAnalysis": make_empty_list_mat(len(TOP_KU_RANGE), len(TOP_KI_RANGE))
    }
    # selected_items = np.concatenate(
    #     (np.argsort(-s[u])[:NB_ITEM], np.argwhere(test_ds.val_adj_matrix[u].numpy()).flatten())
    #     )
    # selected_items = np.unique(selected_items)
    selected_items = np.argsort(-s[u])[:NB_ITEM]
    for i in tqdm(selected_items):
        # comp_at_ui = {"GRACE": [], "GRACEAbsolute": [], "GNNExplainer": [], "SensitivityAnalysis": []}

        for i_top_ku, top_ku in enumerate(TOP_KU_RANGE):
            for i_top_ki, top_ki in enumerate(TOP_KI_RANGE):
                for model in models.values():
                    model.ku = top_ku
                    model.ki = top_ki

                if top_ku == 0 and top_ki == 0:
                    new_y, y_true, top_users, top_items = models["GRACE"].one_forward(train_graph, u, i+train_graph.num_users)
                    new_y = new_y.cpu().item()
                    u_new_scores["GRACE"][i_top_ku][i_top_ki].append(new_y)
                    # u_new_scores["GRACEAbsolute"][i_top_ku][i_top_ki].append(new_y)
                    # u_new_scores["GNNExplainer"][i_top_ku][i_top_ki].append(new_y)
                    # u_new_scores["SensitivityAnalysis"][i_top_ku][i_top_ki].append(new_y)
                    y_true = y_true.cpu().item()
                else:
                    for model_name, model in models.items():
                        new_y, y_wo_top, top_users, top_items = model.one_forward(train_graph, u, i+train_graph.num_users)
                        new_y = new_y.cpu().item()
                        u_new_scores[model_name][i_top_ku][i_top_ki].append(new_y)
                        y_wo_top = y_wo_top.cpu().item()
                        # comp_at_ui[model_name].append(sigmoid(y_true - y_wo_top))

        # for model_name, comp_at_ui_values in comp_at_ui.items():
        #     comp_at_ui[model_name] = np.mean(np.array(comp_at_ui_values))
        #     comp[model_name].append(comp_at_ui[model_name])
    

    for model_name, scores in u_new_scores.items():
        for i in range(len(TOP_KU_RANGE)):
            for j in range(len(TOP_KI_RANGE)):
                sij = np.array(scores[i][j])
                sij = np.argsort(-sij)
                sorted_items = np.take_along_axis(selected_items, sij, axis=0)
                # prec and recall @ 10
                target = test_ds.val_adj_matrix
                target = target[u].numpy()
                hits = np.sum(np.take_along_axis(target, sorted_items[:10], axis=0))
                recalls[model_name][i][j].append((hits/np.sum(target)))
                precisions[model_name][i][j].append(hits/10)
                # mf[model_name][i][j] += ((sij[:25] < 25).sum() / 25)
    # full model recall and precision
    hits = np.sum(np.take_along_axis(target, selected_items[:10], axis=0))
    recalls["full_model"].append((hits/np.sum(target)))
    precisions["full_model"].append(hits/10)
    # save the results every 20 users
    if u_ind % 20 == 0:
        with open(f"/home/ahmad/GRACE/scores/{exp['dataset']}/interpretable_model_recall.pkl", "wb") as f:
            pickle.dump(recalls, f)
        with open(f"/home/ahmad/GRACE/scores/{exp['dataset']}/interpretable_model_prec.pkl", "wb") as f:
            pickle.dump(precisions, f)

    # for model_name in models:
    #     print(model_name)
    #     print("\tcurrent COMP:", np.mean(np.array(comp[model_name])))
    #     print("\tcurrent MF@25:", mf[model_name] / (u+1))

print("recall full model: ", np.mean(recalls["full_model"]))
print("precision full model: ", np.mean(precisions["full_model"]))
print("recall: ", np.mean(recalls["GRACE"][0][0]))
print("precision: ", np.mean(precisions["GRACE"][0][0]))


with open(f"/home/ahmad/GRACE/scores/{exp['dataset']}/interpretable_model_recall.pkl", "wb") as f:
    pickle.dump(recalls, f)
with open(f"/home/ahmad/GRACE/scores/{exp['dataset']}/interpretable_model_prec.pkl", "wb") as f:
    pickle.dump(precisions, f)
# with open("/home/ahmad/GRACE/scores/comp.pkl", "wb") as f:
#     pickle.dump(comp, f)
# with open("/home/ahmad/GRACE/scores/mf.pkl", "wb") as f:
#     pickle.dump(mf, f)
