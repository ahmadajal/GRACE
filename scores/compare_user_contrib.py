"""Script to compare user contributions estimated by LightGCN and User2User CF

"""
import torch
import numpy as np
from tqdm import tqdm
from RecSys.nn.models.LightGCN import LightGCN_simple
from RecSys.nn.models.similarity import User2UserCosineSimilarity
from RecSys.utils.config import get_config, load_everything_from_exp, Experiment

# Device
device = "cuda:1"

LIGHTGCN_RES_PATH = "RecSys/config/best_config/results.yaml"
LIGHTGCN_MODEL_PATH = "RecSys/config/best_config/trained_models/best_val_Rec@25_ne.pt"

# Load LightGCN
exp = get_config(LIGHTGCN_RES_PATH)
exp = Experiment(exp["Config"])
print("#"*20)
print(exp)
print("#"*20)

datas, model = load_everything_from_exp(exp, device, test=True)
(train_graph, test_graph, train_ds, test_ds) = datas
(gnn, optimizer, scheduler, loss_fn) = model
assert isinstance(gnn, LightGCN_simple)
gnn.load_state_dict(torch.load(LIGHTGCN_MODEL_PATH, device))
gnn.eval()
gnn = gnn.to(device)


# Load User2User
adj_matrix = User2UserCosineSimilarity.compute_adj_matrix(train_graph).to(device)
user_sim = User2UserCosineSimilarity.compute_user_similarities(adj_matrix).to(device)

user_degrees = 1 / (torch.sqrt(torch.sum(adj_matrix, dim=1)) + 1e-8)
item_degrees = 1 / (torch.sqrt(torch.sum(adj_matrix, dim=0)) + 1e-8)


def compute_lightgcn_contrib(u, i, e0, user_degrees, item_degrees, adj_matrix):
    user_sim = (e0[u].unsqueeze(0) @ e0[:num_users].T).squeeze()
    contrib = user_sim * adj_matrix[:, i].T * user_degrees * item_degrees[i]
    return contrib


def compute_user2user_contrib(u, i, user_sim):
    contrib = user_sim[u, :] * adj_matrix[:, i].T
    return contrib


class RunningAverage():
    def __init__(self, name):
        self.name = name
        self.avg = 0
        self.n = 0

    def update(self, x):
        if not np.isnan(x):
            self.n += 1
            self.avg = ((self.n - 1) / self.n) * (self.avg + x / self.n)

    def __repr__(self):
        return f"{self.name}: {self.avg}"


def top_k_acc(k, s1, s2):
    if len(s1) < k:
        return np.nan
    s1 = set(s1[:k])
    s2 = set(s2[:k])
    return len(s1.intersection(s2)) / k


def bot_k_acc(k, s1, s2):
    if len(s1) < k:
        return np.nan
    s1 = set(s1[-k:])
    s2 = set(s2[-k:])
    return len(s1.intersection(s2)) / k


def correlation(s1, s2):
    return np.corrcoef(s1, s2)[0, 1]


def top_k_correlation(k, s1, s2, ind_sort):
    if len(s1) < k:
        return np.nan
    return np.corrcoef(s1[ind_sort[:k]], s2[ind_sort[:k]])[0, 1]


def bot_k_correlation(k, s1, s2, ind_sort):
    if len(s1) < k:
        return np.nan
    return np.corrcoef(s1[ind_sort[-k:]], s2[ind_sort[-k:]])[0, 1]


overall_avg_corr = RunningAverage("overall average correlation in score")
overall_avg_rank_corr = RunningAverage("overall average correlation in rank")
overall_avg_corr_top_10 = RunningAverage("overall average correlation in top 10 score")
overall_avg_corr_bot_10 = RunningAverage("overall average correlation in bot 10 score")
overall_avg_corr_top_25 = RunningAverage("overall average correlation in top 25 score")
overall_avg_corr_bot_25 = RunningAverage("overall average correlation in bot 25 score")
overall_avg_top_10 = RunningAverage("overall average top 10 accuracy")
overall_avg_bot_10 = RunningAverage("overall average bot 10 accuracy")
overall_avg_top_25 = RunningAverage("overall average top 25 accuracy")
overall_avg_bot_25 = RunningAverage("overall average bot 25 accuracy")

# Compute the scores
with torch.no_grad():  # no need to compute gradients
    e0 = gnn.embedding.get_all(train_graph).to(device)
    num_users = train_graph.num_users
    num_items = train_graph.num_items

    for u in tqdm(range(num_users)):
        for i in range(num_items):
            gnn_contrib = compute_lightgcn_contrib(u, i, e0, user_degrees, item_degrees, adj_matrix).cpu().numpy()
            cf_contrib = compute_user2user_contrib(u, i, user_sim).cpu().numpy()
            mask = cf_contrib > 0
            gnn_contrib = gnn_contrib[mask]
            cf_contrib = cf_contrib[mask]

            ui_cor = correlation(gnn_contrib, cf_contrib)

            as_gnn_contrib = np.argsort(gnn_contrib)
            as_cf_contrib = np.argsort(cf_contrib)
            ui_rank_cor = correlation(as_gnn_contrib, as_cf_contrib)

            ui_cor_top10 = top_k_correlation(10, gnn_contrib, cf_contrib, as_gnn_contrib)
            ui_cor_bot10 = bot_k_correlation(10, gnn_contrib, cf_contrib, as_gnn_contrib)

            ui_cor_top25 = top_k_correlation(25, gnn_contrib, cf_contrib, as_gnn_contrib)
            ui_cor_bot25 = bot_k_correlation(25, gnn_contrib, cf_contrib, as_gnn_contrib)

            ui_top10 = top_k_acc(10, as_gnn_contrib, as_cf_contrib)
            ui_top25 = top_k_acc(25, as_gnn_contrib, as_cf_contrib)

            ui_bot10 = bot_k_acc(10, as_gnn_contrib, as_cf_contrib)
            ui_bot25 = bot_k_acc(25, as_gnn_contrib, as_cf_contrib)

            overall_avg_corr.update(ui_cor)
            overall_avg_corr_top_10.update(ui_cor_top10)
            overall_avg_corr_bot_10.update(ui_cor_bot10)
            overall_avg_corr_top_25.update(ui_cor_top25)
            overall_avg_corr_bot_25.update(ui_cor_bot25)
            overall_avg_rank_corr.update(ui_rank_cor)
            overall_avg_top_10.update(ui_top10)
            overall_avg_top_25.update(ui_top25)
            overall_avg_bot_10.update(ui_bot10)
            overall_avg_bot_25.update(ui_bot25)

        print(overall_avg_corr)
        print(overall_avg_corr_top_10)
        print(overall_avg_corr_bot_10)
        print(overall_avg_corr_top_25)
        print(overall_avg_corr_bot_25)
        print(overall_avg_rank_corr)
        print(overall_avg_top_10)
        print(overall_avg_bot_10)
        print(overall_avg_top_25)
        print(overall_avg_bot_25)
