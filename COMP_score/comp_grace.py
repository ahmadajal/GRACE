"""
This scripts computes the COMP metric for our interpretable model.

"""
import os
import sys
import torch
import torch_sparse
import numpy as np
from time import time
from tqdm import tqdm
from RecSys.nn.models.LightGCN import LightGCN_simple
from RecSys.utils.data.data import RecGraphData
from RecSys.utils.config import load_everything_from_exp, get_config, Experiment

from torch import Tensor

os.chdir("../../")
device = "cuda:0"

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

os.chdir("RecSys/interpretability")

device = "cuda:0"

gnn = gnn.to(device)
train_graph = train_graph.to(device)
del test_graph
del train_ds
del test_ds
del optimizer
del scheduler

def get_neighbors(n: int, graph: RecGraphData) -> Tensor:
    neighbors = torch.cat(
        (torch.unique(graph.directed_edge_index[1][graph.directed_edge_index[0] == n]),
        torch.unique(graph.directed_edge_index[0][graph.directed_edge_index[1] == n])), dim=0)
    neighbors = torch.sort(torch.unique(neighbors))[0]
    return neighbors


def get_pair_subgraph(u: int, i: int, graph: RecGraphData) -> RecGraphData:
    neighbors = torch.cat((
        torch.tensor([u], device=device),
        torch.tensor([i], device=device),
        get_neighbors(u, graph),
        get_neighbors(i, graph)), dim=0)
    neighbors = torch.sort(torch.unique(neighbors))[0]
    subgraph = graph.subgraph(neighbors)
    idx_u = torch.where(subgraph.x[:, 0] == u)[0][0]
    idx_i = torch.where(subgraph.x[:, 0] == i)[0][0]
    return subgraph, idx_u, idx_i


def distance(eu, ei, true_eu, true_ei):
    e = torch.cat((eu, ei), dim=0)
    true_e = torch.cat((true_eu, true_ei), dim=0)
    return torch.sum((e - true_e) * (e - true_e))


def threshold(mask: Tensor, topku: int, topki: int, idx_u, idx_i):
    top_users = torch.argsort(mask[idx_i], descending=True)
    top_items = torch.argsort(mask[idx_u], descending=True)
    return top_users[:topku], top_items[:topki]  # , top_users[topku:], top_items[topki:]


def get_top(u: int, i: int, graph: RecGraphData):
    subgraph, idx_u, idx_i = get_pair_subgraph(u, i, graph)

    adj_mat = subgraph.edge_index.coalesce().to_dense()

    deg = torch_sparse.sum(graph.edge_index, 1)[subgraph.x[:, 0]]
    deg = 1 / (1e-12 + torch.sqrt(deg))
    # adj_mat = adj_mat * deg.view(-1, 1)  # torch.stack([deg*adj_mat[i] for i in range(adj_mat.shape[0])], 0)
    # adj_mat = adj_mat * deg.view(1, -1)  # torch.stack([deg*adj_mat[:, i] for i in range(adj_mat.shape[1])], 1)

    subgraph.edge_index = torch.clone(adj_mat)
    e0 = gnn.embedding.get_all(subgraph)
    e1 = torch.clone(e0) * deg.view(-1, 1)

    s01 = adj_mat[idx_i] * (e0[idx_u].unsqueeze(0) @ e1.T)
    top_users = torch.argsort(-s01[0])
    top_users = top_users[subgraph.x[top_users, 0] < graph.num_users]

    s10 = adj_mat[idx_u] * (e0[idx_i].unsqueeze(0) @ e1.T)
    top_items = torch.argsort(-s10[0])
    top_items = top_items[subgraph.x[top_items, 0] >= graph.num_users]
    # print(s10[0, subgraph.x[:, 0] >= graph.num_users])
    # print(subgraph.x[:, 0][subgraph.x[:, 0] >= graph.num_users] - graph.num_users)

    return subgraph, idx_u, idx_i, adj_mat, top_users, top_items


def get_new_score(subgraph, idx_u, idx_i, adj_mat, top_users, top_items):
    mask = torch.ones_like(adj_mat, device=device)
    mask[idx_i, top_users] = 0
    mask[idx_u, top_items] = 0
    with torch.no_grad():
        subgraph.edge_index = torch.clone(adj_mat) * mask
        e0 = gnn.embedding.get_all(subgraph)
        e1 = torch.matmul(subgraph.edge_index, e0)
        eu = 0.5*(e0[idx_u] + e1[idx_u])
        ei = 0.5*(e0[idx_i] + e1[idx_i])
        y = torch.dot(eu, ei)

    return y


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    NB_USER = 100
    NB_ITEM = 1000

    s = np.load("scores/s0.npy") + np.load("scores/s01.npy") + np.load("scores/s10.npy") + np.load("scores/s1.npy")
    comp = []
    # top_users, top_items = sensitivity_analysis_top(train_graph)

    for u in tqdm(range(NB_USER)):  # [15914]:  # [359]:
        for i in tqdm(np.argsort(-s[u])[:NB_ITEM]):  # [25617]:  # [10194]:
            comp_at_ui = []
            top = time()
            subgraph, idx_u, idx_i, adj_mat, top_users, top_items = get_top(u, i+train_graph.num_users, train_graph)
            # print(time() - top)
            # print(subgraph.x[top_users[:3], 0])
            # print(subgraph.x[top_items[:5], 0] - train_graph.num_users)
            # exit()

            for i_top_ku, top_ku in enumerate([0, 1, 10, 100]):
                for i_top_ki, top_ki in enumerate([0, 1, 10, 100]):
                    if top_ku == 0 and top_ki == 0:
                        y_true = get_new_score(subgraph, idx_u, idx_i, adj_mat, top_users[:top_ku], top_items[:top_ki]).item()
                    else:
                        y = get_new_score(subgraph, idx_u, idx_i, adj_mat, top_users[:top_ku], top_items[:top_ki])
                        y = y.item()
                        comp_at_ui.append(sigmoid(y_true - y))
            comp_at_ui = np.mean(np.array(comp_at_ui))
            comp.append(comp_at_ui)
        print("current COMP:", np.mean(np.array(comp)))
    comp = np.array(comp)
    print("COMP:", np.mean(comp))

    np.save(f"scores/comp_ours_{NB_USER}users{NB_ITEM}items.npy", comp)
