"""
This scripts computes the COMP metric of GNNExplainer.
"""
import os
import sys
import torch
import torch_sparse
import numpy as np
from tqdm import tqdm
from RecSys.GNN.models.LightGCN import LightGCN_simple
from RecSys.utils.data.data import RecGraphData
from RecSys.utils.config import load_everything_from_exp, get_config, Experiment

from torch import Tensor

os.chdir("../../")
device = "cuda:1"

LIGHTGCN_RES_PATH = "RecSys/GNN/config/best_config/results.yaml"
LIGHTGCN_MODEL_PATH = "RecSys/GNN/config/best_config/trained_models/best_val_Rec@25_ne.pt"

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

device = "cuda:1"

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


def threshold(subgraph, graph, mask: Tensor, topku: int, topki: int, idx_u, idx_i):
    top_users = torch.argsort(mask[idx_i], descending=True)
    top_items = torch.argsort(mask[idx_u], descending=True)
    top_users = top_users[subgraph.x[top_users, 0] < graph.num_users]
    top_items = top_items[subgraph.x[top_items, 0] >= graph.num_users]
    return top_users[:topku], top_items[:topki]  # , top_users[topku:], top_items[topki:]


def gnn_explainer(u: int, i: int, graph: RecGraphData, k: int=500, beta=0.5, alpha=0.1, topku=10, topki=10, verbose=1):
    with torch.no_grad():
        e0 = gnn.embedding.get_all(graph)
        e = gnn.get_embedding(graph)
        true_eu, true_ei = e[u], e[i]
        true_pred = torch.matmul(true_eu, e[graph.num_users:].T)
        # print(torch.argsort(true_pred, descending=True))
        if verbose == 1:
            print("true score:", true_pred[i - graph.num_users])
        thresh = true_pred[torch.argsort(true_pred, descending=True)[25]]
        if verbose == 1:
            print("threshold:", thresh)
        true_pred = torch.isin(i - graph.num_users, torch.argsort(true_pred, descending=True)[:25])
        if verbose == 1:
            print("true pred:", true_pred)
        # true_pred = torch.tensor(true_pred, dtype=torch.float32)

    subgraph, idx_u, idx_i = get_pair_subgraph(u, i, graph)
    if verbose == 1:
        print("idx u, i", idx_u, idx_i)

    adj_mat = subgraph.edge_index.coalesce().to_dense()

    deg = torch_sparse.sum(graph.edge_index, 1)[subgraph.x[:, 0]]
    deg = 1 / (1e-12 + torch.sqrt(deg))
    adj_mat = adj_mat * deg.view(-1, 1)  # torch.stack([deg*adj_mat[i] for i in range(adj_mat.shape[0])], 0)
    adj_mat = adj_mat * deg.view(1, -1)  # torch.stack([deg*adj_mat[:, i] for i in range(adj_mat.shape[1])], 1)

    mask = torch.zeros_like(adj_mat, dtype=torch.float32, requires_grad=True, device=device)
    # mask = torch.where(adj_mat == 0, -100*torch.ones_like(mask), mask)
    # mask = torch.tensor(mask, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.SGD([mask], lr=alpha)

    for i in range(k):
        subgraph.edge_index = torch.clone(adj_mat) * torch.sigmoid(mask)
        e0 = gnn.embedding.get_all(subgraph)
        e1 = torch.matmul(subgraph.edge_index, e0)
        eu = 0.5*(e0[idx_u] + e1[idx_u])
        ei = 0.5*(e0[idx_i] + e1[idx_i])
        y = torch.dot(eu, ei)

        # if true_pred:
        loss = torch.log(1 - torch.sigmoid(y-thresh) + 1e-3)
        # else:
        #     loss = torch.log(torch.sigmoid(y))

        opt.zero_grad()
        loss.backward()
        opt.step()
        if i % 50 == 0 and verbose == 1:
            print(
                "i:", i,
                "  y:", y.item(),
                "  sigma(y):", torch.sigmoid(y).item(),
                "  sigma(y-t):", torch.sigmoid(y-thresh).item(),
                "  max:", torch.sigmoid(mask).max().item(),
                "  min:", torch.sigmoid(mask).min().item()
            )

    # top_users, top_items, other_users, other_items = threshold(mask, 1, 1, idx_u, idx_i)
    # top_users, top_items = threshold(mask, topku, topki, idx_u, idx_i)
    # mask = torch.zeros_like(adj_mat, device=device)
    # mask[idx_i, top_users] = 1
    # mask[idx_u, top_items] = 1
    # with torch.no_grad():
    #     subgraph.edge_index = torch.clone(adj_mat) * torch.sigmoid(mask)
    #     e0 = gnn.embedding.get_all(subgraph)
    #     e1 = torch.matmul(subgraph.edge_index, e0)
    #     eu = 0.5*(e0[idx_u] + e1[idx_u])
    #     ei = 0.5*(e0[idx_i] + e1[idx_i])
    #     y = torch.dot(eu, ei)

    return subgraph, mask, idx_u, idx_i, adj_mat, y


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
    

    s_gnn = np.load("scores/s0.npy")[:NB_USER] \
            + np.load("scores/s01.npy")[:NB_USER] \
            + np.load("scores/s10.npy")[:NB_USER] \
            + np.load("scores/s1.npy")[:NB_USER]
    top_gnn = np.argsort(-s_gnn, axis=-1)

    comp = []

    for user in tqdm(range(NB_USER), "User id"):  # 359
        for i_item_id, item_id in tqdm(enumerate(top_gnn[user, :1000]), "Item id", 1000):  # 10187
            comp_at_ui = []
            big_i = item_id + train_graph.num_users
            subgraph, mask, idx_u, idx_i, adj_mat, _ = gnn_explainer(user, big_i, train_graph, k=500, alpha=1, topku=1, topki=1, verbose=0)
            top_users, top_items = threshold(subgraph, train_graph, mask, 100, 100, idx_u, idx_i)

            for i_topku, topku in enumerate([0, 1, 10, 100]):
                for i_topki, topki in enumerate([0, 1, 10, 100]):
                    if topku == topki == 0:
                        y_true = get_new_score(subgraph, idx_u, idx_i, adj_mat, top_users[:topku], top_items[:topki]).item()
                    else:
                        y = get_new_score(subgraph, idx_u, idx_i, adj_mat, top_users[:topku], top_items[:topki]).item()
                        comp_at_ui.append(sigmoid(y_true - y))

            comp_at_ui = np.mean(np.array(comp_at_ui))
            comp.append(comp_at_ui)
        print("current COMP:", np.mean(np.array(comp)))
    comp = np.array(comp)
    print("COMP:", np.mean(comp))

    np.save(f"scores/comp_gnnexplainer_{NB_USER}users{NB_ITEM}items.npy", comp)
