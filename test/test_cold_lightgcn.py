import pandas as pd
import numpy as np
import torch
import torch_sparse
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from RecSys.nn.models.embeddings import IdEmbedding
from RecSys.nn.models.LightGCN import LightGCN_simple
from RecSys.utils.data.data import RecGraphData
from RecSys.cold_start.data import ColdStartData
from RecSys.cold_start.models.lightgcn import LightGCN_wo_first_embedding_merged
from RecSys.cold_start.models.lightgcn import compute_degree, normalize_adj_matrix
from RecSys.cold_start.models.lightgcn import compute_mean_embedding


def test_compute_degree():
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    deg = compute_degree(edge_index, 3)
    assert torch.allclose(deg, torch.tensor([1, 2, 1], dtype=torch.float32))

    edge_index = torch_sparse.SparseTensor(
        row=torch.tensor([0, 1, 1, 2]),
        col=torch.tensor([1, 0, 2, 1]),
        sparse_sizes=(3, 3),
    )
    deg = compute_degree(edge_index, 3)
    assert torch.allclose(deg, torch.tensor([1, 2, 1], dtype=torch.float32))


def test_normalize_adj_matrix():
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    deg = torch.tensor([1, 1, 1], dtype=torch.float32)
    adj = normalize_adj_matrix(deg, edge_index, 3)
    adj = adj.to_dense()
    assert torch.allclose(adj, torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.float32))

    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    inv_sqrt_2 = 1 / np.sqrt(2)
    deg = torch.tensor([1, 2, 1], dtype=torch.float32)
    adj = normalize_adj_matrix(deg, edge_index, 3)
    adj = adj.to_dense()
    assert torch.allclose(adj, torch.tensor([[0, inv_sqrt_2, 0], [inv_sqrt_2, 0, inv_sqrt_2], [0, inv_sqrt_2, 0]], dtype=torch.float32))

    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    inv_sqrt_2 = 1 / np.sqrt(2)
    deg = torch.tensor([2, 2, 1], dtype=torch.float32)
    adj = normalize_adj_matrix(deg, edge_index, 3)
    adj = adj.to_dense()
    assert torch.allclose(adj, torch.tensor([[0, 0.5, 0], [0.5, 0, inv_sqrt_2], [0, inv_sqrt_2, 0]], dtype=torch.float32))

    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    inv_sqrt_2 = 1 / np.sqrt(2)
    deg = torch.tensor([0, 2, 1], dtype=torch.float32)
    adj = normalize_adj_matrix(deg, edge_index, 3)
    adj = adj.to_dense()
    assert torch.allclose(adj, torch.tensor([[0, 0, 0], [0, 0, inv_sqrt_2], [0, inv_sqrt_2, 0]], dtype=torch.float32))

    user_df = pd.read_csv("data/tomplay/users.csv")
    user_df = user_df[user_df["u"] < 200]
    item_df = pd.read_csv("data/tomplay/items.csv")
    item_df = item_df[item_df["i"] < 300]
    inter_df = pd.read_csv("data/tomplay/interactions.csv")
    inter_df = inter_df[inter_df["u"] < 200]
    inter_df = inter_df[inter_df["i"] < 300]

    graph = RecGraphData(user_df, item_df, inter_df, inter_df)

    num_nodes = graph.num_items + graph.num_users
    edge_index, edge_weights = gcn_norm(graph.edge_index, num_nodes=num_nodes, add_self_loops=False)  # type: ignore
    real_graph = torch_sparse.SparseTensor(
        row=edge_index[0],
        col=edge_index[1],
        value=edge_weights,
        sparse_sizes=(num_nodes, num_nodes)
    )

    my_graph = normalize_adj_matrix(compute_degree(graph.edge_index, num_nodes), graph.edge_index, num_nodes)
    assert torch.allclose(real_graph.to_dense(), my_graph.to_dense())


def test_normalize_lightgcn():
    user_df = pd.read_csv("data/tomplay/users.csv")
    item_df = pd.read_csv("data/tomplay/items.csv")
    inter_df = pd.read_csv("data/tomplay/interactions.csv")

    graph = RecGraphData(user_df, item_df, inter_df, inter_df)
    model = LightGCN_simple(1, IdEmbedding.from_graph(graph, 64))

    num_nodes = graph.num_items + graph.num_users
    adj_normalized = normalize_adj_matrix(compute_degree(graph.edge_index, num_nodes), graph.edge_index, num_nodes)
    x = model.embedding.get_all(graph)
    y_1 = torch_sparse.matmul(adj_normalized, x) + x

    # y_2 = model.layers[0](x, graph.edge_index)
    y_2 = 2*model.get_embedding(graph)

    assert torch.abs(y_1 - y_2).sum() < 1e-2


def test_lightgcn_wo_first_embedding():
    user_df = pd.read_csv("data/tomplay/users.csv")
    item_df = pd.read_csv("data/tomplay/items.csv")
    inter_df = pd.read_csv("data/tomplay/interactions.csv")

    cold_user_id = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    data = ColdStartData(user_df, item_df, inter_df, inter_df, cold_user_id)

    pre_data, post_data, test_ds, user_id = data.get_test_data(2)

    lightgcn = LightGCN_simple(2, IdEmbedding.from_graph(data.train_data, 64))
    model = LightGCN_wo_first_embedding_merged(lightgcn)

    with torch.no_grad():
        preds = model(data, pre_data, user_id)
    print(preds.shape)
    assert preds.shape == (len(user_id), data.train_data.num_items)


def test_compute_mean_embedding():
    user_df = pd.read_csv("data/tomplay/users.csv")
    item_df = pd.read_csv("data/tomplay/items.csv")
    inter_df = pd.read_csv("data/tomplay/interactions.csv")

    cold_user_id = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # train_user_id = np.arange(10, len(user_df))

    graph = RecGraphData(user_df, item_df, inter_df, inter_df)

    model = LightGCN_simple(2, IdEmbedding.from_graph(graph, 64))

    x = model.embedding.get_all(graph)

    # on = None
    mean_embedding = compute_mean_embedding(user_df, [], x, None)
    assert isinstance(mean_embedding, torch.Tensor)
    assert mean_embedding.shape == (64,)
    assert torch.allclose(mean_embedding, x[:graph.num_users].mean(dim=0))

    mean_embedding = compute_mean_embedding(user_df, cold_user_id, x, None)
    assert isinstance(mean_embedding, torch.Tensor)
    assert mean_embedding.shape == (64,)
    assert torch.allclose(mean_embedding, x[10:graph.num_users].mean(dim=0))

    # on = "instrument"
    mean_embedding = compute_mean_embedding(user_df, cold_user_id, x, "instrument_id")
    user_id_0 = user_df[user_df["instrument_id"] == 0]["u"].values
    user_id_0 = torch.tensor(user_id_0, dtype=torch.long)
    user_id_0 = user_id_0[user_id_0 > 9]
    true_res = x[user_id_0].mean(dim=0)
    assert mean_embedding[0].shape == (64,)
    assert torch.allclose(mean_embedding[0], true_res)

    # on = "LEVEL"
    mean_embedding = compute_mean_embedding(user_df, cold_user_id, x, "LEVEL")
    user_id_0 = user_df[user_df["LEVEL"] == 0]["u"].values
    user_id_0 = torch.tensor(user_id_0, dtype=torch.long)
    user_id_0 = user_id_0[user_id_0 > 9]
    true_res = x[user_id_0].mean(dim=0)
    assert mean_embedding[0].shape == (64,)
    assert torch.allclose(mean_embedding[0], true_res)

    # on = "instrument", "LEVEL"
    mean_embedding = compute_mean_embedding(user_df, cold_user_id, x, ["instrument_id", "LEVEL"])
    user_id_0 = user_df[(user_df["instrument_id"] == 0) & (user_df["LEVEL"] == 0)]["u"].values
    user_id_0 = torch.tensor(user_id_0, dtype=torch.long)
    user_id_0 = user_id_0[user_id_0 > 9]
    true_res = x[user_id_0].mean(dim=0)
    assert mean_embedding[(0, 0)].shape == (64,)
    assert torch.allclose(mean_embedding[(0, 0)], true_res)
