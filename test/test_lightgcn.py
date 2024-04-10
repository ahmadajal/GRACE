import pandas as pd
import torch
import torch_geometric.nn
from RecSys.nn.models.LightGCN import LightGCN_simple
from RecSys.nn.models.embeddings import IdEmbedding
from RecSys.utils.data import data
from RecSys.utils.data import edge_weighting


def test_LightGCN_simple():
    users_df = pd.read_csv('data/tomplay/users.csv')
    items_df = pd.read_csv('data/tomplay/items.csv')
    inter_df = pd.read_csv('data/tomplay/interactions.csv')
    graph = data.RecGraphData(users_df, items_df, inter_df, inter_df)
    edge_weighting.edge_weights_ones(graph)

    embedding = IdEmbedding.from_graph(graph, embedding_dim=10)
    torch.nn.init.constant_(embedding.embedding.weight, 1)

    model = LightGCN_simple(nb_layers=1, embedding=embedding)
    model_valid = torch_geometric.nn.models.LightGCN(num_layers=1, num_nodes=graph.num_items+graph.num_users, embedding_dim=10)
    torch.nn.init.constant_(model_valid.embedding.weight, 1)
    out = model.get_embedding(graph)
    out_valid = model_valid.get_embedding(graph.edge_index)
    assert (out != out_valid).sum().item() == 0


def test_LightGCN_simple_weights():
    users_df = pd.read_csv('data/tomplay/users.csv')
    items_df = pd.read_csv('data/tomplay/items.csv')
    inter_df = pd.read_csv('data/tomplay/interactions.csv')
    graph = data.RecGraphData(users_df, items_df, inter_df, inter_df)
    edge_weighting.edge_weights_ones(graph)

    embedding = IdEmbedding.from_graph(graph, embedding_dim=10)
    torch.nn.init.constant_(embedding.embedding.weight, 1)

    model = LightGCN_simple(nb_layers=1, embedding=embedding)
    model.get_embedding(graph)
