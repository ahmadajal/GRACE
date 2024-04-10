""" Edge weighting functions """
import pandas as pd
import torch
import torch_sparse
from tqdm import tqdm

from RecSys.utils.data.data import RecGraphData


def edge_weights_ones(graph: RecGraphData):
    r""" Set edge weights to one

    Args:
        graph (torch_geometric.data.Data): Graph
    """
    edge_index = graph.edge_index
    edge_index = torch.unique(edge_index, dim=1)
    edge_weight = torch.ones(edge_index.size(1), dtype=torch.float32, device=edge_index.device)
    graph.edge_attr = None

    sparse_edge_index = torch_sparse.SparseTensor(
        row=edge_index[0],
        col=edge_index[1],
        value=edge_weight,
        sparse_sizes=(graph.total_num_users + graph.num_items, graph.total_num_users + graph.num_items),
    )
    graph.edge_index = sparse_edge_index  # type: ignore


def edge_weights_num_interaction(graph: RecGraphData):
    r""" Set edge weights to one

    Args:
        graph (torch_geometric.data.Data): Graph
    """
    edge_index = graph.edge_index
    edge_index, counts = torch.unique(edge_index, return_counts=True, dim=1)
    edge_weight = counts.to(torch.float32)
    graph.edge_attr = None

    sparse_edge_index = torch_sparse.SparseTensor(
        row=edge_index[0],
        col=edge_index[1],
        value=edge_weight,
        sparse_sizes=(graph.total_num_users + graph.num_items, graph.total_num_users + graph.num_items),
    )
    graph.edge_index = sparse_edge_index  # type: ignore


def edge_weights_exp_time_since_last_inter(graph: RecGraphData, gamma, only_last=True):
    r""" Edge weights based on the timestamp
    The weight of an edge (u, i, t) is exp(-gamma * Delta_t))
    where Delta_t is the time between the last interaction of user u and the current interaction (u, i, t)

    Args:
        graph (torch_geometric.data.Data): Graph
        gamma (float): hyperparameter controlling the decay rate
    """
    assert graph.edge_attr is not None, "Timestamps are not provided for this dataset"
    edge_index = graph.edge_index
    user_last_interaction = torch.zeros((graph.total_num_users + graph.num_items,), dtype=torch.float32, device=graph.edge_attr.device)
    for u_id in range(graph.total_num_users):
        timestamps_u = graph.edge_attr[edge_index[0] == u_id]
        if len(timestamps_u) > 0:
            user_last_interaction[u_id] = torch.max(timestamps_u)
        else:
            user_last_interaction[u_id] = 1

    edge_weight = torch.ones((edge_index.size(1),), dtype=torch.float32, device=graph.edge_attr.device)
    edge_weight = torch.where(
        ((edge_index[0] < graph.total_num_users)),
        user_last_interaction[edge_index[0]],
        edge_weight
    )
    edge_weight = torch.where(
        ((edge_index[1] < graph.total_num_users)),
        user_last_interaction[edge_index[1]],
        edge_weight
    )
    # edge_weight = (edge_weight - graph.edge_attr[:, 0]) / torch.where(edge_weight > 0, edge_weight, torch.ones_like(edge_weight))
    edge_weight = edge_weight - graph.edge_attr
    edge_weight /= 7263488.  # torch.median(edge_weight[edge_weight > 0])
    edge_weight = torch.exp(-gamma*edge_weight)

    np_edge_weight = edge_weight.cpu().numpy()
    np_edge_index = edge_index.cpu().numpy()
    df = pd.DataFrame({"u": np_edge_index[0], "i": np_edge_index[1], "w": np_edge_weight})
    if only_last:
        df = df.groupby(["u", "i"]).max().reset_index()
    else:
        df = df.groupby(["u", "i"]).sum().reset_index()
    unique_edge_index = torch.tensor(df[["u", "i"]].values.T, dtype=torch.long, device=graph.edge_attr.device)
    unique_edge_weight = torch.tensor(df["w"].values, dtype=torch.float32, device=graph.edge_attr.device)

    edge_index = unique_edge_index
    edge_weight = unique_edge_weight
    sparse_edge_index = torch_sparse.SparseTensor(
        row=edge_index[0],
        col=edge_index[1],
        value=edge_weight,
        sparse_sizes=(graph.total_num_users + graph.num_items, graph.total_num_users + graph.num_items),
    )
    graph.edge_index = sparse_edge_index  # type: ignore
    graph.edge_attr = None


def edge_weights_exp_time(graph: RecGraphData, gamma, only_last=True):
    r""" Edge weights based on the timestamp
    The weight of an edge (u, i, t) is exp(-gamma * Delta_t))
    where Delta_t is the time between the last interaction (overall) and the current interaction (u, i, t)

    Args:
        graph (torch_geometric.data.Data): Graph
        gamma (float): hyperparameter controlling the decay rate
    """
    assert graph.edge_attr is not None, "Timestamps are not provided for this dataset"
    edge_index = graph.edge_index
    edge_index, counts = torch.unique(edge_index, return_counts=True, dim=1)
    last_interaction = graph.edge_attr.max()

    edge_weight = last_interaction * torch.ones((edge_index.size(1),), dtype=torch.float32, device=graph.edge_attr.device)
    edge_weight = torch.where(
        ((edge_index[1] >= graph.total_num_users) & (edge_index[0] >= graph.total_num_users)),
        torch.ones_like(edge_weight),
        edge_weight,
    )
    edge_weight = edge_weight - graph.edge_attr
    edge_weight /= 7263488  # last_interaction
    edge_weight = torch.exp(-gamma*edge_weight)

    np_edge_weight = edge_weight.cpu().numpy()
    np_edge_index = edge_index.cpu().numpy()
    df = pd.DataFrame({"u": np_edge_index[0], "i": np_edge_index[1], "w": np_edge_weight})
    if only_last:
        df = df.groupby(["u", "i"]).max().reset_index()
    else:
        df = df.groupby(["u", "i"]).sum().reset_index()
    unique_edge_index = torch.tensor(df[["u", "i"]].values.T, dtype=torch.long, device=graph.edge_attr.device)
    unique_edge_weight = torch.tensor(df["w"].values, dtype=torch.float32, device=graph.edge_attr.device)

    edge_index = unique_edge_index
    edge_weight = unique_edge_weight

    sparse_edge_index = torch_sparse.SparseTensor(
        row=edge_index[0],
        col=edge_index[1],
        value=edge_weight,
        sparse_sizes=(graph.total_num_users + graph.num_items, graph.total_num_users + graph.num_items),
    )
    graph.edge_index = sparse_edge_index  # type: ignore
    graph.edge_attr = None
