import torch
import torch_sparse
from RecSys.utils.data.data import RecGraphData
from RecSys.cold_start.models import ColdStartModel
from RecSys.cold_start.data import ColdStartData
from RecSys.nn.models.LightGCN import LightGCN_simple
from typing import Union


def compute_degree(edge_index: Union[torch.Tensor, torch_sparse.SparseTensor], num_nodes: int) -> torch.Tensor:
    """
    Compute the degree of each node in the graph

    Args:
        edge_index (torch.Tensor or torch_sparse.SparseTensor): Edge index
        num_nodes (int): Number of nodes in the graph

    Returns:
        torch.Tensor: Degree of each node
    """
    if isinstance(edge_index, torch.Tensor):
        row, col = edge_index[0], edge_index[1]
        deg = torch.zeros(num_nodes, dtype=torch.float32, device=row.device)
        deg = deg.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float32))
        deg = deg.scatter_add_(0, col, torch.ones_like(col, dtype=torch.float32))
        deg *= 0.5
    elif isinstance(edge_index, torch_sparse.SparseTensor):
        deg = torch_sparse.sum(edge_index, dim=0)

    else:
        raise ValueError("edge_index should be a torch.Tensor or torch_sparse.SparseTensor")

    return deg


def normalize_adj_matrix(deg: torch.Tensor, edge_index: Union[torch.Tensor, torch_sparse.SparseTensor], num_nodes: int):
    """ Normalize the adjacency matrix

    Args:
        deg (torch.Tensor): Degree of each node
        edge_index (torch.Tensor or torch_sparse.SparseTensor): Edge index
        num_nodes (int): Number of nodes in the graph

    Returns:
        torch.Tensor or torch_sparse.SparseTensor: Normalized adjacency matrix
    """
    deg = deg.clamp(min=0.0001)
    deg_inv_sqrt = torch.where(deg < 0.001, torch.zeros_like(deg, dtype=torch.float32), deg.pow(-0.5))
    if isinstance(edge_index, torch.Tensor):
        weights = deg_inv_sqrt[edge_index[0]] * deg_inv_sqrt[edge_index[1]]
        edge_index = torch_sparse.SparseTensor(
            row=edge_index[0],
            col=edge_index[1],
            value=weights,
            sparse_sizes=(num_nodes, num_nodes),
        )

    elif isinstance(edge_index, torch_sparse.SparseTensor):
        row = edge_index.row  # type: ignore
        col = edge_index.col  # type: ignore
        weights = edge_index.value  # type: ignore
        weights *= deg_inv_sqrt[row] * deg_inv_sqrt[col]
        edge_index = torch_sparse.SparseTensor(
            row=row,
            col=col,
            value=weights,
            sparse_sizes=(num_nodes, num_nodes),
        )

    else:
        raise ValueError("edge_index should be a torch.Tensor or torch_sparse.SparseTensor")

    return edge_index


def compute_mean_embedding(user_df, cold_user_id, x, on=None):
    if on is None:
        train_user_id = set(list(range(len(user_df))))
        train_user_id = train_user_id.difference(set(cold_user_id))
        train_user_id = torch.tensor(list(train_user_id), dtype=torch.long, device=x.device)
        emb_instrument = x[train_user_id].mean(dim=0)
    elif isinstance(on, str):
        emb_instrument = dict()
        for instrument in user_df[on].unique():
            user_instr = user_df[user_df[on] == instrument]["u"].values
            user_instr = set(user_instr)
            train_user_instr = user_instr.difference(set(cold_user_id))
            train_user_instr = torch.tensor(list(train_user_instr), dtype=torch.long, device=x.device)
            emb_instrument[instrument] = x[train_user_instr].mean(dim=0)
    elif isinstance(on, list):
        assert len(on) == 2
        emb_instrument = dict()
        for instrument in user_df[on[0]].unique():
            for instrument2 in user_df[user_df[on[0]] == instrument][on[1]].unique():
                user_instr = user_df[(user_df[on[0]] == instrument) & (user_df[on[1]] == instrument2)]["u"].values
                user_instr = set(user_instr)
                train_user_instr = user_instr.difference(set(cold_user_id))
                train_user_instr = torch.tensor(list(train_user_instr), dtype=torch.long, device=x.device)
                emb_instrument[(instrument, instrument2)] = x[train_user_instr].mean(dim=0)
    else:
        raise ValueError("on must be str or list")
    return emb_instrument


class LightGCN_wo_first_embedding(ColdStartModel):
    """ LightGCN without the first embedding layer

    e_cold_u^(k+1) = sum_{i in N(u)} e_i^(k) / deg_cold(u) deg_cold(i)
    """

    def __init__(self, model: LightGCN_simple):
        super().__init__()
        self.model = model

    def forward(self, cold_data: ColdStartData, pre_graph: RecGraphData, users_id: torch.Tensor) -> torch.Tensor:
        # e_0
        x = self.model.embedding.get_all(cold_data.train_data)

        # e_1_cold
        out_cold = self.model.layers[0](torch.clone(x), pre_graph.edge_index)

        for i in range(0, self.model.nb_layers-1):
            # A_train^(l-1) e_0
            x = self.model.layers[i](x, cold_data.train_data.edge_index)
            # A_pre A_train^(l-1) e_0
            out_cold += self.model.layers[i+1](torch.clone(x), pre_graph.edge_index)

        # Normalize
        out_cold /= self.model.nb_layers
        out_train = self.model.get_embedding(cold_data.train_data)

        # Compute the score
        out = torch.matmul(out_cold[:cold_data.train_data.num_users], out_train[cold_data.train_data.num_users:].T)
        return out[users_id]


class LightGCN_wo_first_embedding_keep_normalization(ColdStartModel):
    """ LightGCN without the first embedding layer
    The normalization of the train graph is kept for items

    e_cold_u^(k+1) = sum_{i in N(u)} e_i^(k) / deg_cold(u) deg_train(i)
    """

    def __init__(self, model: LightGCN_simple):
        super().__init__()
        self.model = model

    def forward(self, cold_data: ColdStartData, pre_graph: RecGraphData, users_id: torch.Tensor) -> torch.Tensor:
        # compute the degrees in each graph
        num_nodes = cold_data.train_data.num_users + cold_data.train_data.num_items
        deg_train = compute_degree(cold_data.train_data.edge_index, num_nodes)
        deg_pre = compute_degree(pre_graph.edge_index, num_nodes)

        # update degree of items only
        deg_pre[len(cold_data.user_df):] = deg_train[len(cold_data.user_df):]
        # normalize the adjacency matrix of the pre graph with the new degrees
        pre_edge_index = normalize_adj_matrix(deg_pre, pre_graph.edge_index, num_nodes)

        # e_0
        x = self.model.embedding.get_all(cold_data.train_data)

        # e_1_cold
        out_cold = torch_sparse.matmul(pre_edge_index, torch.clone(x))

        for i in range(0, self.model.nb_layers-1):
            # A_train^(l-1) e_0
            x = self.model.layers[i](x, cold_data.train_data.edge_index)
            # A_pre A_train^(l-1) e_0
            out_cold += torch_sparse.matmul(pre_edge_index, torch.clone(x))  # type: ignore

        # Normalize
        out_cold /= self.model.nb_layers  # type: ignore
        out_train = self.model.get_embedding(cold_data.train_data)

        # Compute the score
        out = torch.matmul(out_cold[:cold_data.train_data.num_users], out_train[cold_data.train_data.num_users:].T)
        return out[users_id]


class LightGCN_wo_first_embedding_sum_normalization(ColdStartModel):
    """ LightGCN without the first embedding layer
    The normalization of the train graph is kept for items

    e_cold_u^(k+1) = sum_{i in N(u)} e_i^(k) / deg_cold(u) deg_train(i)
    """

    def __init__(self, model: LightGCN_simple):
        super().__init__()
        self.model = model

    def forward(self, cold_data: ColdStartData, pre_graph: RecGraphData, users_id: torch.Tensor) -> torch.Tensor:
        # compute the degrees in each graph
        num_nodes = cold_data.train_data.num_users + cold_data.train_data.num_items
        deg_train = compute_degree(cold_data.train_data.edge_index, num_nodes)
        deg_pre = compute_degree(pre_graph.edge_index, num_nodes)

        # update degree of items only
        deg_pre[len(cold_data.user_df):] += deg_train[len(cold_data.user_df):]
        # normalize the adjacency matrix of the pre graph with the new degrees
        pre_edge_index = normalize_adj_matrix(deg_pre, pre_graph.edge_index, num_nodes)

        # e_0
        x = self.model.embedding.get_all(cold_data.train_data)

        # e_1_cold
        out_cold = torch_sparse.matmul(pre_edge_index, torch.clone(x))

        for i in range(0, self.model.nb_layers-1):
            # A_train^(l-1) e_0
            x = self.model.layers[i](x, cold_data.train_data.edge_index)
            # A_pre A_train^(l-1) e_0
            out_cold += torch_sparse.matmul(pre_edge_index, torch.clone(x))  # type: ignore

        # Normalize
        out_cold /= self.model.nb_layers  # type: ignore
        out_train = self.model.get_embedding(cold_data.train_data)

        # Compute the score
        out = torch.matmul(out_cold[:cold_data.train_data.num_users], out_train[cold_data.train_data.num_users:].T)
        return out[users_id]


class LightGCN_wo_first_embedding_no_normalization(ColdStartModel):
    """ LightGCN without the first embedding layer
    The normalization of the train graph is kept for items

    e_cold_u^(k+1) = sum_{i in N(u)} e_i^(k) / deg_cold(u) deg_train(i)
    """

    def __init__(self, model: LightGCN_simple):
        super().__init__()
        self.model = model

    def forward(self, cold_data: ColdStartData, pre_graph: RecGraphData, users_id: torch.Tensor) -> torch.Tensor:
        # compute the degrees in each graph
        num_nodes = cold_data.train_data.num_users + cold_data.train_data.num_items
        deg_pre = compute_degree(pre_graph.edge_index, num_nodes)

        # update degree of items only
        deg_pre[len(cold_data.user_df):] = 1
        # normalize the adjacency matrix of the pre graph with the new degrees
        pre_edge_index = normalize_adj_matrix(deg_pre, pre_graph.edge_index, num_nodes)

        # e_0
        x = self.model.embedding.get_all(cold_data.train_data)

        # e_1_cold
        out_cold = torch_sparse.matmul(pre_edge_index, torch.clone(x))

        for i in range(0, self.model.nb_layers-1):
            # A_train^(l-1) e_0
            x = self.model.layers[i](x, cold_data.train_data.edge_index)
            # A_pre A_train^(l-1) e_0
            out_cold += torch_sparse.matmul(pre_edge_index, torch.clone(x))  # type: ignore

        # Normalize
        out_cold /= self.model.nb_layers  # type: ignore
        out_train = self.model.get_embedding(cold_data.train_data)

        # Compute the score
        out = torch.matmul(out_cold[:cold_data.train_data.num_users], out_train[cold_data.train_data.num_users:].T)
        return out[users_id]


class LightGCN_wo_first_embedding_merged(ColdStartModel):
    """ LightGCN without the first embedding layer
    Normal LightGCN with the merged graph

    """

    def __init__(self, model: LightGCN_simple):
        super().__init__()
        self.model = model

    def forward(self, cold_data: ColdStartData, pre_graph: RecGraphData, users_id: torch.Tensor) -> torch.Tensor:
        # Merge the graphs
        edge_index = torch.cat((cold_data.train_data.edge_index, pre_graph.edge_index), dim=1)

        # e_0
        x = self.model.embedding.get_all(cold_data.train_data)
        x[cold_data.cold_user_id] = 0

        out = torch.clone(x)

        for i in range(0, self.model.nb_layers):
            # e_(i+1)
            x = self.model.layers[i](x, edge_index)
            out += torch.clone(x)

        # Normalize
        out /= self.model.nb_layers + 1

        # Compute the score
        out = torch.matmul(out[:cold_data.train_data.num_users], out[cold_data.train_data.num_users:].T)
        return out[users_id]


"""
class LightGCN_first_embedding_mean_instr(ColdStartModel):

    def __init__(self, model: LightGCN_simple):
        super().__init__()
        self.model = model

    def forward(self, cold_data: ColdStartData, pre_graph: RecGraphData, users_id: torch.Tensor) -> torch.Tensor:
        # e_0
        x = self.model.embedding.get_all(cold_data.train_data)

        # compute e_0_cold
        user_df = cold_data.user_df
        cold_user_id = cold_data.cold_user_id.cpu().numpy()
        emb_instrument = compute_mean_embedding(user_df, cold_user_id, x, "instrument_id")

        # e_0_cold
        for u_id in cold_user_id:
            u_instr = user_df[user_df["u"] == u_id]["instrument_id"].values[0]
            x[u_id] = emb_instrument[u_instr]

        # e_0
        out_cold = x

        # e_1_cold
        out_cold += self.model.layers[0](x, pre_graph.edge_index)

        for i in range(0, self.model.nb_layers-1):
            # e_(i+1)_train
            x = self.model.layers[i](x, cold_data.train_data.edge_index)
            # e_(i+2)_cold
            out_cold += self.model.layers[i+1](x, pre_graph.edge_index)

        # Normalize
        out_cold /= self.model.nb_layers + 1
        out_train = self.model.get_embedding(cold_data.train_data)

        # Compute the score
        out = torch.matmul(out_cold[:cold_data.train_data.num_users], out_train[cold_data.train_data.num_users:].T)
        return out[users_id]


class LightGCN_first_embedding_mean_instr_keep_normalization(ColdStartModel):

    def __init__(self, model: LightGCN_simple):
        super().__init__()
        self.model = model

    def forward(self, cold_data: ColdStartData, pre_graph: RecGraphData, users_id: torch.Tensor) -> torch.Tensor:
        # compute the degrees in each graph
        num_nodes = cold_data.train_data.num_users + cold_data.train_data.num_items
        deg_train = compute_degree(cold_data.train_data.edge_index, num_nodes)
        deg_pre = compute_degree(pre_graph.edge_index, num_nodes)

        # update degree of cold users only
        deg_pre[cold_data.train_data.num_users:] = deg_train[cold_data.train_data.num_users:]
        # normalize the adjacency matrix of the pre graph with the new degrees
        pre_edge_index = normalize_adj_matrix(deg_pre, pre_graph.edge_index, num_nodes)

        # e_0
        x = self.model.embedding.get_all(cold_data.train_data)

        # compute e_0_cold
        user_df = cold_data.user_df
        cold_user_id = cold_data.cold_user_id.cpu().numpy()
        emb_instrument = compute_mean_embedding(user_df, cold_user_id, x, on="instrument_id")

        # e_0_cold
        for u_id in cold_user_id:
            u_instr = user_df[user_df["u"] == u_id]["instrument_id"].values[0]
            x[u_id] = emb_instrument[u_instr]
        out_cold = x

        # e_1_cold
        out_cold += torch_sparse.matmul(pre_edge_index, x)

        for i in range(self.model.nb_layers-1):
            # e_(i+1)_train
            x = self.model.layers[i](x, cold_data.train_data.edge_index)
            # e_(i+2)_cold
            out_cold += torch_sparse.matmul(pre_edge_index, x)

        # Normalize
        out_cold /= self.model.nb_layers + 1
        out_train = self.model.get_embedding(cold_data.train_data)

        # compute the score
        out = torch.matmul(out_cold[:cold_data.train_data.num_users], out_train[cold_data.train_data.num_users:].T)
        return out[users_id]


class LightGCN_first_embedding_mean_instr_merged(ColdStartModel):

    def __init__(self, model: LightGCN_simple):
        super().__init__()
        self.model = model

    def forward(self, cold_data: ColdStartData, pre_graph: RecGraphData, users_id: torch.Tensor) -> torch.Tensor:
        # e_0
        x = self.model.embedding.get_all(cold_data.train_data)

        # compute e_0_cold
        user_df = cold_data.user_df
        cold_user_id = cold_data.cold_user_id.cpu().numpy()
        emb_instrument = compute_mean_embedding(user_df, cold_user_id, x, on="instrument_id")

        # e_0_cold
        for u_id in cold_user_id:
            u_instr = user_df[user_df["u"] == u_id]["instrument_id"].values[0]
            x[u_id] = emb_instrument[u_instr]

        # merge train and pre_graph
        edge_index = torch.cat((cold_data.train_data.edge_index, pre_graph.edge_index), dim=1)

        # e_0
        out = self.model.alpha * x
        for i in range(self.model.nb_layers):
            # e_(i+1)
            x = self.model.layers[i](x, edge_index)
            out += self.model.alpha * x

        # compute the score
        out = torch.matmul(out[:cold_data.train_data.num_users], out[cold_data.train_data.num_users:].T)
        return out[users_id]
"""


class LightGCN_first_embedding_mean_instr_level(ColdStartModel):

    def __init__(self, model: LightGCN_simple):
        super().__init__()
        self.model = model

    def forward(self, cold_data: ColdStartData, pre_graph: RecGraphData, users_id: torch.Tensor) -> torch.Tensor:
        # e_0
        x = self.model.embedding.get_all(cold_data.train_data)

        # compute e_0_cold
        user_df = cold_data.user_df
        cold_user_id = cold_data.cold_user_id.cpu().numpy()
        emb_instrument = compute_mean_embedding(user_df, cold_user_id, x, ["instrument_id", "LEVEL"])

        # e_0_cold
        for u_id in cold_user_id:
            u_instr = user_df[user_df["u"] == u_id]["instrument_id"].values[0]
            u_level = user_df[user_df["u"] == u_id]["LEVEL"].values[0]
            x[u_id] = emb_instrument[(u_instr, u_level)]

        # e_0
        out_cold = torch.clone(x)

        # e_1_cold
        out_cold += self.model.layers[0](torch.clone(x), pre_graph.edge_index)

        for i in range(0, self.model.nb_layers-1):
            # e_(i+1)_train
            x = self.model.layers[i](x, cold_data.train_data.edge_index)
            # e_(i+2)_cold
            out_cold += self.model.layers[i+1](torch.clone(x), pre_graph.edge_index)

        # Normalize
        out_cold /= self.model.nb_layers + 1
        out_train = self.model.get_embedding(cold_data.train_data)

        # Compute the score
        out = torch.matmul(out_cold[:cold_data.train_data.num_users], out_train[cold_data.train_data.num_users:].T)
        return out[users_id]


class LightGCN_first_embedding_mean_instr_level_sum_normalization(ColdStartModel):

    def __init__(self, model: LightGCN_simple):
        super().__init__()
        self.model = model

    def forward(self, cold_data: ColdStartData, pre_graph: RecGraphData, users_id: torch.Tensor) -> torch.Tensor:
        # compute the degrees in each graph
        num_nodes = cold_data.train_data.num_users + cold_data.train_data.num_items
        deg_train = compute_degree(cold_data.train_data.edge_index, num_nodes)
        deg_pre = compute_degree(pre_graph.edge_index, num_nodes)

        # update degree of items only
        deg_pre[len(cold_data.user_df):] += deg_train[len(cold_data.user_df):]
        # normalize the adjacency matrix of the pre graph with the new degrees
        pre_edge_index = normalize_adj_matrix(deg_pre, pre_graph.edge_index, num_nodes)

        # e_0
        x = self.model.embedding.get_all(cold_data.train_data)

        # compute e_0_cold
        user_df = cold_data.user_df
        cold_user_id = cold_data.cold_user_id.cpu().numpy()
        emb_instrument = compute_mean_embedding(user_df, cold_user_id, x, on=["instrument_id", "LEVEL"])

        # e_0_cold
        for u_id in cold_user_id:
            u_instr = user_df[user_df["u"] == u_id]["instrument_id"].values[0]
            u_level = user_df[user_df["u"] == u_id]["LEVEL"].values[0]
            x[u_id] = emb_instrument[(u_instr, u_level)]
        out_cold = torch.clone(x)

        # e_1_cold
        out_cold += torch_sparse.matmul(pre_edge_index, torch.clone(x))

        for i in range(self.model.nb_layers-1):
            # e_(i+1)_train
            x = self.model.layers[i](x, cold_data.train_data.edge_index)
            # e_(i+2)_cold
            out_cold += torch_sparse.matmul(pre_edge_index, torch.clone(x))

        # Normalize
        out_cold /= self.model.nb_layers + 1
        out_train = self.model.get_embedding(cold_data.train_data)

        # compute the score
        out = torch.matmul(out_cold[:cold_data.train_data.num_users], out_train[cold_data.train_data.num_users:].T)
        return out[users_id]


class LightGCN_first_embedding_mean_instr_level_keep_normalization(ColdStartModel):

    def __init__(self, model: LightGCN_simple):
        super().__init__()
        self.model = model

    def forward(self, cold_data: ColdStartData, pre_graph: RecGraphData, users_id: torch.Tensor) -> torch.Tensor:
        # compute the degrees in each graph
        num_nodes = cold_data.train_data.num_users + cold_data.train_data.num_items
        deg_train = compute_degree(cold_data.train_data.edge_index, num_nodes)
        deg_pre = compute_degree(pre_graph.edge_index, num_nodes)

        # update degree of items only
        deg_pre[len(cold_data.user_df):] = deg_train[len(cold_data.user_df):]
        # normalize the adjacency matrix of the pre graph with the new degrees
        pre_edge_index = normalize_adj_matrix(deg_pre, pre_graph.edge_index, num_nodes)

        # e_0
        x = self.model.embedding.get_all(cold_data.train_data)

        # compute e_0_cold
        user_df = cold_data.user_df
        cold_user_id = cold_data.cold_user_id.cpu().numpy()
        emb_instrument = compute_mean_embedding(user_df, cold_user_id, x, on=["instrument_id", "LEVEL"])

        # e_0_cold
        for u_id in cold_user_id:
            u_instr = user_df[user_df["u"] == u_id]["instrument_id"].values[0]
            u_level = user_df[user_df["u"] == u_id]["LEVEL"].values[0]
            x[u_id] = emb_instrument[(u_instr, u_level)]
        out_cold = torch.clone(x)

        # e_1_cold
        out_cold += torch_sparse.matmul(pre_edge_index, torch.clone(x))

        for i in range(self.model.nb_layers-1):
            # e_(i+1)_train
            x = self.model.layers[i](x, cold_data.train_data.edge_index)
            # e_(i+2)_cold
            out_cold += torch_sparse.matmul(pre_edge_index, torch.clone(x))

        # Normalize
        out_cold /= self.model.nb_layers + 1
        out_train = self.model.get_embedding(cold_data.train_data)

        # compute the score
        out = torch.matmul(out_cold[:cold_data.train_data.num_users], out_train[cold_data.train_data.num_users:].T)
        return out[users_id]


class LightGCN_first_embedding_mean_instr_level_no_normalization(ColdStartModel):

    def __init__(self, model: LightGCN_simple):
        super().__init__()
        self.model = model

    def forward(self, cold_data: ColdStartData, pre_graph: RecGraphData, users_id: torch.Tensor) -> torch.Tensor:
        # compute the degrees in each graph
        num_nodes = cold_data.train_data.num_users + cold_data.train_data.num_items
        deg_pre = compute_degree(pre_graph.edge_index, num_nodes)

        # update degree of items only
        deg_pre[len(cold_data.user_df):] = 1
        # normalize the adjacency matrix of the pre graph with the new degrees
        pre_edge_index = normalize_adj_matrix(deg_pre, pre_graph.edge_index, num_nodes)

        # e_0
        x = self.model.embedding.get_all(cold_data.train_data)

        # compute e_0_cold
        user_df = cold_data.user_df
        cold_user_id = cold_data.cold_user_id.cpu().numpy()
        emb_instrument = compute_mean_embedding(user_df, cold_user_id, x, on=["instrument_id", "LEVEL"])

        # e_0_cold
        for u_id in cold_user_id:
            u_instr = user_df[user_df["u"] == u_id]["instrument_id"].values[0]
            u_level = user_df[user_df["u"] == u_id]["LEVEL"].values[0]
            x[u_id] = emb_instrument[(u_instr, u_level)]
        out_cold = torch.clone(x)

        # e_1_cold
        out_cold += torch_sparse.matmul(pre_edge_index, torch.clone(x))

        for i in range(self.model.nb_layers-1):
            # e_(i+1)_train
            x = self.model.layers[i](x, cold_data.train_data.edge_index)
            # e_(i+2)_cold
            out_cold += torch_sparse.matmul(pre_edge_index, torch.clone(x))

        # Normalize
        out_cold /= self.model.nb_layers + 1
        out_train = self.model.get_embedding(cold_data.train_data)

        # compute the score
        out = torch.matmul(out_cold[:cold_data.train_data.num_users], out_train[cold_data.train_data.num_users:].T)
        return out[users_id]


class LightGCN_first_embedding_mean_instr_level_merged(ColdStartModel):

    def __init__(self, model: LightGCN_simple):
        super().__init__()
        self.model = model

    def forward(self, cold_data: ColdStartData, pre_graph: RecGraphData, users_id: torch.Tensor) -> torch.Tensor:
        # e_0
        x = self.model.embedding.get_all(cold_data.train_data)

        # compute e_0_cold
        user_df = cold_data.user_df
        cold_user_id = cold_data.cold_user_id.cpu().numpy()
        emb_instrument = compute_mean_embedding(user_df, cold_user_id, x, on=["instrument_id", "LEVEL"])

        # e_0_cold
        for u_id in cold_user_id:
            u_instr = user_df[user_df["u"] == u_id]["instrument_id"].values[0]
            u_level = user_df[user_df["u"] == u_id]["LEVEL"].values[0]
            x[u_id] = emb_instrument[(u_instr, u_level)]

        # merge train and pre_graph
        edge_index = torch.cat((cold_data.train_data.edge_index, pre_graph.edge_index), dim=1)

        # e_0
        out = self.model.alpha * torch.clone(x)
        for i in range(self.model.nb_layers):
            # e_(i+1)
            x = self.model.layers[i](x, edge_index)
            out += self.model.alpha * torch.clone(x)

        # compute the score
        out = torch.matmul(out[:cold_data.train_data.num_users], out[cold_data.train_data.num_users:].T)
        return out[users_id]


class LightGCN_first_embedding_mean_instr_level_test(ColdStartModel):

    def __init__(self, model: LightGCN_simple):
        super().__init__()
        self.model = model

    def forward(self, cold_data: ColdStartData, pre_graph: RecGraphData, users_id: torch.Tensor) -> torch.Tensor:
        # compute the degrees in each graph
        num_nodes = cold_data.train_data.num_users + cold_data.train_data.num_items
        deg_train = compute_degree(cold_data.train_data.edge_index, num_nodes)
        deg_pre = compute_degree(pre_graph.edge_index, num_nodes)

        # update degree of items only
        no_norm_degree = torch.where(torch.arange(0, num_nodes) < len(cold_data.user_df), deg_pre, 1)
        keep_norm_degree = torch.where(torch.arange(0, num_nodes) < len(cold_data.user_df), deg_pre, deg_train)
        sum_norm_degree = deg_pre + deg_train

        # Cold train users
        cold_user_id = cold_data.cold_user_id.cpu().numpy()
        train_user_id = set(list(range(len(cold_data.user_df))))
        train_user_id = train_user_id.difference(set(cold_user_id))
        train_user_id = torch.tensor(list(train_user_id), dtype=torch.int64)

        for deg_norm in [no_norm_degree, keep_norm_degree, sum_norm_degree]:
            # normalize the adjacency matrix of the pre graph with the new degrees
            pre_edge_index = normalize_adj_matrix(deg_norm, pre_graph.edge_index, num_nodes)

            # e_0
            x = self.model.embedding.get_all(cold_data.train_data)

            # compute e_0_cold
            user_df = cold_data.user_df
            emb_instrument = compute_mean_embedding(user_df, cold_user_id, x, on=["instrument_id", "LEVEL"])

            # e_0_cold
            for u_id in cold_user_id:
                u_instr = user_df[user_df["u"] == u_id]["instrument_id"].values[0]
                u_level = user_df[user_df["u"] == u_id]["LEVEL"].values[0]
                if torch.isnan(emb_instrument[(u_instr, u_level)]).any():
                    print(u_instr, u_level)
                    x[u_id] = 0
                else:
                    x[u_id] = emb_instrument[(u_instr, u_level)]
            out_cold = torch.clone(x)

            print("||e_0|| overall:", x.pow(2).sum(-1).mean())
            print("||e_0|| train:", x[train_user_id].pow(2).sum(-1).mean())
            print("||e_0|| cold:", x[cold_user_id].pow(2).sum(-1).mean())

            # e_1_cold
            out_cold += torch_sparse.matmul(pre_edge_index, torch.clone(x))
            print("||e_1|| cold:", out_cold[cold_user_id].pow(2).sum(-1).mean())
            x = self.model.layers[0](x, cold_data.train_data.edge_index)
            print("||e_1|| train:", x[train_user_id].pow(2).sum(-1).mean())

            for i in range(self.model.nb_layers-1):
                # e_(i+1)_train
                x = self.model.layers[i](x, cold_data.train_data.edge_index)
                # e_(i+2)_cold
                out_cold += torch_sparse.matmul(pre_edge_index, torch.clone(x))
                print(f"||e_{i+1}|| train:", x[train_user_id].pow(2).sum(-1).mean())
                print(f"||e_{i+2}|| cold:", x[cold_user_id].pow(2).sum(-1).mean())

            # Normalize
            # out_cold /= self.model.nb_layers + 1
            # out_train = self.model.get_embedding(cold_data.train_data)

            # compute the score
            # out = torch.matmul(out_cold[:cold_data.train_data.num_users], out_train[cold_data.train_data.num_users:].T)
        exit()
