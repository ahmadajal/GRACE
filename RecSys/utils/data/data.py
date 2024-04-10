""" Data utils """
from typing import Union
from copy import deepcopy
import pandas as pd
import numpy as np
import torch
import torch.utils.data
import torch_sparse


def get_node_features(user_df: Union[pd.DataFrame, None],
                      item_df: Union[pd.DataFrame, None],
                      inter_df: pd.DataFrame):
    r""" Get node features from dataframes

    For the Tomplay dataset, the user features are:
        - the user id
        - the level
        - the instrument
        - the timestamp of first interaction

    For the other dataset, the user features are:
        - the user id

    Args:
        user_df (pd.DataFrame | None):   user dataframe if available
        item_df (pd.DataFrame | None):   item dataframe if available
        inter_df (pd.DataFrame):   interaction dataframe

    """
    if user_df is None:
        num_users = int(inter_df["u"].max() + 1)
        assert isinstance(num_users, int)
        user_df = pd.DataFrame({"u": list(range(num_users))})
    if item_df is None:
        num_items = int(inter_df["i"].max() + 1)
        assert isinstance(num_items, int)
        item_df = pd.DataFrame({"i": list(range(num_items))})

    user_df.sort_values("u", inplace=True)
    item_df.sort_values("i", inplace=True)
    total_num_users = inter_df["u"].max() + 1

    # User features
    if "LEVEL" in user_df.columns and "instrument_id" in user_df.columns and "first_interaction" in user_df.columns:  # Specific to the Tomplay dataset
        x_user = torch.stack(
            (
                torch.tensor(user_df["u"].values, dtype=torch.int64),
                torch.tensor(user_df["LEVEL"].values, dtype=torch.int64),
                torch.tensor(user_df["instrument_id"].values, dtype=torch.int64),
                torch.tensor(user_df["first_interaction"].values, dtype=torch.int64),
                torch.zeros(user_df.shape[0], dtype=torch.int64),
                torch.zeros(user_df.shape[0], dtype=torch.int64),
                torch.zeros(user_df.shape[0], dtype=torch.int64),
            ),
            dim=-1
        )
    else:
        x_user = torch.tensor(user_df["u"].values, dtype=torch.int64).unsqueeze(-1)

    # Item features
    if len({"composer_id", "music_id", "acc_type_id", "style_id", "level_id", "instrument_id"}.intersection(item_df.columns)) == 6:  # Specific to the Tomplay dataset
        x_item = torch.stack(
            (
                torch.tensor(item_df["i"].to_numpy(), dtype=torch.int64) + total_num_users,
                torch.tensor(item_df["composer_id"].to_numpy(), dtype=torch.int64),
                torch.tensor(item_df["music_id"].to_numpy(), dtype=torch.int64),
                torch.tensor(item_df["acc_type_id"].to_numpy(), dtype=torch.int64),
                torch.tensor(item_df["style_id"].to_numpy(), dtype=torch.int64),
                torch.tensor(item_df["level_id"].to_numpy(), dtype=torch.int64),
                torch.tensor(item_df["instrument_id"].to_numpy(), dtype=torch.int64),
            ),
            dim=-1
        )
    else:
        x_item = torch.tensor(item_df["i"].to_numpy(), dtype=torch.int64).unsqueeze(-1) + total_num_users

    return torch.cat((x_user, x_item), 0)


class RecGraphData:
    r""" Graph data for recommender systems

    Attributes:
        x (Tensor): the node features
        directed_edge_index (Tensor): the directed edge index (2, num_edges)
        edge_index (Tensor): the undirected edge index and potential edges added (2, num_edges)
        edge_attr (Tensor | None): if provided, the edge attributes are the timestamps of the interactions
        num_users (int): the number of users
        num_items (int): the number of items
        total_num_users (int): the total number of users in the dataset
    """

    def __init__(self,
                 user_df: Union[pd.DataFrame, None],
                 item_df: Union[pd.DataFrame, None],
                 inter_df: pd.DataFrame,
                 total_num_users: Union[int, None] = None):
        r"""
        Create a graph data object for the recommender system

        Args:
            user_df (pd.DataFrame | None): user dataframe if available
            item_df (pd.DataFrame | None): item dataframe if available
            inter_df (pd.DataFrame): interaction dataframe
            inter_df_augmented (pd.DataFrame)(optional): interaction dataframe with possible additional edges
            total_num_users (int)(optional): total number of users in the dataset

        Returns:
            recomender system graph data
        """
        if user_df is None:
            num_users = int(inter_df["u"].max() + 1)
        else:
            num_users = int(user_df.shape[0])
        if item_df is None:
            num_items = int(inter_df["i"].max() + 1)
        else:
            num_items = int(item_df.shape[0])
        assert isinstance(num_users, int)
        assert isinstance(num_items, int)

        if total_num_users is None:
            total_num_users = num_users

        # Features
        self.x = get_node_features(user_df, item_df, inter_df.copy())

        # Original edges
        src = torch.tensor(inter_df["u"].to_numpy(), dtype=torch.long)
        dest = torch.tensor(inter_df["i"].to_numpy(), dtype=torch.long) + total_num_users
        directed_edge_index = torch.stack([src, dest], dim=0)
        directed_edge_index, directed_counts = torch.unique(directed_edge_index, return_counts=True, dim=1)

        # All edges
        src = torch.tensor(inter_df["u"].to_numpy(), dtype=torch.long)
        dest = torch.tensor(inter_df["i"].to_numpy(), dtype=torch.long) + total_num_users
        edge_index = torch.stack([src, dest], dim=0)

        # Edge features
        if "t" in inter_df.columns:
            edge_attr = torch.tensor(inter_df["t"].to_numpy(), dtype=torch.int64)
        else:
            edge_attr = None

        # Transform to undirected
        self.edge_index = torch.cat((edge_index, torch.stack((dest, src), dim=0)), dim=-1).contiguous()
        if edge_attr is not None:
            self.edge_attr = torch.cat((edge_attr, edge_attr), dim=0).contiguous()
        else:
            self.edge_attr = None

        # Graph metadata
        self.num_users = num_users
        self.total_num_users = total_num_users
        self.num_items = num_items
        self.directed_edge_index = directed_edge_index
        self.num_other_nodes = 0

    def to(self, device) -> "RecGraphData":
        r""" Move data to device

        Args:
            device (torch.device): device to move data to

        Returns:
            RecGraphData: data on device

        """
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        if self.edge_attr is not None:
            self.edge_attr = self.edge_attr.to(device)
        self.directed_edge_index = self.directed_edge_index.to(device)
        return self

    def subgraph(self, node_idx) -> "RecGraphData":
        r""" Get subgraph of the graph data induced by the nodes in node_idx

        Args:
            node_idx (torch.tensor): node index to keep

        Returns:
            RecGraphData: subgraph
        """
        # Get subgraph
        subgraph = deepcopy(self)

        subgraph.x = subgraph.x[node_idx]

        mask = torch.isin(subgraph.directed_edge_index[0], node_idx) | torch.isin(subgraph.directed_edge_index[1], node_idx)
        subgraph.directed_edge_index = subgraph.directed_edge_index[:, mask]

        if isinstance(subgraph.edge_index, torch.Tensor):
            mask = torch.isin(subgraph.edge_index[0], node_idx) | torch.isin(subgraph.edge_index[1], node_idx)
            subgraph.edge_index = subgraph.directed_edge_index[:, mask]
            if subgraph.edge_attr is not None:
                subgraph.edge_attr = subgraph.edge_attr[:, mask]
        elif isinstance(subgraph.edge_index, torch_sparse.SparseTensor):
            subgraph.edge_index = subgraph.edge_index.coalesce()[node_idx, node_idx]
        else:
            print(type(subgraph.edge_index))
            raise ValueError

        subgraph.num_users = int(torch.sum(node_idx < self.num_users))
        subgraph.num_items = int(torch.sum(node_idx >= self.num_users))
        subgraph.total_num_users = self.total_num_users

        return subgraph


class TrainingGraphDataset(torch.utils.data.Dataset):
    """ Training Dataset

    Sample negative items for each positive items
    """

    def __init__(self, graph: RecGraphData, nb_neg_sampling=1):
        """
        Args:
            graph (RecGraphData): graph data
            nb_neg_sampling (int): number of negative items to sample for each positive item
        """
        self.graph = graph
        self.nb_neg_sampling = nb_neg_sampling
        self.compute_neg_items()

    def compute_neg_items_1(self):
        """ Compute items for negative sampling
        Based on the implementation of torch_geometric.utils.negative_sampling
        """
        i, j = self.graph.directed_edge_index
        i, j = i.cpu().numpy(), j.cpu().numpy()
        edges_set = set([(ui, uj) for (ui, uj) in zip(i, j)])
        k = np.random.randint(
            self.graph.num_users,
            self.graph.num_users + self.graph.num_items,
            (len(i),))

        @np.vectorize
        def not_valid_mask(u_i, u_k):
            return (u_i, u_k) in edges_set

        mask = not_valid_mask(i, k)
        rest = mask.nonzero()[0]
        while rest.shape[0] > 0:
            k[rest] = torch.randint(
                self.graph.num_users,
                self.graph.num_users + self.graph.num_items,
                (rest.shape[0],))
            mask = not_valid_mask(i[rest], k[rest])
            rest = rest[mask]

        # Store items
        i = torch.tensor(i, dtype=torch.int64, device=self.graph.x.device)
        j = torch.tensor(j, dtype=torch.int64, device=self.graph.x.device)
        k = torch.tensor(k, dtype=torch.int64, device=self.graph.x.device)
        return i, j, k

    def compute_neg_items(self):
        """ Compute items for negative sampling """
        if self.nb_neg_sampling == 1:
            self.i, self.j, self.k = self.compute_neg_items_1()
        else:
            all_k = []
            for _ in range(self.nb_neg_sampling):
                i, j, k = self.compute_neg_items_1()
                all_k.append(k)
            self.i = i  # type: ignore
            self.j = j  # type: ignore
            self.k = torch.stack(all_k, -1)

    def __len__(self):
        i = self.graph.directed_edge_index[0]
        return i.shape[0]

    def __getitem__(self, idx):
        return self.i[idx], self.j[idx], self.k[idx]


class TestDataset(torch.utils.data.Dataset):
    """ Test Dataset

    Compute the train and test adjacency matrix
    """

    def __init__(self, train_graph: RecGraphData, val_graph: RecGraphData):
        """
        Args:
            train_graph (RecGraphData): Training graph
            val_graph (RecGraphData): Validation graph
        """
        self.train_graph = train_graph
        self.val_graph = val_graph
        self.train_adj_matrix = None
        self.val_adj_matrix = None
        self.first_val_interactions = None

    def __len__(self):
        return self.val_graph.num_users

    def compute_adj_matrix(self):
        """ Compute the adjacency matrix for train and test """
        train_adj_matrix = torch.zeros((self.train_graph.num_users, self.train_graph.num_items), dtype=torch.bool)
        train_adj_matrix[self.train_graph.directed_edge_index[0], self.train_graph.directed_edge_index[1] - self.train_graph.num_users] = True
        self.train_adj_matrix = train_adj_matrix

        val_adj_matrix = torch.zeros((self.val_graph.num_users, self.val_graph.num_items), dtype=torch.bool)
        val_adj_matrix[self.val_graph.directed_edge_index[0], self.val_graph.directed_edge_index[1] - self.val_graph.num_users] = True
        self.val_adj_matrix = val_adj_matrix

    def compute_first_test_interactions(self):
        """ Compute the first interaction for each user in the test set """
        assert self.val_graph.edge_attr is not None, "Timestamps are not available for this dataset"
        self.first_val_interactions = torch.zeros((self.val_graph.num_users, 1), dtype=torch.long)
        for u in range(self.val_graph.num_users):
            timestamps_u = self.val_graph.edge_attr[self.val_graph.edge_index[0] == u]
            items_u = self.val_graph.edge_index[1][self.val_graph.edge_index[0] == u] - self.val_graph.num_users
            argmin_timestamp = torch.argmin(timestamps_u)
            argmin_item = items_u[argmin_timestamp]
            self.first_val_interactions[u, 0] = argmin_item.to(torch.long)

    def __getitem__(self, idx):
        assert self.val_adj_matrix is not None, "Adjacency matrix not computed"
        assert self.train_adj_matrix is not None, "Adjacency matrix not computed"
        assert self.first_val_interactions is not None, "First test interactions not computed"
        return idx, self.val_adj_matrix[idx], self.train_adj_matrix[idx], self.first_val_interactions[idx]


def get_DataLoader(ds, batch_size, shuffle=True, num_workers=0):
    """ Alias for torch.utils.data.DataLoader """
    return torch.utils.data.DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
