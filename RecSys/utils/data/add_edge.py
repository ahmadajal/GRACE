"""Add edges to the graph"""
import pandas as pd
import torch
from torch_geometric.typing import SparseTensor
from RecSys.utils.data.data import RecGraphData


def add_nothing(data: RecGraphData,
                _users_df: pd.DataFrame,
                _items_df: pd.DataFrame,
                _inter_df: pd.DataFrame) -> RecGraphData:
    r""" Add nothing to the graph """
    return data


def connect_same_musics(data: RecGraphData,
                        users_df: pd.DataFrame,
                        items_df: pd.DataFrame,
                        inter_df: pd.DataFrame,
                        merge_on: "str | list[str]" = ["music_id", "instrument_id"]
                        ) -> RecGraphData:
    r""" Add nodes connecting items with similar attributes

    Args:
        data (RecGraphData): Data object
        users_df (pandas.DataFrame): User dataframe
        items_df (pandas.DataFrame): Item dataframe
        inter_df (pandas.DataFrame): Interactions dataframe
        merge_on (str or list of str): Column(s) to merge on (default: ["music_id", "instrument_id"])

    Returns:
        pandas.DataFrame: Interactions dataframe with added edges
    """
    print("Connecting items with the same", merge_on)

    groups = items_df.groupby(merge_on)

    cur_idx = data.total_num_users + data.num_items
    new_nodes = []
    new_edges = []
    for group in groups:
        if len(group[1]) == 1:
            continue
        new_nodes.append(
            torch.tensor([
                cur_idx,
                group[1]["composer_id"].iloc[0],
                group[1]["music_id"].iloc[0],
                group[1]["acc_type_id"].iloc[0],
                group[1]["style_id"].iloc[0],
                group[1]["level_id"].iloc[0],
                group[1]["instrument_id"].iloc[0]
                ], dtype=torch.int64)
        )
        new_edges.append(torch.tensor(
            [[cur_idx, group[1]["i"].iloc[i]] for i in range(len(group[1]))],
            dtype=torch.int64))
        new_edges.append(torch.tensor(
            [[group[1]["i"].iloc[i], cur_idx] for i in range(len(group[1]))],
            dtype=torch.int64))
    new_nodes = torch.stack(new_nodes, dim=0).to(data.x.device)
    new_edges = torch.cat(new_edges, dim=0).to(data.x.device)

    data.x = torch.cat([data.x, new_nodes], dim=0)
    data.num_other_nodes = data.num_other_nodes + len(new_nodes)

    if isinstance(data.edge_index, SparseTensor):
        row = data.edge_index.storage._row
        col = data.edge_index.storage._col
        values = data.edge_index.storage._value
        indices = torch.stack([row, col], dim=0)
        indices = torch.cat([indices, new_edges.T], dim=1)
        values = torch.cat([values,
                            torch.ones(len(new_edges), dtype=torch.float32, device=data.x.device)],
                           dim=0)
        num_nodes = data.num_users + data.num_items + data.num_other_nodes
        data.edge_index = SparseTensor(
            row=indices[0],
            col=indices[1],
            value=values,
            sparse_sizes=(num_nodes, num_nodes))
    else:
        data.edge_index = torch.cat([data.edge_index, new_edges.T], dim=1)

    print("Number of nodes added:", len(new_nodes))
    print("Number of edges added:", len(new_edges))

    return data


def _connect_same_musics(data: RecGraphData,
                         users_df: pd.DataFrame, items_df: pd.DataFrame, inter_df: pd.DataFrame,
                         delta_level_threshold: int = 99,
                         merge_on: "str | tuple[str, ...]" = ("music_id", "instrument_id")) -> pd.DataFrame:
    r""" Add edges between items with similar attributes

    Args:
        users_df (pandas.DataFrame): User dataframe
        items_df (pandas.DataFrame): Item dataframe
        inter_df (pandas.DataFrame): Interactions dataframe
        delta_level_threshold (int): Threshold for the difference in level between two items to be connected
        merge_on (str or list of str): Column(s) to merge on (default: ["music_id", "instrument_id"])

    Returns:
        pandas.DataFrame: Interactions dataframe with added edges
    """
    print("Adding edges between items with the same", merge_on)
    print("Delta level threshold:", delta_level_threshold)
    # Get items with the same music_id
    pairs = items_df.merge(items_df, on=merge_on, suffixes=("_1", "_2"))

    # Remove self-loops and duplicates
    pairs = pairs[pairs["n_item_id_1"] < pairs["n_item_id_2"]]

    # Remove pairs with a too big difference in level
    pairs = pairs[(pairs["level_id_1"] - pairs["level_id_2"]).abs() <= delta_level_threshold]

    # Add edges
    pairs["TIMESTAMP"] = 1
    pairs["nb_interactions"] = 1
    pairs["nb_interactions_before"] = 1
    pairs["USER_ID"] = pairs["ITEM_ID_1"]
    pairs["ITEM_ID"] = pairs["ITEM_ID_2"]
    pairs["u"] = pairs["n_item_id_1"] + len(users_df)
    pairs["i"] = pairs["n_item_id_2"]
    inter_df = inter_df.append(pairs[inter_df.columns], ignore_index=True)  # type: ignore
    print("Number of edges added:", len(pairs))
    return inter_df


def connect_same_musics_users(_users_df: pd.DataFrame, items_df: pd.DataFrame, inter_df: pd.DataFrame,
                              delta_level_threshold: int = 99,
                              merge_on: "str | tuple[str, ...]" = ("music_id", "instrument_id")) -> pd.DataFrame:
    r""" Add edges between users and items with similar attributes of what they interacted with

    Args:
        users_df (pandas.DataFrame): User dataframe
        items_df (pandas.DataFrame): Item dataframe
        inter_df (pandas.DataFrame): Interactions dataframe
        delta_level_threshold (int): Threshold for the difference in level between two items to be connected
        merge_on (str or list of str): Column(s) to merge on (default: ["music_id", "instrument_id"])

    Returns:
        pandas.DataFrame: Interactions dataframe with added edges
    """
    print("Adding edges between users and items with the same music_id and instrument of what they interacted with")
    print("Delta level threshold:", delta_level_threshold)
    initial_len = len(inter_df)

    # Get items with the same music_id
    inter_df = inter_df.merge(items_df[["ITEM_ID", "i", "music_id", "level_id", "INSTRUMENT", "acc_type_id",
                              "style_id", "composer_id"]], on=["ITEM_ID", "i"], suffixes=("", ""))
    inter_df = inter_df.merge(items_df[["ITEM_ID", "i", "music_id", "level_id", "INSTRUMENT", "acc_type_id", "style_id"]],
                              on=merge_on, suffixes=("", "_2"))

    # Remove pairs with a too big difference in level
    inter_df = inter_df[(inter_df["level_id"] - inter_df["level_id_2"]).abs() <= delta_level_threshold]

    # Add edges
    inter_df = inter_df[["USER_ID", "ITEM_ID_2", "TIMESTAMP", "nb_interactions", "nb_interactions_before", "u", "n_item_id_2"]]
    inter_df = inter_df.rename(columns={"ITEM_ID_2": "ITEM_ID", "n_item_id_2": "i"})
    print("Number of edges added:", len(inter_df) - initial_len)
    return inter_df
