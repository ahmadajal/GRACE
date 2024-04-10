import pandas as pd
import numpy as np
from RecSys.cold_start.data import ColdStartData
from RecSys.utils.config import Experiment, load_everything_from_exp_cold


def test_cold_data():
    """ Check ColdStartData class"""
    user_df = pd.read_csv("data/tomplay/users.csv")
    item_df = pd.read_csv("data/tomplay/items.csv")
    inter_df = pd.read_csv("data/tomplay/interactions.csv")

    cold_user_id = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    data = ColdStartData(user_df, item_df, inter_df, inter_df, cold_user_id)

    # Check attributes
    assert data.device == "cpu"

    assert data.cold_user_id.shape == (10,)
    assert data.cold_user_df.shape == (10, user_df.shape[1])
    assert data.cold_inter_df.shape == (len(inter_df[(0 <= inter_df.n_user_id) & (inter_df.n_user_id < 10)]), inter_df.shape[1])

    assert data.train_user_df.shape == (user_df.shape[0] - 10, user_df.shape[1])
    assert data.train_inter_df.shape == (len(inter_df[inter_df.n_user_id >= 10]), inter_df.shape[1])

    assert data.train_data.total_num_users == user_df.shape[0]
    assert data.train_data.num_users == user_df.shape[0]
    assert data.train_data.num_items == item_df.shape[0]

    # Check that train data does not contain cold users
    for u in data.train_data.directed_edge_index[0]:
        u = u.cpu().item()
        assert u < len(user_df)
        assert u not in cold_user_id

    # Check that cold data contains only cold users
    for u in data.cold_data.directed_edge_index[0]:
        u = u.cpu().item()
        assert u < len(user_df)
        assert u in cold_user_id


def test_get_test_data():
    """ Check get_test_data method """
    user_df = pd.read_csv("data/tomplay/users.csv")
    item_df = pd.read_csv("data/tomplay/items.csv")
    inter_df = pd.read_csv("data/tomplay/interactions.csv")

    cold_user_id = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    data = ColdStartData(user_df, item_df, inter_df, inter_df, cold_user_id)

    pre_data, post_data, test_ds, user_id_enough = data.get_test_data(2)
    user_id_enough = set([int(u.cpu().item()) for u in user_id_enough])

    test_ds.compute_adj_matrix()
    val_adj_matrix = test_ds.val_adj_matrix

    # Check that pre_data and post_data have the right number of interactions per user
    for user_id in range(10):
        if len(inter_df[inter_df["u"] == user_id].sort_values("t").iloc[2:]) > 2 + 25:
            assert (pre_data.directed_edge_index[0] == user_id).sum() == 2
            assert (post_data.directed_edge_index[0] == user_id).sum() == 25
            assert user_id in user_id_enough
        else:
            assert (pre_data.directed_edge_index[0] == user_id).sum() == 0
            assert (post_data.directed_edge_index[0] == user_id).sum() == 0
            assert user_id not in user_id_enough

    for user_id in user_df.n_user_id.unique():
        if user_id >= 10:
            assert val_adj_matrix[user_id].sum() == 0

    # for user_id in data.cold_user_id:
    #     row = val_adj_matrix[user_id]
    #     user_id = user_id.cpu().item()
    #     assert row.sum() == len(inter_df[inter_df["u"] == user_id].sort_values("TIMESTAMP").iloc[2:].drop_duplicates(["u", "i"]))


def test_load_everything_form_exp_cold():
    config = {
        # Default config
        "name": "test",

        # Type
        "type": "GNN",

        # Dataset
        "dataset": "tomplay",
        "edge_weighting": "ones",
        "add_edges": "add_nothing",

        # Model type
        "model": "LightGCN_simple",
        "embedding_type": "IdEmbeddingPlusNameEmbedding",
        "embedding_dim": 64,
        "num_layers": 2,

        # Training parameters
        "epochs": 250,
        "lr": 0.0001,
        "batch_size": 1024,
        "weight_decay": 0,
        "lr_decay": 0.95,
        "loss": "BPR"
    }
    exp = Experiment(config)

    datas, model = load_everything_from_exp_cold(exp=exp, device="cpu")
    (cold_data, train_graph, val_graph, train_ds, val_ds) = datas
    (model, optimizer, scheduler, loss_fn) = model

    cold_user_id = set(cold_data.cold_user_id.numpy())

    for u_id in train_graph.directed_edge_index[0]:
        u_id = u_id.cpu().item()
        assert u_id < len(cold_data.user_df)
        assert u_id not in cold_user_id

    for u_id in val_graph.directed_edge_index[0]:
        u_id = u_id.cpu().item()
        assert u_id < len(cold_data.user_df)
        assert u_id not in cold_user_id
