import torch
import pandas as pd
from RecSys.utils.data import data
from tqdm import tqdm


def test_RecGraphData():
    """ Check RecGraphData class """
    users_df = pd.read_csv('data/tomplay/users.csv')
    items_df = pd.read_csv('data/tomplay/items.csv')
    inter_df = pd.read_csv('data/tomplay/split/interactions_train.csv')

    graph = data.RecGraphData(users_df, items_df, inter_df, inter_df)

    # Set of all edges
    set_inter = set()
    for i, j in zip(graph.directed_edge_index[0], graph.directed_edge_index[1]):
        i, j = int(i), int(j)
        set_inter.add((i, j))

    # Check that the number of users and items is correct
    assert graph.num_users == users_df.shape[0]
    assert graph.num_items == items_df.shape[0]

    # Check that features shape are correct
    assert graph.x.size(0) == graph.num_users + graph.num_items
    assert graph.directed_edge_index.shape == (2, inter_df.shape[0])
    assert graph.edge_index.shape == (2, 2*inter_df.shape[0])

    # Check that first edges are users and second edges are items
    assert graph.directed_edge_index[0].min() >= 0
    assert graph.directed_edge_index[0].max() < graph.num_users
    assert graph.directed_edge_index[1].min() >= graph.num_users
    assert graph.directed_edge_index[1].max() < graph.num_users + graph.num_items

    # Check the number of edges
    assert isinstance(graph.edge_attr, torch.Tensor)
    assert graph.edge_attr.size(0) == 2*inter_df.shape[0]

    # Check index
    for (u, i) in zip(graph.directed_edge_index[0, :25], graph.directed_edge_index[1, :25]):
        assert graph.x[u][0] == u
        assert graph.x[i][0] == i
    for (u, i) in zip(graph.edge_index[0, :25], graph.edge_index[1, :25]):
        assert graph.x[u][0] == u
        assert graph.x[i][0] == i
        if u < i:
            assert 0 <= u < graph.num_users
            assert graph.num_users <= i < graph.num_users + graph.num_items
            assert (u.item(), i.item()) in set_inter
        else:
            assert 0 <= i < graph.num_users
            assert graph.num_users <= u < graph.num_users + graph.num_items
            assert (i.item(), u.item()) in set_inter


def test_TrainingGraphDataset():
    """ Check TrainingGraphDataset class"""
    users_df = pd.read_csv('data/tomplay/users.csv')
    items_df = pd.read_csv('data/tomplay/items.csv')
    inter_df = pd.read_csv('data/tomplay/split/interactions_train.csv')
    graph = data.RecGraphData(users_df, items_df, inter_df, inter_df)

    dataset = data.TrainingGraphDataset(graph, nb_neg_sampling=1)

    # Set of all edges
    set_inter = set()
    for i, j in zip(graph.directed_edge_index[0], graph.directed_edge_index[1]):
        i, j = int(i), int(j)
        set_inter.add((i, j))

    # Check that the number of edges is correct
    assert dataset.i.size(0) == dataset.j.size(0) == dataset.k.size(0)
    assert dataset.i.size(0) == inter_df.shape[0]
    assert dataset.k.shape == (dataset.i.size(0),)

    # Check the correct definition of i, j, k
    for i, j, k in zip(dataset.i, dataset.j, dataset.k):
        i, j, k = int(i), int(j), int(k)
        # Check that i is a user and j, k are items
        assert 0 <= i < graph.num_users
        assert graph.num_users <= j < graph.num_users + graph.num_items
        assert graph.num_users <= k < graph.num_users + graph.num_items
        # Check that j is a positive interaction and k is a negative interaction
        assert (i, j) in set_inter
        assert (i, k) not in set_inter

    dataset = data.TrainingGraphDataset(graph, nb_neg_sampling=2)
    dataset.compute_neg_items()

    # Check that the number of edges is correct
    assert dataset.i.size(0) == dataset.j.size(0) == dataset.k.size(0)
    assert dataset.i.size(0) == inter_df.shape[0]
    assert dataset.k.shape == (dataset.i.size(0), 2)

    # Check the correct definition of i, j, k
    for i, j, k in zip(dataset.i, dataset.j, dataset.k):
        i, j, k0, k1 = int(i), int(j), int(k[0]), int(k[1])
        # Check that i is a user and j, k are items
        assert 0 <= i < graph.num_users
        assert graph.num_users <= j < graph.num_users + graph.num_items
        assert graph.num_users <= k0 < graph.num_users + graph.num_items
        assert graph.num_users <= k1 < graph.num_users + graph.num_items
        # Check that j is a positive interaction and k is a negative interaction
        assert (i, j) in set_inter
        assert (i, k0) not in set_inter
        assert (i, k1) not in set_inter


def test_dataloader():
    """ Check that the dataloader is working correctly """
    users_df = pd.read_csv('data/tomplay/users.csv')
    items_df = pd.read_csv('data/tomplay/items.csv')
    inter_df = pd.read_csv('data/tomplay/split/interactions_train.csv')
    graph = data.RecGraphData(users_df, items_df, inter_df, inter_df)

    dataset = data.TrainingGraphDataset(graph)
    dataloader = data.get_DataLoader(dataset, shuffle=False, batch_size=len(dataset))

    # Check that the sampled neg items are different each epochs
    dataloader.dataset.compute_neg_items()  # type: ignore
    _, _, k1 = next(iter(dataloader))

    dataloader.dataset.compute_neg_items()  # type: ignore
    _, _, k2 = next(iter(dataloader))

    assert (k1 != k2).any()


def test_ValidDataset_adj_matrix():
    """ Check that the adjacency matrix is correctly computed """
    users_df = pd.read_csv('data/tomplay/users.csv')
    items_df = pd.read_csv('data/tomplay/items.csv')
    train_inter_df = pd.read_csv('data/tomplay/split/interactions_train.csv')
    valid_inter_df = pd.read_csv('data/tomplay/split/interactions_val.csv')

    train_graph = data.RecGraphData(users_df, items_df, train_inter_df, train_inter_df)
    valid_graph = data.RecGraphData(users_df, items_df, valid_inter_df, valid_inter_df)

    val_ds = data.TestDataset(train_graph, valid_graph)
    val_ds.compute_adj_matrix()

    # Set of all edges
    set_train_inter = set()
    for i, j in zip(train_graph.directed_edge_index[0], train_graph.directed_edge_index[1]):
        i, j = int(i), int(j)
        set_train_inter.add((i, j))
    set_val_inter = set()
    for i, j in zip(valid_graph.directed_edge_index[0], valid_graph.directed_edge_index[1]):
        i, j = int(i), int(j)
        set_val_inter.add((i, j))

    assert val_ds.val_adj_matrix.shape == (train_graph.num_users, train_graph.num_items)

    for i in tqdm(range(25)):
        for j in range(train_graph.num_items):
            if val_ds.train_adj_matrix[i, j]:
                assert (i, j+train_graph.num_users) in set_train_inter
            else:
                assert (i, j+train_graph.num_users) not in set_train_inter

    for i in range(25):
        for j in tqdm(range(train_graph.num_items)):
            if val_ds.val_adj_matrix[i, j]:
                assert (i, j+train_graph.num_users) in set_val_inter
            else:
                assert (i, j+train_graph.num_users) not in set_val_inter
