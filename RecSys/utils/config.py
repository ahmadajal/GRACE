"""Configuration utils."""
import os
import numpy as np
import torch
import yaml
import pandas as pd

# Graph utils
from RecSys.utils.data import data
from RecSys.utils.data import add_edge
from RecSys.utils.data import edge_weighting

# Training utils
from RecSys.nn.training import BPRLoss
from RecSys.nn.training import CELoss

# Models
from RecSys.nn.models.embeddings import IdEmbedding, IdEmbeddingPlusNameEmbedding, IdEmbeddingPlusNameEmbedding2
from RecSys.nn.models.embeddings import UsersFeaturesEmbeddingPlusNameEmbdding, UsersFeaturesAndIdEmbeddingPlusNameEmbedding
from RecSys.nn.models.LightGCN import LightGCN_simple, LightGCN_wo_first_emb
from RecSys.nn.models.similarity import Item2ItemCosineSimilarity, User2UserCosineSimilarity
from RecSys.nn.models.LightGCN_interpretable import TopKUsersItemsLightGCN


def get_default_config():
    """Get default configuration."""
    with open("../config/default_config.yaml", 'r', encoding="utf-8") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    return config


def get_config(path):
    r"""Get configuration from path.

    Args:
        path (str): Path to the configuration file.
    """
    with open(path, 'r', encoding="utf-8") as config_file:
        user_config = yaml.load(config_file, Loader=yaml.FullLoader)
    config = get_default_config()
    config.update(user_config)
    return config


class Experiment:
    r"""Experiment class.

    This class is used to store the configuration and the results of an experiment.
    """

    def __init__(self, config, res_path="RecSys/config/res/"):
        r"""Experiment class.

        Args:
            config (dict): Configuration of the experiment.
            res_path (str, optional): Path to the results. Defaults to "RecSys/LightGCN/config/res/".
        """
        self.config = config
        self.exp = {"Config": config, "Results": {"Epoch": {}}}
        self.res_path = os.path.join(res_path, config["name"])
        self.model_path = os.path.join(self.res_path, "trained_models")
        if not os.path.exists(self.res_path):
            os.makedirs(self.res_path)
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

    def __repr__(self):
        rep = ""
        for key, value in self.config.items():
            rep += f"{key}: {value}\n"
        return rep[:-2]

    def __getitem__(self, key):
        return self.config[key]

    def __setitem__(self, key, value):
        self.config[key] = value

    def on_epoch_end(self, epoch, metrics):
        r""" Save the results of the epoch.

        Args:
            epoch (int): Epoch number.
            metrics (list): List of metrics.
                Metrics are objects that have a name and a compute method.
        """
        if epoch not in self.exp["Results"]["Epoch"]:
            self.exp["Results"]["Epoch"][epoch] = dict([
                (met.name, met.compute().item()) for met in metrics
            ])
        else:
            for met in metrics:
                self.exp["Results"]["Epoch"][epoch][met.name] = met.compute().item()

    def save(self):
        r""" Save the results of the experiment. """
        print("Saving results in:", os.path.join(self.res_path, "results.yaml"))
        with open(os.path.join(self.res_path, "results.yaml"), "w", encoding="utf-8") as config_file:
            yaml.dump(self.exp, config_file)

    def save_model(self, model: torch.nn.Module):
        r""" Save the model if it is the best one.

        The best model is the one that has the best Precision@25 on the validation set.
        """
        epochs = sorted(list(self.exp["Results"]["Epoch"]))
        # metrics = list(self.exp["Results"]["Epoch"][epochs[-1]])
        metrics = ["val_Rec@25_ne"]
        for met in metrics:
            if met[:4] != "val_":
                continue
            previous_m = []
            for epoch in epochs[:-1]:
                if met in self.exp["Results"]["Epoch"][epoch]:
                    previous_m.append(self.exp["Results"]["Epoch"][epoch][met])
            if self.exp["Results"]["Epoch"][epochs[-1]][met] > max(previous_m, default=-1000):
                save_path = os.path.join(self.model_path, "best_"+met+".pt")
                torch.save(model.state_dict(), save_path)


def get_results(exp: Experiment):
    """ Extract the results of an experiment """
    epochs = list(exp["Results"]["Epoch"].keys())
    metrics = dict()
    for epoch in epochs:
        for met in exp["Results"]["Epoch"][epoch]:
            if met not in metrics:
                metrics[met] = [[], []]
            metrics[met][0].append(epoch)
            metrics[met][1].append(exp["Results"]["Epoch"][epoch][met])
    return metrics


def load_data(dataset, test=False):
    r""" Load the data of a dataset.

    Args:
        dataset (str): Name of the dataset.
            Dataset must be in the data folder.
            interactions_train.csv, interactions_val.csv and interactions_test.csv must be in the split folder.
        test (bool, optional): If True, the test set is loaded. Defaults to False.
    """
    # Users
    if os.path.exists(f"../../data/{dataset}/users.csv"):
        users_df = pd.read_csv(f"../../data/{dataset}/users.csv")
    else:
        users_df = None

    # Items
    if os.path.exists(f"../../data/{dataset}/items.csv"):
        items_df = pd.read_csv(f"../../data/{dataset}/items.csv")
    else:
        items_df = None

    # Interactions
    inter_df = pd.read_csv(f"../../data/{dataset}/split/interactions_train.csv")
    val_inter_df = pd.read_csv(f"../../data/{dataset}/split/interactions_val.csv")
    if test:
        test_inter_df = pd.read_csv(f"../../data/{dataset}/split/interactions_test.csv")
        inter_df = pd.concat([inter_df, val_inter_df])
        val_inter_df = test_inter_df
    return users_df, items_df, inter_df, val_inter_df


def load_everything_from_exp(exp, device, test=False):
    r""" Load everything from an experiment

    Args:
        exp (Experiment): Experiment to load.
        device (torch.device): Device to use.
        test (bool, optional): If True, load the test set. Defaults to False.

    Returns:
        tuple: (graph, val_graph, train_ds, val_ds), (model, optimizer, scheduler, loss)
    """

    # Hyperparameters
    embedding_dim = exp["embedding_dim"]
    learning_rate = exp["lr"]
    weight_decay = exp["weight_decay"]
    num_layers = exp["num_layers"]

    # Type of model
    if exp["model"] in ["LightGCN_simple", "NGCF", "LightGCN_wo_first_emb"]:
        exp["type"] = "GNN"
    elif exp["model"] in ["MF", "NCF", "User2UeserCosineSimilarity", "Item2ItemCosineSimilarity", "LightGCN_interpretable"]:
        exp["type"] = "CF"
    else:
        raise ValueError("Model not recognized")

    if exp["type"] == "CF":
        dropout = exp["dropout"]
        assert isinstance(dropout, (float, int))
    else:
        dropout = None

    # Load data
    users_df, items_df, inter_df, val_inter_df = load_data(exp["dataset"], test=test)

    graph = data.RecGraphData(users_df, items_df, inter_df).to(device)  # type: ignore
    val_graph = data.RecGraphData(users_df, items_df, val_inter_df).to(device)

    # Edge weights
    if exp["type"] == "GNN" or exp["model"] == "LightGCN_interpretable":
        if exp["edge_weighting"] == "ones":
            edge_weighting.edge_weights_ones(graph)
        elif exp["edge_weighting"] == "num_interaction":
            edge_weighting.edge_weights_num_interaction(graph)
        elif exp["edge_weighting"] == "exp_time_since_last_inter":
            edge_weighting.edge_weights_exp_time_since_last_inter(graph, exp["edge_weighting_gamma"], exp["edge_weighting_only_last"])
        elif exp["edge_weighting"] == "exp_time":
            edge_weighting.edge_weights_exp_time(graph, exp["edge_weighting_gamma"], exp["edge_weighting_only_last"])
        else:
            raise ValueError("Edge weighting not recognized")
    edge_weight = graph.edge_index.storage.value()
    print("Training graph weights:")
    print("\tmax", torch.max(edge_weight).item())
    print("\tmin", torch.min(edge_weight).item())
    print("\tmean", torch.mean(edge_weight).item())
    print("\tmedian", torch.median(edge_weight).item())

    # Add edges
    if exp["type"] == "GNN":
        if exp["add_edges"] == "add_nothing":
            pass
        elif exp["add_edges"] == "connect_same_musics":
            assert users_df is not None, "No user features provided"
            assert items_df is not None, "No item features provided"
            if "merge_on" in exp.config:
                merge_on = list(exp["merge_on"])
            else:
                merge_on = "music_id"
            graph = add_edge.connect_same_musics(graph, users_df, items_df, inter_df, merge_on)
            val_graph = add_edge.connect_same_musics(val_graph, users_df, items_df, inter_df, merge_on)
        else:
            raise ValueError("Method to add edges not recognized")

    train_ds = data.TrainingGraphDataset(graph)
    val_ds = data.TestDataset(graph, val_graph)

    # Embedding
    if exp["embedding_type"] == "IdEmbedding":
        embedding = IdEmbedding.from_graph(graph, embedding_dim)
    elif exp["embedding_type"] == "IdEmbeddingPlusNameEmbedding":
        embedding = IdEmbeddingPlusNameEmbedding.from_graph(graph, embedding_dim)
    elif exp["embedding_type"] == "IdEmbeddingPlusNameEmbedding2":
        embedding = IdEmbeddingPlusNameEmbedding2.from_graph(graph, embedding_dim)
    elif exp["embedding_type"] == "UsersFeaturesEmbeddingPlusNameEmbdding":
        embedding = UsersFeaturesEmbeddingPlusNameEmbdding.from_graph(graph, embedding_dim)
    elif exp["embedding_type"] == "UsersFeaturesAndIdEmbeddingPlusNameEmbedding":
        embedding = UsersFeaturesAndIdEmbeddingPlusNameEmbedding.from_graph(graph, embedding_dim)
    # elif exp["embedding_type"] == "ItemsFeaturesEmbedding_plus_name_emb":
    #     embedding = ItemsFeaturesEmbedding_plus_name_emb.from_graph(graph, embedding_dim)
    # elif exp["embedding_type"] == "AllFeaturesEmbedding_plus_name_emb":
    #     embedding = AllFeaturesEmbedding_plus_name_emb.from_graph(graph, embedding_dim)
    else:
        raise ValueError("Embedding not recognized")

    # Model
    if exp["model"] == "LightGCN_simple":
        model = LightGCN_simple(nb_layers=num_layers, embedding=embedding).to(device)
    elif exp["model"] == "LightGCN_wo_first_emb":
        model = LightGCN_wo_first_emb(nb_layers=num_layers, embedding=embedding).to(device)
    elif exp["model"] == "Item2ItemCosineSimilarity":
        model = Item2ItemCosineSimilarity(graph=graph).to(device)
    elif exp["model"] == "User2UserCosineSimilarity":
        model = User2UserCosineSimilarity(graph=graph).to(device)
    elif exp["model"] == "LightGCN_interpretable":
        model = TopKUsersItemsLightGCN(embedding=embedding, ku=exp["top_users"], ki=exp["top_items"]).to(device)
    else:
        raise ValueError("Model not found")

    # Training parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=exp["lr_decay"])  # type: ignore

    if exp["loss"] == "BPR":
        loss_fn = BPRLoss(weight_decay=weight_decay)
    elif exp["loss"] == "CE":
        loss_fn = CELoss(weight_decay=weight_decay)
        train_ds.nb_neg_sampling = exp["nb_neg_sampling"]
    else:
        raise ValueError("Loss not found")

    datas = (graph, val_graph, train_ds, val_ds)
    model = (model, optimizer, scheduler, loss_fn)
    return datas, model


    """ Load everything from an experiment config for a cold start evaluation """
    import RecSys.cold_start.data as cold_start_data
    # Hyperparameters
    embedding_dim = exp["embedding_dim"]
    learning_rate = exp["lr"]
    weight_decay = exp["weight_decay"]
    num_layers = exp["num_layers"]

    # Type of model
    if exp["model"] in ["LightGCN_simple", "NGCF", "LightGCN_wo_first_emb"]:
        exp["type"] = "GNN"
    elif exp["model"] in ["MF", "NCF", "User2UeserCosineSimilarity", "Item2ItemCosineSimilarity", "LightGCN_interpretable"]:
        exp["type"] = "CF"
    else:
        raise ValueError("Model not recognized")

    if exp["type"] == "CF":
        dropout = exp["dropout"]
    else:
        dropout = None

    # Load data
    users_df = pd.read_csv('data/tomplay/users.csv')
    items_df = pd.read_csv('data/tomplay/items.csv')
    inter_df = pd.read_csv("data/tomplay/interactions_renamed.csv")
    train_inter_df = pd.read_csv("data/tomplay/users_split/interactions_train.csv")
    val_inter_df = pd.read_csv("data/tomplay/users_split/interactions_val.csv")
    if test:
        test_inter_df = pd.read_csv("data/tomplay/users_split/interactions_test.csv")
        train_inter_df = pd.concat([train_inter_df, val_inter_df])
        val_inter_df = test_inter_df

    val_users_ids = np.load("data/tomplay/users_split/users_test.npy")
    test_users_ids = np.load("data/tomplay/users_split/users_val.npy")
    val_users_ids = np.concatenate([val_users_ids, test_users_ids], axis=0)
    print(len(val_users_ids) / len(users_df))

    graph = data.RecGraphData(users_df, items_df, train_inter_df).to(device)  # type: ignore
    val_graph = data.RecGraphData(users_df, items_df, val_inter_df).to(device)

    train_ds = data.TrainingGraphDataset(graph)
    val_ds = data.TestDataset(graph, val_graph)

    cold_data = cold_start_data.ColdStartData(users_df, items_df, inter_df, val_users_ids).to(device)  # type: ignore

    # Edge weights
    if exp["type"] == "GNN":
        if exp["edge_weighting"] == "ones":
            edge_weighting.edge_weights_ones(graph)
        elif exp["edge_weighting"] == "num_interaction":
            edge_weighting.edge_weights_num_interaction(graph)
        elif exp["edge_weighting"] == "exp_time_since_last_inter":
            edge_weighting.edge_weights_exp_time_since_last_inter(graph, exp["edge_weighting_gamma"], exp["edge_weighting_only_last"])
        elif exp["edge_weighting"] == "exp_time":
            edge_weighting.edge_weights_exp_time(graph, exp["edge_weighting_gamma"], exp["edge_weighting_only_last"])
        else:
            raise ValueError("Edge weighting not recognized")
    edge_weight = graph.edge_index.storage.value()
    print("Training graph weights:")
    print("\tmax", torch.max(edge_weight).item())
    print("\tmin", torch.min(edge_weight).item())
    print("\tmean", torch.mean(edge_weight).item())
    print("\tmedian", torch.median(edge_weight).item())

    # Add edges
    if exp["type"] == "GNN":
        if exp["add_edges"] == "add_nothing":
            pass
        elif exp["add_edges"] == "connect_same_musics":
            assert users_df is not None, "No user features provided"
            assert items_df is not None, "No item features provided"
            if "merge_on" in exp.config:
                merge_on = list(exp["merge_on"])
            else:
                merge_on = "music_id"
            graph = add_edge.connect_same_musics(graph, users_df, items_df, inter_df, merge_on)
            val_graph = add_edge.connect_same_musics(val_graph, users_df, items_df, inter_df, merge_on)
        else:
            raise ValueError("Method to add edges not recognized")

    # Embedding
    if exp["embedding_type"] == "IdEmbedding":
        embedding = IdEmbedding.from_graph(graph, embedding_dim)
    elif exp["embedding_type"] == "IdEmbeddingPlusNameEmbedding":
        embedding = IdEmbeddingPlusNameEmbedding.from_graph(graph, embedding_dim)
    elif exp["embedding_type"] == "IdEmbeddingPlusNameEmbedding2":
        embedding = IdEmbeddingPlusNameEmbedding2.from_graph(graph, embedding_dim)
    elif exp["embedding_type"] == "UsersFeaturesEmbeddingPlusNameEmbdding":
        embedding = UsersFeaturesEmbeddingPlusNameEmbdding.from_graph(graph, embedding_dim)
    elif exp["embedding_type"] == "UsersFeaturesAndIdEmbeddingPlusNameEmbedding":
        embedding = UsersFeaturesAndIdEmbeddingPlusNameEmbedding.from_graph(graph, embedding_dim)
    # elif exp["embedding_type"] == "ItemsFeaturesEmbedding_plus_name_emb":
    #     embedding = ItemsFeaturesEmbedding_plus_name_emb.from_graph(graph, embedding_dim)
    # elif exp["embedding_type"] == "AllFeaturesEmbedding_plus_name_emb":
    #     embedding = AllFeaturesEmbedding_plus_name_emb.from_graph(graph, embedding_dim)
    else:
        raise ValueError("Embedding not recognized")

    # Model
    if exp["model"] == "LightGCN_simple":
        model = LightGCN_simple(nb_layers=num_layers, embedding=embedding).to(device)
    elif exp["model"] == "LightGCN_wo_first_emb":
        model = LightGCN_wo_first_emb(nb_layers=num_layers, embedding=embedding).to(device)
    elif exp["model"] == "Item2ItemCosineSimilarity":
        model = Item2ItemCosineSimilarity(graph=graph).to(device)
    elif exp["model"] == "User2UserCosineSimilarity":
        model = User2UserCosineSimilarity(graph=graph).to(device)
    else:
        raise ValueError("Model not found")

    # Training parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=exp["lr_decay"])

    if exp["loss"] == "BPR":
        loss_fn = BPRLoss(weight_decay=weight_decay)
    elif exp["loss"] == "CE":
        loss_fn = CELoss(weight_decay=weight_decay)
        train_ds.nb_neg_sampling = exp["nb_neg_sampling"]
    else:
        raise ValueError("Loss not found")

    datas = (cold_data, graph, val_graph, train_ds, val_ds)
    model = (model, optimizer, scheduler, loss_fn)
    return datas, model
