r""" Embedding module

This module contains the embedding classes used in the models.

"""
import numpy as np

import torch
import torch.nn.init
import torch.nn
from torch import Tensor
from torch.nn import Module, Embedding
from sklearn.decomposition import PCA

from RecSys.utils.data.data import RecGraphData


# Specific to the Tomplay dataset
PATH_NAME_EMBEDDINGS = "data/tomplay/name_composer_embeddings.npy"


class RecSysEmbedding(Module):
    """ Base class for embeddings. """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        r""" Initialize a RecSysEmbedding instance

        Args:
            num_embeddings (int): Number of users and items.
            embedding_dim (int): Size of the embedding vector.
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    @classmethod
    def from_graph(cls, graph: RecGraphData, embedding_dim: int):
        r""" Create a RecSysEmbedding instance from a graph

        Args:
            graph (RecGraphData): Graph data.
            embedding_dim (int): Size of the embedding vector.

        Returns:
            RecSysEmbedding: RecSysEmbedding instance.
        """
        num_embeddings = graph.total_num_users + graph.num_items + graph.num_other_nodes
        return cls(num_embeddings, embedding_dim)

    def forward(self, graph: RecGraphData, idx: Tensor) -> Tensor:
        r""" Forward pass of the embedding

        Args:
            graph (RecGraphData): Graph data.
            e (Tensor): Index of the embedding to retrieve.

        Returns:
            Tensor: Embedding vector.
        """
        raise NotImplementedError

    def get_all(self, graph: RecGraphData) -> Tensor:
        r""" Get all embeddings

        Args:
            graph (RecGraphData): Graph data.

        Returns:
            Tensor: Embedding of all nodes in the graph.
        """
        raise NotImplementedError

    def get_weight_norm(self) -> Tensor:
        r""" Get the weight norm of the embedding

        Returns:
            Tensor: Weight norm of the embedding.
        """
        raise NotImplementedError


class IdEmbedding(RecSysEmbedding):
    """ Embedding based on the id of the user or item. """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        r""" IdEmbedding constructor
        Each user and item is represented by a vector of size embedding_dim.
        The embedding is initialized with Xavier uniform initialization.

        Args:
            num_embeddings (int): Number of users and items.
            embedding_dim (int): Size of the embedding vector.
        """
        super(IdEmbedding, self).__init__(num_embeddings, embedding_dim)
        self.embedding = Embedding(num_embeddings, embedding_dim)
        self.reset_parameters()

    def forward(self, graph, idx):
        return self.embedding(idx)

    def get_all(self, graph):
        return self.embedding(graph.x[:, 0])

    def reset_parameters(self):
        """ Initialize the embedding with Xavier uniform initialization. """
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def get_weight_norm(self):
        return self.embedding.weight.pow(2).sum()


class IdEmbeddingPlusNameEmbedding(RecSysEmbedding):
    """ Embedding based on the id of the user or item and the name of the item.

    The name of the item is embedded by a BERT model.

    Parameters of the id embedding are initialized with Xavier uniform initialization.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        r""" IdEmbeddingPlusNameEmbedding constructor
        Each user and item is represented by a vector of size embedding_dim.
        The embedding is initialized with Xavier uniform initialization + the name embedded by a BERT model.

        Args:
            num_embeddings (int): Number of users and items.
            embedding_dim (int): Size of the embedding vector.
        """
        super().__init__(num_embeddings, embedding_dim)
        self.embedding = Embedding(num_embeddings, embedding_dim)
        self.get_name_emb()
        self.reset_parameters()

    def forward(self, graph: RecGraphData, idx: Tensor) -> Tensor:
        return self.embedding(idx) + self.name_emb[idx]

    def get_all(self, graph) -> Tensor:
        return self.embedding(graph.x[:, 0]) + self.name_emb[graph.x[:, 0]]

    def get_name_emb(self):
        """Load name embeddings and reduce dimensionality with PCA."""
        name_emb = np.load(PATH_NAME_EMBEDDINGS)

        pca = PCA(n_components=self.embedding_dim)
        name_emb = pca.fit_transform(name_emb, None)

        print("PCA explained variance ratio: ", sum(pca.explained_variance_ratio_))  # type: ignore

        self.name_emb = torch.tensor(name_emb, dtype=torch.float32, requires_grad=False)
        self.name_emb = torch.cat((
            torch.zeros((self.num_embeddings - self.name_emb.size(0), self.embedding_dim), dtype=torch.float32),
            self.name_emb),
            0
        )
        self.name_emb = torch.nn.parameter.Parameter(self.name_emb, requires_grad=False)

    def reset_parameters(self):
        """ Initialize the embedding with Xavier uniform initialization. """
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def get_weight_norm(self):
        return self.embedding.weight.pow(2).sum()


class IdEmbeddingPlusNameEmbedding2(RecSysEmbedding):
    """ Embedding based on the id of the user or item and the name of the item.

    The name of the item is embedded by a BERT model.

    Parameters of the id embedding are initialized with N(0, 0.01**2) initialization.

    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        r""" IdEmbedding constructor
        Each user and item is represented by a vector of size embedding_dim.
        The embedding is initialized with normal N(0, 0.01**2) initialization.
        Args:
            num_embeddings (int): Number of users and items.
            embedding_dim (int): Size of the embedding vector.
        """
        super().__init__(num_embeddings, embedding_dim)
        self.embedding = Embedding(num_embeddings, embedding_dim)
        self.get_name_emb()
        self.reset_parameters()

    def forward(self, graph: RecGraphData, idx: Tensor) -> Tensor:
        return self.embedding(idx) + self.name_emb[idx]

    def get_all(self, graph) -> Tensor:
        return self.embedding(graph.x[:, 0]) + self.name_emb[graph.x[:, 0]]

    def get_name_emb(self):
        """Load name embeddings and reduce dimensionality with PCA."""
        name_emb = np.load(PATH_NAME_EMBEDDINGS)

        pca = PCA(n_components=self.embedding_dim)
        name_emb = pca.fit_transform(name_emb, None)

        print("PCA explained variance ratio: ", sum(pca.explained_variance_ratio_))  # type: ignore

        self.name_emb = torch.tensor(name_emb, dtype=torch.float32, requires_grad=False)
        self.name_emb = torch.cat((
            torch.zeros((self.num_embeddings - self.name_emb.size(0), self.embedding_dim), dtype=torch.float32),
            self.name_emb),
            0
        )
        self.name_emb = torch.nn.parameter.Parameter(self.name_emb, requires_grad=False)

    def reset_parameters(self):
        """Initialize the embedding with normal N(0, 0.01**2) initialization."""
        torch.nn.init.normal_(self.embedding.weight.data, std=0.01)

    def get_weight_norm(self):
        return self.embedding.weight.pow(2).sum()


class UsersFeaturesEmbeddingPlusNameEmbdding(RecSysEmbedding):
    """ Embedding based on the user categorical features (level, instrument) and item's id and name. """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        r""" UsersFeaturesEmbedding constructor
        Each user and item is represented by a vector of size embedding_dim.
        The embedding is initialized with Xavier uniform initialization.
        Users are represented by a concatenation of their features.
        Args:
            num_embeddings (int): Number of users and items.
            embedding_dim (int): Size of the embedding vector.
        """
        super().__init__(num_embeddings, embedding_dim)
        self.embedding = Embedding(num_embeddings, embedding_dim)  # 4 for the levels, 26 for the instruments
        self.get_name_emb()
        self.reset_parameters()

    @classmethod
    def from_graph(cls, graph: RecGraphData, embedding_dim: int):
        num_embeddings = 4 + 26 + graph.num_items  # 4 for the levels, 26 for the instruments
        return cls(num_embeddings, embedding_dim)

    def get_all(self, graph):
        user_features_emb = self.embedding(graph.x[:graph.num_users, 1])  # level
        user_features_emb += self.embedding(graph.x[:graph.num_users, 2]+4)  # instrument
        user_features_emb = torch.cat(
            (
                user_features_emb,
                self.embedding(graph.x[graph.num_users: graph.num_users+graph.num_items, 0] - graph.num_users + 4 + 26)  # items
            ),
            0)
        return user_features_emb + self.name_emb

    def forward(self, graph: RecGraphData, idx: Tensor) -> Tensor:
        # e is a tensor of user ids and item ids
        mask_users = idx < graph.num_users
        mask_items = idx >= graph.num_users

        u_levels = graph.x[idx, 1]
        u_levels = torch.where(mask_items, 0, u_levels)
        u_instr = graph.x[idx, 2]
        u_instr = torch.where(mask_items, 0, u_instr)
        i_id = graph.x[idx, 0]
        i_id = torch.where(mask_users, graph.num_users, i_id)

        emb = torch.zeros((idx.size(0), self.embedding_dim), dtype=torch.float32, device=idx.device)
        emb = torch.where(mask_users.repeat(self.embedding_dim, 1).T, self.embedding(u_levels), emb)
        emb = torch.where(mask_users.repeat(self.embedding_dim, 1).T, self.embedding(u_instr + 4) + emb, emb)
        emb = torch.where(mask_items.repeat(self.embedding_dim, 1).T, self.embedding(i_id - graph.num_users + 4 + 26) + emb, emb)
        idx = torch.where(mask_users, 0, idx - graph.num_users + 4 + 26)
        emb += self.name_emb[idx]
        return emb

    def get_name_emb(self):
        """Load name embeddings and reduce dimensionality with PCA."""
        name_emb = np.load(PATH_NAME_EMBEDDINGS)

        pca = PCA(n_components=self.embedding_dim)
        name_emb = pca.fit_transform(name_emb, None)

        print("PCA explained variance ratio: ", sum(pca.explained_variance_ratio_))  # type: ignore

        self.name_emb = torch.tensor(name_emb, dtype=torch.float32, requires_grad=False)
        self.name_emb = torch.cat((
            torch.zeros((self.num_embeddings - self.name_emb.size(0), self.embedding_dim), dtype=torch.float32),
            self.name_emb),
            0
        )
        self.name_emb = torch.nn.parameter.Parameter(self.name_emb, requires_grad=False)

    def reset_parameters(self):
        """ Initialize the embedding with Xavier uniform initialization. """
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def get_weight_norm(self):
        w_norm = self.embedding.weight.pow(2).sum()
        return w_norm


class UsersFeaturesAndIdEmbeddingPlusNameEmbedding(RecSysEmbedding):
    """ Embedding based on the user id and categorical features (level, instrument) and item's id and name."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        r""" IdEmbedding constructor
        Each user and item is represented by a vector of size embedding_dim.
        The embedding is initialized with Xavier uniform initialization.
        Users are represented by a concatenation of their features and their id.

        Args:
            num_embeddings (int): Number of users and items.
            embedding_dim (int): Size of the embedding vector.
        """
        super().__init__(num_embeddings, embedding_dim)
        self.embedding = Embedding(num_embeddings, embedding_dim)
        self.get_name_emb()
        self.reset_parameters()

    @classmethod
    def from_graph(cls, graph: RecGraphData, embedding_dim: int):
        num_embeddings = graph.num_users + graph.num_items + 4 + 26  # 4 for the levels, 26 for the instruments
        return cls(num_embeddings, embedding_dim)

    def get_all(self, graph: RecGraphData) -> Tensor:
        user_features_emb = self.embedding(graph.x[:graph.num_users, 0])  # user id
        user_features_emb += self.embedding(graph.x[:graph.num_users, 1] + graph.num_users)  # level
        user_features_emb += self.embedding(graph.x[:graph.num_users, 2] + graph.num_users + 4)  # instrument
        user_features_emb = torch.cat(
            (
                user_features_emb,
                self.embedding(graph.x[graph.num_users: graph.num_users+graph.num_items, 0] + 4 + 26)  # items
            ),
            0)
        return user_features_emb + self.name_emb

    def forward(self, graph: RecGraphData, idx: Tensor) -> Tensor:
        # e is a tensor of user ids and item ids
        mask_users = idx < graph.num_users
        mask_items = idx >= graph.num_users

        u_id = graph.x[idx, 0]
        u_id = torch.where(mask_items, 0, u_id)
        u_levels = graph.x[idx, 1]
        u_levels = torch.where(mask_items, 0, u_levels)
        u_instr = graph.x[idx, 2]
        u_instr = torch.where(mask_items, 0, u_instr)
        i_id = graph.x[idx, 0]
        i_id = torch.where(mask_users, graph.num_users, i_id)

        emb = torch.zeros((idx.size(0), self.embedding_dim), dtype=torch.float32, device=idx.device)
        emb = torch.where(mask_users.repeat(self.embedding_dim, 1).T, self.embedding(u_id), emb)
        emb = torch.where(mask_users.repeat(self.embedding_dim, 1).T, self.embedding(u_levels + graph.num_users) + emb, emb)
        emb = torch.where(mask_users.repeat(self.embedding_dim, 1).T, self.embedding(u_instr + graph.num_users + 4) + emb, emb)
        emb = torch.where(mask_items.repeat(self.embedding_dim, 1).T, self.embedding(i_id + 4 + 26) + emb, emb)
        idx = torch.where(mask_users, 0, idx + 4 + 26)
        emb += self.name_emb[idx]
        return emb

    def get_name_emb(self):
        """Load name embeddings and reduce dimensionality with PCA."""
        name_emb = np.load(PATH_NAME_EMBEDDINGS)

        pca = PCA(n_components=self.embedding_dim)
        name_emb = pca.fit_transform(name_emb, None)

        print("PCA explained variance ratio: ", sum(pca.explained_variance_ratio_))  # type: ignore

        self.name_emb = torch.tensor(name_emb, dtype=torch.float32, requires_grad=False)
        self.name_emb = torch.cat((
            torch.zeros((self.num_embeddings - self.name_emb.size(0), self.embedding_dim), dtype=torch.float32),
            self.name_emb),
            0
        )
        self.name_emb = torch.nn.parameter.Parameter(self.name_emb, requires_grad=False)

    def reset_parameters(self):
        """ Initialize the embedding with Xavier uniform initialization. """
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def get_weight_norm(self):
        w_norm = self.embedding.weight.pow(2).sum()
        return w_norm


# TODO: DEPRECATED, Format the class according to the RecSysEmbedding class
class ItemsFeaturesEmbedding_plus_name_emb(Module):

    def __init__(self, num_embeddings: int, embedding_dim: int):
        """ IdEmbedding constructor
        Each user and item is represented by a vector of size embedding_dim.
        The embedding is initialized with Xavier uniform initialization.

        Args:
            num_embeddings (int): Number of users and items.
            embedding_dim (int): Size of the embedding vector.
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # self.id_embedding = Embedding(num_embeddings, embedding_dim)
        self.embedding = Embedding(9+26+11+22, embedding_dim)
        self.get_name_emb()
        self.reset_parameters()

    @staticmethod
    def from_graph(graph: RecGraphData, embedding_dim: int):
        """ Create an IdEmbedding instance from a graph"""
        num_embeddings = graph.num_users + graph.num_items
        return ItemsFeaturesEmbedding_plus_name_emb(num_embeddings, embedding_dim)

    def get_all(self, graph):
        item_features_emb = self.embedding(graph.x[graph.num_users: graph.num_users+graph.num_items, 5])  # level
        item_features_emb += self.embedding(graph.x[graph.num_users: graph.num_users+graph.num_items, 6]+9)  # instrument
        item_features_emb += self.embedding(graph.x[graph.num_users: graph.num_users+graph.num_items, 4]+9+26)  # style
        item_features_emb += self.embedding(graph.x[graph.num_users: graph.num_users+graph.num_items, 3]+9+26+11)  # acc type
        item_features_emb = torch.cat((
            torch.zeros((graph.num_users, self.embedding_dim), dtype=item_features_emb.dtype, device=item_features_emb.device),
            item_features_emb), 0)
        return item_features_emb + self.name_emb

    def forward(self, e):
        emb = self.get_all(None)
        return emb[e]

    def get_name_emb(self):
        name_emb = np.load(PATH_NAME_EMBEDDINGS)

        pca = PCA(n_components=self.embedding_dim)
        name_emb = pca.fit_transform(name_emb, None)

        print("PCA explained variance ratio: ", sum(pca.explained_variance_ratio_))  # type: ignore

        self.name_emb = torch.tensor(name_emb, dtype=torch.float32, requires_grad=False)
        self.name_emb = torch.cat((
            torch.zeros((self.num_embeddings - self.name_emb.size(0), self.embedding_dim), dtype=torch.float32),
            self.name_emb),
            0
        )
        self.name_emb = torch.nn.parameter.Parameter(self.name_emb, requires_grad=False)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def get_weight_norm(self):
        w_norm = self.embedding.weight.pow(2).sum()
        return w_norm


class AllFeaturesEmbedding_plus_name_emb(Module):

    def __init__(self, num_embeddings: int, embedding_dim: int):
        """ IdEmbedding constructor
        Each user and item is represented by a vector of size embedding_dim.
        The embedding is initialized with Xavier uniform initialization.

        Args:
            num_embeddings (int): Number of users and items.
            embedding_dim (int): Size of the embedding vector.
        """
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding = Embedding(4+9+26+11+22, embedding_dim)
        self.get_name_emb()
        self.reset_parameters()

    @staticmethod
    def from_graph(graph: RecGraphData, embedding_dim: int):
        """ Create an IdEmbedding instance from a graph"""
        num_embeddings = graph.num_users + graph.num_items
        return AllFeaturesEmbedding_plus_name_emb(num_embeddings, embedding_dim)

    def get_all(self, graph):
        # Users_features
        user_features_emb = self.embedding(graph.x[:graph.num_users, 1])
        user_features_emb += self.embedding(graph.x[:graph.num_users, 2]+4+9)

        # Items_features
        item_features_emb = self.embedding(graph.x[graph.num_users: graph.num_users+graph.num_items, 5]+4)
        item_features_emb += self.embedding(graph.x[graph.num_users: graph.num_users+graph.num_items, 6]+4+9)
        item_features_emb += self.embedding(graph.x[graph.num_users: graph.num_users+graph.num_items, 4]+4+9+26)
        item_features_emb += self.embedding(graph.x[graph.num_users: graph.num_users+graph.num_items, 3]+4+9+26+11)

        features_emb = torch.cat((user_features_emb, item_features_emb), 0)
        return features_emb + self.name_emb

    def forward(self, e):
        emb = self.get_all(None)
        return emb[e]

    def get_name_emb(self):
        name_emb = np.load(PATH_NAME_EMBEDDINGS)

        pca = PCA(n_components=self.embedding_dim)
        name_emb = pca.fit_transform(name_emb, None)

        print("PCA explained variance ratio: ", sum(pca.explained_variance_ratio_))  # type: ignore

        self.name_emb = torch.tensor(name_emb, dtype=torch.float32, requires_grad=False)
        self.name_emb = torch.cat((
            torch.zeros((self.num_embeddings - self.name_emb.size(0), self.embedding_dim), dtype=torch.float32),
            self.name_emb),
            0
        )
        self.name_emb = torch.nn.parameter.Parameter(self.name_emb, requires_grad=False)

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def get_weight_norm(self):
        w_norm = self.embedding.weight.pow(2).sum()
        return w_norm
