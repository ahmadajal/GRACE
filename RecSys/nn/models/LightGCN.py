""" LightGCN model.

This file contains the LightGCN model.

"""
import torch
import torch.nn
import torch_geometric
import torch_geometric.nn

from RecSys.nn.models import GNN
from RecSys.nn.models.embeddings import RecSysEmbedding


class LightGCN_simple(GNN):
    """ LightGCN model """

    def __init__(self, nb_layers: int, embedding: RecSysEmbedding):
        """ LightGCN constructor
        The alpha parameter is initialized to 1/(nb_layers+1).

        Args:
            nb_layers (int): Number of layers.
            embedding (IdEmbedding): Embedding layer.
        """
        super(LightGCN_simple, self).__init__()
        # Embedding layer
        self.embedding = embedding

        # LightGCN layers
        self.nb_layers = nb_layers
        self.layers = torch.nn.ModuleList()
        for _ in range(nb_layers):
            self.layers.append(torch_geometric.nn.LGConv())

        # Alpha hyperparameter
        self.alpha = 1 / (nb_layers + 1)

    def get_embedding(self, graph):
        x = self.embedding.get_all(graph)
        out = x * self.alpha

        for i in range(self.nb_layers):
            x = self.layers[i](x, graph.edge_index)
            out += x * self.alpha

        return out

    def get_weight_norm(self):
        return self.embedding.get_weight_norm()

    def forward(self, x, edge_index):
        """x must be tensor of indexes."""
        e = self.embedding(x)
        out = e * self.alpha

        for i in range(self.nb_layers):
            e = self.layers[i](x, edge_index)
            out += e * self.alpha
        return out


class LightGCN_wo_first_emb(GNN):
    """ LightGCN model without the first embedding """

    def __init__(self, nb_layers: int, embedding: RecSysEmbedding):
        """ modified LightGCN that does not use the first embedding
        The alpha parameter is initialized to 1/nb_layers.

        Args:
            nb_layers (int): Number of layers.
            embedding (IdEmbedding): Embedding layer.
        """
        super(LightGCN_wo_first_emb, self).__init__()
        # Embedding layer
        self.embedding = embedding

        # LightGCN layers
        self.nb_layers = nb_layers
        self.layers = torch.nn.ModuleList()
        for _ in range(nb_layers):
            self.layers.append(torch_geometric.nn.LGConv())

        # Alpha hyperparameter
        self.alpha = 1 / nb_layers

    def get_embedding(self, graph):
        x = self.embedding.get_all(graph)
        out = torch.zeros_like(x)

        for i in range(self.nb_layers):
            x = self.layers[i](x, graph.edge_index)
            out += x * self.alpha

        return out

    def get_weight_norm(self):
        return self.embedding.get_weight_norm()
