""" LightGCN model.

This file contains the LightGCN model.

"""
import torch
import torch.nn
import torch_geometric
import torch_geometric.nn

from RecSys.nn.models import GNN
from RecSys.nn.models.embeddings import RecSysEmbedding


class GAT(GNN):
    """ GAT model """

    def __init__(self, nb_layers: int, embedding: RecSysEmbedding):
        """ GAT constructor

        Args:
            nb_layers (int): Number of layers.
            embedding (IdEmbedding): Embedding layer.
        """
        super().__init__()
        # Embedding layer
        self.embedding = embedding

        # LightGCN layers
        self.nb_layers = nb_layers
        self.layers = torch.nn.ModuleList()
        for _ in range(nb_layers):
            self.layers.append(torch_geometric.nn.GATConv(
                embedding.embedding_dim,
                embedding.embedding_dim, improved=True
                ))

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
