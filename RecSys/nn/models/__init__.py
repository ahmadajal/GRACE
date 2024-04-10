"""
Define the abstract classes for the GNN and CF models

The GNN model is a model that computes the embedding of all the nodes of a graph.
The CF model is a model that computes the predictions for a batch of user-item pairs.
"""
from abc import ABC, abstractmethod
from torch.nn import Module
from torch import Tensor
from RecSys.utils.data.data import RecGraphData


class GNN(Module, ABC):
    """ Abstract class for the GNN model """

    @abstractmethod
    def get_embedding(self, graph: RecGraphData) -> Tensor:
        r""" Get the embedding of the nodes

        Args:
            graph: the graph data

        Returns:
            the embedding of the nodes
        """

    @abstractmethod
    def get_weight_norm(self) -> Tensor:
        """ Get the norm of the model weights """


class CF(Module, ABC):
    """ Abstract class for the CF model """

    @abstractmethod
    def forward(self, graph: RecGraphData, u_idx: Tensor, i_idx: Tensor) -> Tensor:
        r""" Compute the predictions for a batch

        Args:
            graph: the graph data
            u: the batch of users
            i: the batch of items

        Returns:
            the predictions for the batch
        """

    @abstractmethod
    def get_weight_norm(self) -> Tensor:
        """ Get the norm of the model weights """
