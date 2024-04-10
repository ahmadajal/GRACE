from abc import ABC, abstractmethod

from torch import Tensor
from torch.nn import Module
from RecSys.cold_start.data import ColdStartData
from RecSys.utils.data.data import RecGraphData


class ColdStartModel(ABC, Module):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def forward(self, cold_data: ColdStartData, pre_graph: RecGraphData, users_id: Tensor) -> Tensor:
        r""" Compute the predictions for a batch

        Args:
            cold_data: the cold start data
            pre_graph: the graph data that the model knows
            users_id: the batch of users

        Returns:
            the predictions for the batch
        """
        pass
