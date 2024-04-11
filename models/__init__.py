from abc import abstractmethod
from typing import Tuple
from torch import Tensor
from RecSys.utils.data.data import RecGraphData


class InterpretableModel:
    """ Abstract class for interpretable models

    interpretable models here are models able to provide explanations for their predictions.
    i.e. they are able to provide a list of users and items that are the most important for a given prediction.
    """

    @abstractmethod
    def forward(self, graph: RecGraphData, u_idx: int, i_idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        """ Returns the list of users ids and items ids ordered by importance for the given prediction

        Args:
            graph: the graph on which the model is trained
            u_id: the user id for which the prediction is made
            i_id: the item id for which the prediction is made
            top_ku: the number of users used to compute the new prediction
            top_ki: the number of items used to compute the new prediction

        Returns:
            The score of the prediction with only the most important users and items
            The score of the prediction without the most important users and items
            The list of users ids ordered by importance for the given prediction
            The list of items ids ordered by importance for the given prediction
        """
        raise NotImplementedError
