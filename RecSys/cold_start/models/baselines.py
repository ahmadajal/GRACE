import torch
import numpy as np
from torch import Tensor
from RecSys.utils.data.data import RecGraphData
from RecSys.cold_start.models import ColdStartModel
from RecSys.cold_start.data import ColdStartData


class RandomPrediction(ColdStartModel):
    """ A model that predicts random scores """

    def forward(self, data: ColdStartData, pre_graph: RecGraphData, users_id: Tensor):
        return torch.rand(users_id.shape[0], data.item_df.shape[0])


class MostVisitedItems(ColdStartModel):
    """ A model that predicts the most visited items """

    def forward(self, data: ColdStartData, pre_graph: RecGraphData, users_id: Tensor) -> Tensor:
        count_inter = data.train_inter_df.groupby("i")["u"].count().reset_index()
        count_inter = count_inter.sort_values("i", ascending=True)["u"].values
        score = count_inter

        pred = []
        for u_id in users_id:
            pred.append(score)

        pred = np.stack(pred, 0)
        pred = torch.tensor(pred, dtype=torch.float32)

        return pred


class MostVisitedItemsByInstrument(ColdStartModel):
    """ A model that predicts the most visited items for the instrument of the user """

    def forward(self, data: ColdStartData, pre_graph: RecGraphData, users_id: Tensor) -> Tensor:
        instruments = data.cold_user_df["instrument_id"].unique()
        train_inter_df = data.train_inter_df.merge(data.item_df[["i", "instrument_id"]], on="i")
        item_df = data.item_df[["i", "instrument_id"]]
        scores = dict()
        for instrument in instruments:
            count_inter = train_inter_df[train_inter_df["instrument_id"] == instrument].groupby("i")["u"].count()
            scores[instrument] = [0 if item_id not in count_inter.index else count_inter.loc[item_id] for item_id in sorted(item_df.n_item_id.unique())]  # item_df["count"].values

        pred = []
        for u_id in users_id:
            u_id = u_id.cpu().item()
            pred.append(scores[data.cold_user_df[data.cold_user_df["u"] == u_id]["instrument_id"].values[0]])

        pred = torch.tensor(pred, dtype=torch.float32)

        return pred


class MostVisitedItemsByInstrumentLevel(ColdStartModel):
    """ A model that predicts the most visited items for the instrument and level of the user """

    def forward(self, data: ColdStartData, pre_graph: RecGraphData, users_id: Tensor) -> Tensor:
        instruments = data.cold_user_df["instrument_id"].unique()
        levels = data.cold_user_df["LEVEL"].unique()
        inter = data.train_inter_df.merge(data.train_user_df[["u", "LEVEL"]], on="u")
        inter = inter.merge(data.item_df[["i", "instrument_id"]], on="i")
        item_df = data.item_df[["i", "instrument_id"]]
        scores = dict()
        for instrument in instruments:
            for level in levels:
                count_inter = inter[(inter["instrument_id"] == instrument) & (inter["LEVEL"] == level)].groupby("i")["u"].count()
                scores[(instrument, level)] = [0 if item_id not in count_inter.index else count_inter.loc[item_id] for item_id in sorted(item_df.n_item_id.unique())]

        pred = []
        for u_id in users_id:
            u_id = u_id.cpu().item()
            u_instrument = data.cold_user_df[data.cold_user_df["u"] == u_id]["instrument_id"].values[0]
            u_level = data.cold_user_df[data.cold_user_df["u"] == u_id]["LEVEL"].values[0]
            pred.append(scores[(u_instrument, u_level)])

        pred = torch.tensor(pred, dtype=torch.float32)

        return pred
