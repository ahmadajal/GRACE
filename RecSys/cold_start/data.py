""" Data class for cold start recommendation """
import torch
import pandas as pd
from RecSys.utils.data.data import RecGraphData
from RecSys.utils.data.data import TestDataset
from typing import Union


class ColdStartData():
    """Data class for cold start recommendation"""

    def __init__(self, user_df: pd.DataFrame, item_df: pd.DataFrame, inter_df: pd.DataFrame,
                 cold_user_id, device="cpu"):
        """
        Initialize cold start data

        Args:
            user_df (pd.DataFrame): user dataframe
            item_df (pd.DataFrame): item dataframe
            inter_df (pd.DataFrame): interaction dataframe
            inter_df_augmented (pd.DataFrame): interaction dataframe with possible additional edges
            cold_user_id (list): list of cold start user ids
            device (str): device to use

        """
        self.device = device
        self.user_df = user_df
        self.item_df = item_df

        cold_user_id = [int(e) for e in cold_user_id]

        self.train_user_df = user_df[~user_df["u"].isin(cold_user_id)]
        self.train_inter_df = inter_df[~inter_df["u"].isin(cold_user_id)]

        self.cold_user_df = user_df[user_df["u"].isin(cold_user_id)]
        self.cold_inter_df = inter_df[inter_df["u"].isin(cold_user_id)]

        self.cold_user_id = torch.tensor(cold_user_id, dtype=torch.long).to(device)

        # Train graph
        self.train_data = RecGraphData(
            user_df=user_df,
            item_df=item_df,
            inter_df=self.train_inter_df,
            total_num_users=len(user_df)
        ).to(device)
        # Cold graph
        self.cold_data = RecGraphData(
            user_df=user_df,
            item_df=item_df,
            inter_df=self.cold_inter_df,
            total_num_users=len(user_df)
        ).to(device)

        self.__sort_cold_user_inter()

    def to(self, device):
        """ Move data to device """
        self.device = device
        self.cold_user_id = self.cold_user_id.to(device)
        self.train_data = self.train_data.to(device)
        self.cold_data = self.cold_data.to(device)
        return self

    def __sort_cold_user_inter(self):
        """ Sort cold user interactions by timestamp """
        if not hasattr(self, "user_inter_sorted"):
            self.user_inter_sorted = dict()
            for u_id in self.cold_user_id:
                u_id = u_id.cpu().item()
                u_inter = self.cold_inter_df[self.cold_inter_df["u"] == u_id]
                u_inter.sort_values(by="t", inplace=True)
                self.user_inter_sorted[u_id] = u_inter

    def __get_test_data(self, nb_inter):
        """
        [WARNING]: Deprecated

        Get test data for cold start recommendation

        It devides the cold users interactions into two parts:
        - pre_data: nb_inter first interactions of the new users
        - post_data: 25 next interactions of the new users

        Users with less than nb_inter + 25 interactions are not considered

        Args:
            nb_inter (int): number of interactions considered as known for new users

        Returns:
            pre_data (RecGraphData): nb_inter first interactions of the new users
            post_data (RecGraphData): 25 next interactions of the new users
            test_ds (TestDataset): test dataset
            user_with_enough_inter (list): list of users with enough interactions
        """
        pre_inter = []
        post_inter = []
        nb_user_wo_enough_inter = 0
        user_with_enough_inter = []
        for u_id in self.cold_user_id:
            u_id = u_id.cpu().item()
            u_inter = self.user_inter_sorted[u_id]
            if len(u_inter) <= nb_inter + 25:
                nb_user_wo_enough_inter += 1
                # pre_inter.append(u_inter.iloc[:len(u_inter) - 1])
                # post_inter.append(u_inter.iloc[-1:])
            else:
                user_with_enough_inter.append(u_id)
                pre_inter.append(u_inter.iloc[:nb_inter])
                post_inter.append(u_inter.iloc[nb_inter:nb_inter + 25])
        if nb_user_wo_enough_inter > 0:
            print(f"{nb_user_wo_enough_inter} ({nb_user_wo_enough_inter/len(self.cold_user_id)*100:.1f}%) users do not have enough interactions for {nb_inter} interactions")
        pre_inter = pd.concat(pre_inter)
        post_inter = pd.concat(post_inter)

        pre_data = RecGraphData(
            user_df=self.user_df,
            item_df=self.item_df,
            inter_df=pre_inter,
            inter_df_augmented=pre_inter,
            total_num_users=len(self.user_df)
        ).to(self.device)

        post_data = RecGraphData(
            user_df=self.user_df,
            item_df=self.item_df,
            inter_df=post_inter,
            inter_df_augmented=post_inter,
            total_num_users=len(self.user_df)
        ).to(self.device)

        test_ds = TestDataset(train_graph=self.train_data, val_graph=post_data)
        user_with_enough_inter = torch.tensor(user_with_enough_inter, dtype=torch.long).to(self.device)
        return pre_data, post_data, test_ds, user_with_enough_inter

    def get_test_data(self, nb_inter, nb_inter_max=300):
        """
        Get test data for cold start recommendation

        It devides the cold users interactions into two parts:
        - post_data: 25 last interactions of the new users
        - pre_data: nb_inter last interactions of the new users
            before the last 25

        Users with less than nb_inter_max + 25 interactions are not considered

        Args:
            nb_inter (int): number of interactions considered as known for new users

        Returns:
            pre_data (RecGraphData): nb_inter last interactions of the new users
                before the last 25
            post_data (RecGraphData): 25 last interactions of the new users
            test_ds (TestDataset): test dataset
            user_with_enough_inter (list): list of users with enough interactions
        """
        pre_inter = []
        post_inter = []
        nb_user_wo_enough_inter = 0
        user_with_enough_inter = []
        for u_id in self.cold_user_id:
            u_id = u_id.cpu().item()
            u_inter = self.user_inter_sorted[u_id]
            if len(u_inter) <= nb_inter_max + 25:
                nb_user_wo_enough_inter += 1
                # pre_inter.append(u_inter.iloc[:len(u_inter) - 1])
                # post_inter.append(u_inter.iloc[-1:])
            else:
                user_with_enough_inter.append(u_id)
                pre_inter.append(u_inter.iloc[-nb_inter-25:-25])
                post_inter.append(u_inter.iloc[-25:])
        if nb_user_wo_enough_inter > 0:
            print(f"{nb_user_wo_enough_inter} ({nb_user_wo_enough_inter/len(self.cold_user_id)*100:.1f}%) users do not have {nb_inter_max+25} interactions")
        pre_inter = pd.concat(pre_inter)
        post_inter = pd.concat(post_inter)

        pre_data = RecGraphData(
            user_df=self.user_df,
            item_df=self.item_df,
            inter_df=pre_inter,
            inter_df_augmented=pre_inter,
            total_num_users=len(self.user_df)
        ).to(self.device)

        post_data = RecGraphData(
            user_df=self.user_df,
            item_df=self.item_df,
            inter_df=post_inter,
            inter_df_augmented=post_inter,
            total_num_users=len(self.user_df)
        ).to(self.device)

        test_ds = TestDataset(train_graph=self.train_data, val_graph=post_data)
        user_with_enough_inter = torch.tensor(user_with_enough_inter, dtype=torch.long).to(self.device)
        return pre_data, post_data, test_ds, user_with_enough_inter
