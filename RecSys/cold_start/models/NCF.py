import torch
from tqdm import tqdm
from RecSys.utils.data.data import RecGraphData
from RecSys.nn.models.NCF import NCF
from RecSys.cold_start.models import ColdStartModel
from RecSys.cold_start.data import ColdStartData
from RecSys.nn.models.embeddings import UsersFeaturesAndIdEmbeddingPlusNameEmbedding, UsersFeaturesEmbeddingPlusNameEmbdding


def compute_mean_embedding(user_df, cold_user_id, x, on=None):
    if on is None:
        train_user_id = set(list(range(len(user_df))))
        train_user_id = train_user_id.difference(set(cold_user_id))
        train_user_id = torch.tensor(list(train_user_id), dtype=torch.long, device=x.device)
        emb_instrument = x[train_user_id].mean(dim=0)
    elif isinstance(on, str):
        emb_instrument = dict()
        for instrument in user_df[on].unique():
            user_instr = user_df[user_df[on] == instrument]["u"].values
            user_instr = set(user_instr)
            train_user_instr = user_instr.difference(set(cold_user_id))
            train_user_instr = torch.tensor(list(train_user_instr), dtype=torch.long, device=x.device)
            emb_instrument[instrument] = x[train_user_instr].mean(dim=0)
    elif isinstance(on, list):
        assert len(on) == 2
        emb_instrument = dict()
        for instrument in user_df[on[0]].unique():
            for instrument2 in user_df[user_df[on[0]] == instrument][on[1]].unique():
                user_instr = user_df[(user_df[on[0]] == instrument) & (user_df[on[1]] == instrument2)]["u"].values
                user_instr = set(user_instr)
                train_user_instr = user_instr.difference(set(cold_user_id))
                train_user_instr = torch.tensor(list(train_user_instr), dtype=torch.long, device=x.device)
                emb_instrument[(instrument, instrument2)] = x[train_user_instr].mean(dim=0)
    else:
        raise ValueError("on must be str or list")
    return emb_instrument


def compute_mean_embedding_similar_inter(user_df, cold_user_id, x, edge_index, train_edge_index, train_user_id):
    x_cold = dict()
    edge_index = edge_index.to("cuda")
    train_edge_index = train_edge_index.to("cuda")
    train_user_id = train_user_id.to("cuda")
    for u_id in tqdm(cold_user_id):
        u_id = u_id.item()
        u_items = edge_index[1][edge_index[0] == u_id]
        u_neigh = train_edge_index[0][(torch.isin(train_edge_index[1], u_items)) & torch.isin(train_edge_index[0], train_user_id)]
        x_cold[u_id] = x[u_neigh].mean(dim=0)
    return x_cold


class NCFAvgAtt(ColdStartModel):

    def __init__(self, model: NCF):
        super().__init__()
        self.model = model

    def forward(self, cold_data: ColdStartData, pre_graph: RecGraphData, users_id: torch.Tensor) -> torch.Tensor:
        num_users = len(cold_data.user_df)
        num_items = len(cold_data.item_df)

        # e_0
        x = self.model.embedding.get_all(cold_data.train_data)

        # compute e_0_cold
        user_df = cold_data.user_df
        cold_user_id = cold_data.cold_user_id.cpu().numpy()
        emb_instrument = compute_mean_embedding(user_df, cold_user_id, x, ["instrument_id", "LEVEL"])

        # e_0_cold
        for u_id in cold_user_id:
            u_instr = user_df[user_df["u"] == u_id]["instrument_id"].values[0]
            u_level = user_df[user_df["u"] == u_id]["LEVEL"].values[0]
            x[u_id] = emb_instrument[(u_instr, u_level)]

        all_scores = []
        item_emb = self.model.embedding(torch.arange(num_users, num_users + num_items, dtype=torch.long, device=x.device))
        item_emb = item_emb.to("cuda")
        self.model = self.model.to("cuda")
        for u_id in tqdm(users_id):
            user_emb = x[u_id].to("cuda")
            user_emb = user_emb.unsqueeze(0).repeat(num_items, 1)
            interactions = torch.cat((user_emb, item_emb), dim=1)
            scores = self.model.MLP_layers(interactions)
            scores = self.model.predict_layer(scores).squeeze().cpu()
            all_scores.append(scores)
        return torch.stack(all_scores, dim=0)


class NCFAvgInter(ColdStartModel):

    def __init__(self, model: NCF):
        super().__init__()
        self.model = model

    def forward(self, cold_data: ColdStartData, pre_graph: RecGraphData, users_id: torch.Tensor) -> torch.Tensor:
        num_users = len(cold_data.user_df)
        num_items = len(cold_data.item_df)

        # e_0
        x = self.model.embedding.get_all(cold_data.train_data)

        # compute e_0_cold
        user_df = cold_data.user_df
        # cold_user_id = cold_data.cold_user_id.cpu().numpy()
        cold_x = compute_mean_embedding_similar_inter(user_df, users_id, x, pre_graph.directed_edge_index, cold_data.train_data.edge_index,
                                                      torch.arange(0, len(user_df) - len(cold_data.cold_user_id), dtype=torch.long, device="cpu"))

        all_scores = []
        item_emb = self.model.embedding(torch.arange(num_users, num_users + num_items, dtype=torch.long, device=x.device))
        item_emb = item_emb.to("cuda")
        self.model = self.model.to("cuda")
        for u_id in tqdm(users_id):
            u_id = u_id.item()
            user_emb = cold_x[u_id].to("cuda")
            user_emb = user_emb.unsqueeze(0).repeat(num_items, 1)
            interactions = torch.cat((user_emb, item_emb), dim=1)
            scores = self.model.MLP_layers(interactions)
            scores = self.model.predict_layer(scores).squeeze().cpu()
            all_scores.append(scores)
        return torch.stack(all_scores, dim=0)


class NCFUserFeaturesEmbedding(ColdStartModel):

    def __init__(self, model: NCF):
        super().__init__()
        self.model = model
        assert isinstance(self.model.embedding, UsersFeaturesEmbeddingPlusNameEmbdding)

    def forward(self, cold_data: ColdStartData, pre_graph: RecGraphData, users_id: torch.Tensor) -> torch.Tensor:
        num_users = len(cold_data.user_df)
        num_items = len(cold_data.item_df)

        preds = []
        items_id = torch.arange(num_users, num_users + num_items, dtype=torch.long, device="cuda")
        for u in tqdm(users_id):
            for p in torch.split(self.model(pre_graph, u.repeat(num_items), items_id).to("cuda"), num_items):
                preds.append(p)
        preds = torch.stack(preds, 0)

        return preds


class NCFUserFeaturesAndIdEmbedding(ColdStartModel):

    def __init__(self, model: NCF):
        super().__init__()
        self.model = model
        print(type(self.model.embedding))
        assert isinstance(self.model.embedding, UsersFeaturesAndIdEmbeddingPlusNameEmbedding)

    def forward(self, cold_data: ColdStartData, pre_graph: RecGraphData, users_id: torch.Tensor) -> torch.Tensor:
        emb = self.model.embedding.embedding.weight  # type: ignore
        emb[users_id] = 0  # type: ignore
        self.model.embedding.embedding = torch.nn.Embedding.from_pretrained(emb)

        num_users = len(cold_data.user_df)
        num_items = len(cold_data.item_df)

        preds = []
        items_id = torch.arange(num_users, num_users + num_items, dtype=torch.long, device="cuda")
        for u in tqdm(users_id):
            for p in torch.split(self.model(pre_graph, u.repeat(num_items), items_id).to("cuda"), num_items):
                preds.append(p)
        preds = torch.stack(preds, 0)

        return preds
