import torch
from tqdm import tqdm
from RecSys.nn.models import CF
from RecSys.utils.data.data import RecGraphData


class Item2ItemCosineSimilarity(CF):
    """ Item2Item Cosine Similarity model """

    def __init__(self, graph: RecGraphData, *args, **kwargs):
        super().__init__()
        self.fake_param = torch.nn.parameter.Parameter(torch.zeros(1))
        adj_matrix = self.compute_adj_matrix(graph)
        item_similarities = self.compute_item_similarities(adj_matrix)
        self.scores = self.compute_scores(adj_matrix, item_similarities)

    @staticmethod
    def compute_adj_matrix(graph: RecGraphData):
        """ Compute the adjacency matrix for train and test """
        adj_matrix = torch.zeros((graph.num_users, graph.num_items), dtype=torch.float32)
        # adj_matrix[graph.directed_edge_index[0], graph.directed_edge_index[1] - graph.num_users] = 1
        for (u, i) in tqdm(graph.directed_edge_index.T, "Computing rating matrix"):
            adj_matrix[u, i - graph.num_users] += 1

        return adj_matrix.to("cuda:0")

    @staticmethod
    def compute_item_similarities(adj_matrix):
        """ Compute the item similarities """
        tol = 1e-6
        adj_matrix = adj_matrix - torch.mean(adj_matrix.T, dim=1, keepdim=True).T
        adj_matrix = adj_matrix / (torch.sqrt(torch.sum((adj_matrix**2).T, dim=1, keepdim=True)) + tol).T
        item_similarities = torch.matmul(adj_matrix.T, adj_matrix)
        # bs = 256
        # for i in tqdm(range(0, adj_matrix.shape[1], bs)):
        #     sij = torch.matmul(adj_matrix[:, i: min(i+bs, adj_matrix.shape[1])].t(), adj_matrix)
        #     item_similarities[i: min(i+bs, adj_matrix.shape[1]), :] = sij
        return item_similarities

    def compute_scores(self, adj_matrix, item_similarities):
        """ Compute all scores """
        # scores = torch.zeros((adj_matrix.shape[0], adj_matrix.shape[1]))
        scores = torch.matmul(adj_matrix, item_similarities)
        return scores

    def forward(self, graph, u_idx, i_idx):
        return self.scores[u_idx, i_idx-graph.num_users]

    def get_weight_norm(self):
        return 0


class User2UserCosineSimilarity(CF):
    """ User2User Cosine Similarity model """

    def __init__(self, graph: RecGraphData, *args, **kwargs):
        super().__init__()
        self.fake_param = torch.nn.parameter.Parameter(torch.zeros(1))
        adj_matrix = self.compute_rating_matrix(graph)
        user_similarities = self.compute_user_similarities(adj_matrix)
        self.scores = self.compute_scores(adj_matrix, user_similarities)

    @staticmethod
    def compute_rating_matrix(graph: RecGraphData):
        """ Compute the adjacency matrix for train and test """
        adj_matrix = torch.zeros((graph.num_users, graph.num_items), dtype=torch.float32)
        # adj_matrix[graph.directed_edge_index[0], graph.directed_edge_index[1] - graph.num_users] = 1
        for (u, i) in tqdm(graph.directed_edge_index.T, "Computing rating matrix"):
            adj_matrix[u, i - graph.num_users] += 1
        return adj_matrix.to("cuda:0")

    @staticmethod
    def compute_user_similarities(adj_matrix):
        """ Compute the user similarities """
        tol = 1e-6
        # user_similarities = torch.zeros((adj_matrix.shape[0], adj_matrix.shape[0]))
        adj_matrix = adj_matrix - torch.mean(adj_matrix, dim=1, keepdim=True)
        adj_matrix = adj_matrix / (torch.sqrt(torch.sum(adj_matrix**2, dim=1, keepdim=True)) + tol)
        # bs = 256
        # for u in tqdm(range(0, adj_matrix.shape[0], bs)):
        #     sij = torch.matmul(adj_matrix[u: min(u+bs, adj_matrix.shape[0])], adj_matrix.T)
        #     user_similarities[u: min(u+bs, adj_matrix.shape[0])] = sij
        user_similarities = torch.matmul(adj_matrix, adj_matrix.T)
        return user_similarities

    def compute_scores(self, adj_matrix, user_similarities):
        """ Compute all scores """
        # scores = torch.zeros((adj_matrix.shape[0], adj_matrix.shape[1]))
        scores = torch.matmul(user_similarities, adj_matrix)
        return scores

    def forward(self, graph, u_idx, i_idx):
        return self.scores[u_idx, i_idx-graph.num_users]

    def get_weight_norm(self):
        return 0
