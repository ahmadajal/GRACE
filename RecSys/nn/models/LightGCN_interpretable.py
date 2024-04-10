import torch

from RecSys.nn.models import CF
from RecSys.nn.models.embeddings import RecSysEmbedding


class TopKUsersItemsLightGCN(CF):
    r""" Interpretable LightGCN model

    It differ from the original LightGCN with 1 layer model by only taking
    the top-ku users and top-ki items contributions
    """

    def __init__(self, embedding: RecSysEmbedding, ku: int, ki: int):
        super().__init__()
        self.embedding = embedding
        self.ku = ku
        self.ki = ki

    def forward(self, graph, u_idx, i_idx):
        tol = 1e-12
        e0 = self.embedding.get_all(graph)
        deg = 1 / (torch.sqrt(graph.edge_index.sum(1)) + tol)
        e1 = torch.clone(e0)
        e1 = e1 * deg.unsqueeze(1)
        adj_matrix = graph.edge_index.cpu().to_dense()[:graph.num_users, graph.num_users:]
        # adj_matrix = adj_matrix

        s0 = torch.sum(e0[u_idx] * e0[i_idx], dim=1)

        a = adj_matrix.T[i_idx-graph.num_users].to(torch.float32).to(e0.device)

        s01 = a * (e0[u_idx] @ e1[:graph.num_users].T)
        s01 = torch.topk(s01, self.ku, dim=1, sorted=False)[0]
        s01 = torch.sum(s01, dim=1)
        s01 = s01 * deg[i_idx]

        a = adj_matrix[u_idx].to(torch.float32).to(e0.device)

        s10 = a * (e0[i_idx] @ e1[graph.num_users:].T)
        s10 = torch.topk(s10, self.ki, dim=1, sorted=False)[0]
        s10 = torch.sum(s10, dim=1)
        s10 = s10 * deg[u_idx]

        return s0 + s01 + s10

    def get_weight_norm(self) -> torch.Tensor:
        return torch.tensor(0.0)
