""" GRACE module

GRACE is our approach.
"""
import torch
from torch import Tensor
from RecSys.utils.data.data import RecGraphData
from RecSys.nn.models.LightGCN import LightGCN_simple
from models import InterpretableModel


class GRACE(InterpretableModel):
    r""" GRACE model

    It is our approach.
    """

    def __init__(self, gnn: LightGCN_simple, graph: RecGraphData, ku: int, ki: int):
        self.gnn = gnn
        self.ku = ku
        self.ki = ki

        tol = 1e-12
        self.e0 = self.gnn.embedding.get_all(graph)
        self.deg = 1 / (torch.sqrt(graph.edge_index.sum(1)) + tol)
        self.e1 = torch.clone(self.e0)
        self.e1 = self.e1 * self.deg.unsqueeze(1)
        self.adj_matrix = graph.edge_index.cpu().to_dense()[:graph.num_users, graph.num_users:]
        self.adj_matrix = self.adj_matrix.to(torch.float32).to(self.e0.device)

    @torch.no_grad()
    def one_forward(self, graph, u_id, i_id) -> Tensor:
        """ One forward pass of the model """
        s00 = torch.dot(self.e0[u_id], self.e0[i_id])

        adj_matrix_i = self.adj_matrix.T[i_id-graph.num_users]

        s01 = adj_matrix_i * (self.e0[u_id] @ self.e1[:graph.num_users].T)
        top_users = torch.argsort(s01, dim=0, descending=True)
        # s01_top = torch.take_along_dim(s01, top_users[:, :self.ku], dim=1)
        # s01_bot = torch.take_along_dim(s01, top_users[:, self.ku:], dim=1)
        s01_top = s01[top_users[:self.ku]]
        s01_bot = s01[top_users[self.ku:]]
        # s01 = torch.topk(s01, self.ku, dim=1, sorted=False)[0]
        s01_top = torch.sum(s01_top, dim=0)
        s01_bot = torch.sum(s01_bot, dim=0)
        s01_top = s01_top * self.deg[i_id]
        s01_bot = s01_bot * self.deg[i_id]

        adj_matrix_u = self.adj_matrix[u_id]

        s10 = adj_matrix_u * (self.e0[i_id] @ self.e1[graph.num_users:].T)
        top_items = torch.argsort(s10, dim=0, descending=True)
        # s10_top = torch.take_along_dim(s10, top_items[:, :self.ki], dim=1)
        # s10_bot = torch.take_along_dim(s10, top_items[:, self.ki:], dim=1)
        s10_top = s10[top_items[:self.ki]]
        s10_bot = s10[top_items[self.ki:]]
        top_items = top_items + graph.num_users
        # s10 = torch.topk(s10, self.ki, dim=1, sorted=False)[0]
        s10_top = torch.sum(s10_top, dim=0)
        s10_bot = torch.sum(s10_bot, dim=0)
        s10_top = s10_top * self.deg[u_id]
        s10_bot = s10_bot * self.deg[u_id]

        e1_u_top = torch.sum(
            (adj_matrix_u.view((-1, 1)) * self.e1[graph.num_users:])[top_items[:self.ki]-graph.num_users], 
            dim=0) * self.deg[u_id]
        e1_i_top = torch.sum(
            (adj_matrix_i.view((-1, 1)) * self.e1[:graph.num_users])[top_users[:self.ku]], 
            dim=0) * self.deg[i_id]
        e1_u = torch.sum(adj_matrix_u.view((-1, 1)) * self.e0[graph.num_users:], dim=0) * self.deg[u_id] * self.deg[i_id]
        e1_i = torch.sum(adj_matrix_i.view((-1, 1)) * self.e0[:graph.num_users], dim=0) * self.deg[u_id] * self.deg[i_id]
        s1_top = torch.dot(e1_u_top, e1_i_top)
        s1 = torch.dot(e1_u, e1_i)
        s1_bot = s1 - s1_top

        return 0.25*(s00 + s01_top + s10_top + s1_top), 0.25*(s00 + s01_bot + s10_bot + s1_bot), top_users, top_items

    def forward(self, graph, u_idx, i_idx):
        new_scores, scores_wo_top, top_users, top_items = [], [], [], []
        for u_id, i_id in zip(u_idx, i_idx):
            new_score, score_wo_top, top_u, top_i = self.one_forward(graph, u_id, i_id)
            new_scores.append(new_score)
            scores_wo_top.append(score_wo_top)
            top_users.append(top_u)
            top_items.append(top_i)
        new_scores = torch.stack(new_scores, dim=0)
        top_users = torch.stack(top_users, dim=0)
        top_items = torch.stack(top_items, dim=0)
        return new_scores, top_users, top_items


class GRACEAbsolute(InterpretableModel):
    """ GRACE model but with absolute values """

    def __init__(self, gnn: LightGCN_simple, graph: RecGraphData, ku: int, ki: int):
        self.gnn = gnn
        self.ku = ku
        self.ki = ki

        tol = 1e-12
        self.e0 = self.gnn.embedding.get_all(graph)
        self.deg = 1 / (torch.sqrt(graph.edge_index.sum(1)) + tol)
        self.e1 = torch.clone(self.e0)
        self.e1 = self.e1 * self.deg.unsqueeze(1)
        self.adj_matrix = graph.edge_index.cpu().to_dense()[:graph.num_users, graph.num_users:]
        self.adj_matrix = self.adj_matrix.to(torch.float32).to(self.e0.device)

    @torch.no_grad()
    def one_forward(self, graph, u_id, i_id) -> Tensor:
        """ One forward pass of the model """
        s00 = torch.dot(self.e0[u_id], self.e0[i_id])

        adj_matrix = self.adj_matrix.T[i_id-graph.num_users]

        s01 = adj_matrix * (self.e0[u_id] @ self.e1[:graph.num_users].T)
        top_users = torch.argsort(torch.abs(s01), dim=0, descending=True)
        # s01_top = torch.take_along_dim(s01, top_users[:, :self.ku], dim=1)
        # s01_bot = torch.take_along_dim(s01, top_users[:, self.ku:], dim=1)
        s01_top = s01[top_users[:self.ku]]
        s01_bot = s01[top_users[self.ku:]]
        # s01 = torch.topk(s01, self.ku, dim=1, sorted=False)[0]
        s01_top = torch.sum(s01_top, dim=0)
        s01_bot = torch.sum(s01_bot, dim=0)
        s01_top = s01_top * self.deg[i_id]
        s01_bot = s01_bot * self.deg[i_id]

        adj_matrix = self.adj_matrix[u_id]

        s10 = adj_matrix * (self.e0[i_id] @ self.e1[graph.num_users:].T)
        top_items = torch.argsort(torch.abs(s10), dim=0, descending=True)
        # s10_top = torch.take_along_dim(s10, top_items[:, :self.ki], dim=1)
        # s10_bot = torch.take_along_dim(s10, top_items[:, self.ki:], dim=1)
        s10_top = s10[top_items[:self.ki]]
        s10_bot = s10[top_items[self.ki:]]
        top_items = top_items + graph.num_users
        # s10 = torch.topk(s10, self.ki, dim=1, sorted=False)[0]
        s10_top = torch.sum(s10_top, dim=0)
        s10_bot = torch.sum(s10_bot, dim=0)
        s10_top = s10_top * self.deg[u_id]
        s10_bot = s10_bot * self.deg[u_id]

        return 0.25*(s00 + s01_top + s10_top), 0.25*(s00 + s01_bot + s10_bot), top_users, top_items

    def forward(self, graph, u_idx, i_idx):
        new_scores, scores_wo_top, top_users, top_items = [], [], [], []
        for u_id, i_id in zip(u_idx, i_idx):
            new_score, score_wo_top, top_u, top_i = self.one_forward(graph, u_id, i_id)
            new_scores.append(new_score)
            scores_wo_top.append(score_wo_top)
            top_users.append(top_u)
            top_items.append(top_i)
        new_scores = torch.stack(new_scores, dim=0)
        top_users = torch.stack(top_users, dim=0)
        top_items = torch.stack(top_items, dim=0)
        return new_scores, top_users, top_items
