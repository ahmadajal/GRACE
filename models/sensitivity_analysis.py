""" Sensitivity Analysis module """
from torch import Tensor
import torch
import torch_sparse
from RecSys.utils.data.data import RecGraphData
from models import InterpretableModel
from RecSys.nn.models.LightGCN import LightGCN_simple


def get_neighbors(node_id: int, graph: RecGraphData) -> Tensor:
    """ Returns the list of neighbors of a node in a graph """
    neighbors = torch.cat(
        (torch.unique(graph.directed_edge_index[1][graph.directed_edge_index[0] == node_id]),
         torch.unique(graph.directed_edge_index[0][graph.directed_edge_index[1] == node_id])), dim=0)
    neighbors = torch.sort(torch.unique(neighbors))[0]
    return neighbors


def get_pair_subgraph(graph: RecGraphData, u_id: int, i_id: int) -> RecGraphData:
    """ Get the 1-hop subgraph induced by the pair (u, i)

    Args:
        graph: the graph
        u_id: the user id
        i_id: the item id

    Returns:
        The 1-hop subgraph induced by the pair (u, i)
        The index of u in the subgraph
        The index of i in the subgraph
    """
    neighbors = torch.cat((
        torch.tensor([u_id], device=graph.x.device),
        torch.tensor([i_id], device=graph.x.device),
        get_neighbors(u_id, graph),
        get_neighbors(i_id, graph)), dim=0)
    neighbors = torch.sort(torch.unique(neighbors))[0]
    subgraph = graph.subgraph(neighbors)
    idx_u = torch.where(subgraph.x[:, 0] == u_id)[0][0]
    idx_i = torch.where(subgraph.x[:, 0] == i_id)[0][0]
    return subgraph, idx_u, idx_i


class SensitivityAnalysis(InterpretableModel):
    """ Sensitivity Analysis class """

    def __init__(self, gnn: LightGCN_simple, ku: int, ki: int):
        self.gnn = gnn
        self.ku = ku
        self.ki = ki

    @torch.no_grad()
    def one_forward(self, graph, u_id, i_id):
        """ One forward pass of the GNNExplainer """
        # Get the subgraph
        subgraph, n_u_id, n_i_id = get_pair_subgraph(graph, u_id, i_id)

        # Get the normalized adjacency matrix
        adj_mat = subgraph.edge_index.coalesce().to_dense()
        deg = torch_sparse.sum(graph.edge_index, 1)[subgraph.x[:, 0]]
        deg = 1 / (1e-12 + torch.sqrt(deg))
        adj_mat = adj_mat * deg.view(-1, 1)  # torch.stack([deg*adj_mat[i] for i in range(adj_mat.shape[0])], 0)
        adj_mat = adj_mat * deg.view(1, -1)  # torch.stack([deg*adj_mat[:, i] for i in range(adj_mat.shape[1])], 1)
        # deg = 1 / (1e-12 + torch.sqrt(deg))

        top_users = torch.argsort(-deg)
        top_items = torch.argsort(-deg)
        top_users = top_users[subgraph.x[top_users, 0] < graph.num_users]
        top_items = top_items[subgraph.x[top_items, 0] >= graph.num_users]

        new_score = self.get_new_score(subgraph, n_u_id, n_i_id, adj_mat, top_users[:self.ku], top_items[:self.ki])
        score_wo_top = self.get_score_without_top(subgraph, n_u_id, n_i_id, adj_mat, top_users[:self.ku], top_items[:self.ki])

        return new_score, score_wo_top, top_users, top_items

    def get_new_score(self, subgraph, n_idx_u, n_idx_i, adj_mat, top_users, top_items):
        """ Get the new score """
        mask = torch.zeros_like(adj_mat, device=subgraph.x.device)
        mask[n_idx_i, top_users] = 1
        mask[n_idx_u, top_items] = 1
        with torch.no_grad():
            subgraph.edge_index = torch.clone(adj_mat) * mask
            e0 = self.gnn.embedding.get_all(subgraph)
            e1 = torch.matmul(subgraph.edge_index, e0)
            eu = 0.5*(e0[n_idx_u] + e1[n_idx_u])
            ei = 0.5*(e0[n_idx_i] + e1[n_idx_i])
            y = torch.dot(eu, ei)
        return y

    def get_score_without_top(self, subgraph, n_idx_u, n_idx_i, adj_mat, top_users, top_items):
        """ Get the score without the top users and items """
        mask = torch.ones_like(adj_mat, device=subgraph.x.device)
        mask[n_idx_i, top_users] = 0
        mask[n_idx_u, top_items] = 0
        with torch.no_grad():
            subgraph.edge_index = torch.clone(adj_mat) * mask
            e0 = self.gnn.embedding.get_all(subgraph)
            e1 = torch.matmul(subgraph.edge_index, e0)
            eu = 0.5*(e0[n_idx_u] + e1[n_idx_u])
            ei = 0.5*(e0[n_idx_i] + e1[n_idx_i])
            y = torch.dot(eu, ei)
        return y

    @torch.no_grad()
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
