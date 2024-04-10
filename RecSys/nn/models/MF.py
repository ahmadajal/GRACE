import torch
import torch.nn as nn
from tqdm import tqdm
from RecSys.nn.models import CF
from RecSys.nn.models.embeddings import RecSysEmbedding


class MF(CF):
    """ Matrix Factorization model """

    def __init__(self, embedding: RecSysEmbedding, *args, **kwargs):
        """
        Initialize the model by setting up the various layers.

        Args:
            embedding: embedding layer

        """
        super(MF, self).__init__()
        self.embedding = embedding

        self.reset_parameters()

    def forward(self, graph, u_idx, i_idx):
        e_u = self.embedding(graph, u_idx)
        e_i = self.embedding(graph, i_idx)
        prediction = torch.sum(e_u * e_i, dim=-1)
        return prediction

    def get_all_scores(self, graph):
        bs = 256
        e_i = self.embedding(graph, torch.arange(graph.num_users, graph.num_users+graph.num_items, device=self.embedding.embedding.weight.device))  # type: ignore
        e_u = self.embedding(graph, torch.arange(graph.num_users, device=self.embedding.embedding.weight.device))  # type: ignore
        scores = torch.zeros((graph.num_users, graph.num_items))
        for u in tqdm(range(0, graph.num_users, bs)):
            s_u = torch.matmul(e_u[u: min(u+bs, graph.num_users)], e_i.t()).cpu()
            scores[u: min(u+bs, graph.num_users), :] = s_u
        return scores

    def get_weight_norm(self):
        return torch.norm(self.embedding.embedding.weight)  # type: ignore

    def reset_parameters(self):
        """ Initialize the weights with N(0, 0.01) """
        torch.nn.init.normal_(self.embedding.embedding.weight, std=0.01)  # type: ignore


class MFBiases(CF):
    """ Matrix Factorization model with biases """

    def __init__(self, embedding: RecSysEmbedding, *args, **kwargs):
        """
        Initialize the model by setting up the various layers.

        Args:
            embedding: embedding layer

        """
        super(MFBiases, self).__init__()
        self.embedding = embedding
        self.bias = nn.parameter.Parameter(torch.zeros(1))
        self.node_bias = nn.parameter.Parameter(torch.zeros(embedding.num_embeddings))

        self.reset_parameters()

    def forward(self, graph, u_idx, i_idx):
        e_u = self.embedding(graph, u_idx)
        e_i = self.embedding(graph, i_idx)
        prediction = torch.sum(e_u * e_i, dim=-1) + self.bias + self.node_bias[u_idx] + self.node_bias[i_idx]
        return prediction

    def get_weight_norm(self):
        return torch.norm(self.embedding.embedding.weight)  # type: ignore

    def reset_parameters(self):
        """ Initialize the weights with N(0, 0.01) """
        torch.nn.init.normal_(self.embedding.embedding.weight, std=0.01)  # type: ignore
        torch.nn.init.normal_(self.bias, std=0.01)
        torch.nn.init.normal_(self.node_bias, std=0.01)
