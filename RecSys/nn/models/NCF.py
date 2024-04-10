import torch
from torch import nn
from RecSys.nn.models import CF
from RecSys.nn.models.embeddings import RecSysEmbedding


class NCF(CF):
    """ Neural Collaborative Filtering model """

    def __init__(self, embedding: RecSysEmbedding, num_layers: int, dropout: float):
        """
        Initialize the model by setting up the various layers.

        Args:
            embedding: embedding layer
            num_layers: the number of layers in MLP model;
            dropout: dropout rate between fully connected layers;

        """
        super(NCF, self).__init__()
        self.dropout = dropout

        self.embedding = embedding
        self.avg_weights = nn.parameter.Parameter(data=torch.tensor([1.0, 1.0]))

        MLP_modules = []
        input_size = int(2 * self.embedding.embedding_dim)
        for i in range(num_layers):
            MLP_modules.append(nn.Linear(input_size, input_size//2))
            # MLP_modules.append(nn.BatchNorm1d(input_size//2))
            MLP_modules.append(nn.ReLU())
            MLP_modules.append(nn.Dropout(p=self.dropout))
            input_size = input_size//2
        self.MLP_layers = nn.Sequential(*MLP_modules)

        self.predict_layer = nn.Linear(input_size, 1)

        self.reset_parameters()

    def reset_parameters(self):
        """ We leave the weights initialization here. """
        # self.embedding.reset_parameters()

        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')

        for m in self.modules():
            if isinstance(m, nn.Linear) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, graph, u_idx, i_idx):
        e_u = self.embedding(graph, u_idx)
        e_i = self.embedding(graph, i_idx)
        interaction = torch.cat([e_u, e_i], dim=-1)
        output_MLP = self.MLP_layers(interaction)

        prediction = self.predict_layer(output_MLP)
        return prediction.squeeze()

    def get_weight_norm(self):
        w_norm = self.embedding.get_weight_norm()
        for m in self.MLP_layers:
            if isinstance(m, nn.Linear):
                w_norm += m.weight.pow(2).sum()
        w_norm += self.predict_layer.weight.pow(2).sum()
        return w_norm
