import pandas as pd
from RecSys.nn.models.embeddings import IdEmbedding
from RecSys.utils.data.data import RecGraphData


def test_IdEmbedding():
    """ Check IdEmbedding class """
    num_embeddings = 10
    embedding_dim = 5
    embedding = IdEmbedding(num_embeddings, embedding_dim)

    # Check the sizes
    assert embedding.num_embeddings == num_embeddings
    assert embedding.embedding_dim == embedding_dim
    assert embedding.embedding.weight.size() == (num_embeddings, embedding_dim)
    assert embedding.embedding.weight.requires_grad

    users_df = pd.read_csv('data/tomplay/users.csv')
    items_df = pd.read_csv('data/tomplay/items.csv')
    inter_df = pd.read_csv('data/tomplay/split/interactions_train.csv')
    graph = RecGraphData(users_df, items_df, inter_df, inter_df)
    embedding = IdEmbedding.from_graph(graph, 64)
    # Check the sizes
    assert embedding.num_embeddings == graph.num_users + graph.num_items
    assert embedding.embedding_dim == 64
    assert embedding.embedding.weight.size() == (graph.num_users + graph.num_items, 64)
    assert embedding.embedding.weight.requires_grad
