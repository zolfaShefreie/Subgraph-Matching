from dataset_process import Dataset
import models


if __name__ == '__main__':
    dataset_obj = Dataset('AIDS')
    # x, y = dataset_obj.load_dataset()

    graph_embed_model = models.GraphEmbeddingModel(hidden_units=[32, 32], node_dim=dataset_obj.node_attr_dim,
                                                   edge_dim=dataset_obj.edge_attr_dim)
    subgraph_searching_model = models.SearchSubgraph(max_nodes=dataset_obj.max_graph_nodes)
    graph_matching_model = models.GraphMatchingModel()
    subgraph_matching_model = models.SubgraphMatchingModel(graph_embed_model=graph_embed_model,
                                                           subgraph_search_model=subgraph_searching_model, 
                                                           graph_matching_model=graph_matching_model)

    subgraph_matching_model.compile(optimizer='sgd', loss="mse", metrics=['accuracy'])
