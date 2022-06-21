from dataset_process import Dataset
import models


if __name__ == '__main__':
    dataset_obj = Dataset('AIDS')
    x, y = dataset_obj.load_dataset()
    print("done")
    subgraph_matching_model = models.SubgraphMatchingModel(hidden_units=32, node_dim=dataset_obj.node_attr_dim,
                                                           edge_dim=dataset_obj.edge_attr_dim,
                                                           max_nodes=dataset_obj.max_graph_nodes,
                                                           node_embed_dim=32, name="SubgraphMatchingModel")
    subgraph_matching_model.compile(optimizer='sgd', loss="categorical_crossentropy", metrics=['accuracy'], run_eagerly=True)
    subgraph_matching_model.fit(x, y)
