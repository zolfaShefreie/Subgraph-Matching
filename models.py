import tensorflow as tf
from layers import GNNBaseLayer, RowChooserLayer, ThresholdLayer, PaddingLayer
import keras


class GraphEmbeddingModel(tf.keras.Model):
    """
    this module uses 2 gnn layer to embed graph
    """

    def __init__(self, hidden_units, node_dim, edge_dim, dropout_rate=0.2, aggregation_type="mean",
                 combination_type="concat", normalize=False, *args, **kwargs):
        super(GraphEmbeddingModel, self).__init__(*args, **kwargs)

        self.GNN_1 = GNNBaseLayer(hidden_units, node_dim, edge_dim, dropout_rate, aggregation_type, combination_type,
                                  normalize,
                                  name="graph_embed_layer_1")

        self.GNN_2 = GNNBaseLayer(hidden_units, hidden_units, edge_dim, dropout_rate, aggregation_type,
                                  combination_type, normalize,
                                  name="graph_embed_layer_2")

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def call(self, inputs, training=None, mask=None):
        # node_features, edges, edge_weights = inputs
        embed_1 = self.GNN_1(inputs)
        return self.GNN_2(tf.concat([tf.expand_dims(embed_1, 1), inputs[:, 1:3]], 1))


class GraphMatchingModel(tf.keras.models.Model):

    def __init__(self, lstm_units=20, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.attention = tf.keras.layers.Attention()
        self.aggregator = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units), merge_mode='ave')
        self.dot_layer = tf.keras.layers.Dot(axes=1)
        self.dense_1 = tf.keras.layers.Dense(10, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(1, activation='relu')
        self.output_layer = ThresholdLayer()

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def call(self, inputs, mask):
        x1, x2 = inputs
        mask1, mask2 = mask
        x1_aggregated = self.aggregator(self.attention([x1, x2],
                                                       mask=[tf.reduce_any(mask1, 2), tf.reduce_any(mask2, 2)]),
                                        mask=self.attention.compute_mask([x1, x2],
                                                                         mask=[tf.reduce_any(mask1, 2),
                                                                               tf.reduce_any(mask2, 2)]))
        x2_aggregated = self.aggregator(self.attention([x2, x1],
                                                       mask=[tf.reduce_any(mask2, 2), tf.reduce_any(mask1, 2)]),
                                        mask=self.attention.compute_mask([x2, x1],
                                                                         mask=[tf.reduce_any(mask2, 2),
                                                                               tf.reduce_any(mask1, 2)]))
        combined = self.dot_layer([x1_aggregated, x2_aggregated])
        combined = self.dense_1(combined)
        return self.output_layer(self.dense_2(combined))


class SearchSubgraph(tf.keras.models.Model):
    """work based on masks and the main result is new mask"""

    def __init__(self, max_nodes, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.attention = tf.keras.layers.Attention()
        self.aggregator = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(max_nodes), merge_mode='ave',
                                                        name='search_subgraph_aggregator')
        self.subgraph_maker = RowChooserLayer(units=max_nodes)

    def call(self, inputs, mask):
        x1, x2 = inputs
        return x1

    def compute_mask(self, inputs, mask=None):
        x1, x2 = inputs
        mask1, mask2 = mask
        _, attention_scores = self.attention([x1, x2], mask=[tf.reduce_any(mask1, 2), tf.reduce_any(mask2, 2)],
                                             return_attention_scores=True)
        attention_mask = self.attention.compute_mask([x1, x2], mask=[tf.reduce_any(mask1, 2), tf.reduce_any(mask2, 2)])
        x1_aggregated = self.aggregator(attention_scores, mask=attention_mask)
        return self.subgraph_maker.compute_output_mask([x1, x1_aggregated], mask=[mask1, attention_mask])


class SubgraphMatchingModel(tf.keras.models.Model):

    def __init__(self, hidden_units, node_dim, edge_dim, max_nodes, node_embed_dim, dropout_rate=0.2,
                 aggregation_type="mean",
                 combination_type="concat", normalize=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph_embed_model = GraphEmbeddingModel(hidden_units=hidden_units, node_dim=node_dim,
                                                     edge_dim=edge_dim, dropout_rate=dropout_rate,
                                                     aggregation_type=aggregation_type,
                                                     combination_type=combination_type, normalize=normalize,
                                                     name="GraphEmbeddingModel")
        self.subgraph_search_model = SearchSubgraph(max_nodes=max_nodes, name="SearchSubgraph")
        self.graph_matching_model = GraphMatchingModel(name='GraphMatchingModel')
        self.max_nodes = max_nodes
        self.node_embed_dim = node_embed_dim

    def call(self, inputs):
        padding_layer = PaddingLayer((inputs.shape[0], self.max_nodes, self.node_embed_dim))
        graph_1 = inputs[:, 0:1].merge_dims(0, 1)
        graph_2 = inputs[:, 1:2].merge_dims(0, 1)

        # graph embedding
        graph_1_embed = self.graph_embed_model(graph_1)
        graph_2_embed = self.graph_embed_model(graph_2)

        # padding and masking for graph embeds
        graph_1_embed, graph_1_masking = padding_layer(graph_1_embed), padding_layer.compute_output_mask(graph_1_embed)
        graph_2_embed, graph_2_masking = padding_layer(graph_2_embed), padding_layer.compute_output_mask(graph_2_embed)
        # choose subgraph
        subgraph_embed_mask = self.subgraph_search_model.compute_mask([graph_1_embed, graph_2_embed],
                                                                      mask=[graph_1_masking, graph_2_masking])
        return self.graph_matching_model([graph_1_embed, graph_2_embed], mask=[subgraph_embed_mask, graph_2_masking])
