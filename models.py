import tensorflow as tf
from layers import GNNBaseLayer, RowChooserLayer
import keras


class GraphEmbeddingModel(tf.keras.Model):

    def __int__(self, hidden_units, dropout_rate=0.2, aggregation_type="mean", combination_type="concat",
                normalize=False, *args, **kwargs):
        super(GraphEmbeddingModel, self).__init__(*args, **kwargs)

        self.GNN_1 = GNNBaseLayer(hidden_units, dropout_rate, aggregation_type, combination_type, normalize,
                                  name="graph_embed_layer_1")

        self.GNN_2 = GNNBaseLayer(hidden_units, dropout_rate, aggregation_type, combination_type, normalize,
                                  name="graph_embed_layer_2")

    def call(self, inputs, training=None, mask=None):
        node_features, edges, edge_weights = inputs
        embed_1 = self.GNN_1((node_features, edges, edge_weights))
        return self.GNN_2((embed_1, edges, edge_weights))


class SubgraphSearchEncoder(tf.keras.Model):

    def __init__(self, max_nodes, *args, **kwargs):
        super(SubgraphSearchEncoder, self).__init__(*args, **kwargs)

        self.subgraph_maker = RowChooserLayer(max_nodes)
        self.flatten_1 = tf.keras.layers.Flatten()
        self.dense_2 = tf.keras.layers.Dense(max_nodes, activation='relu')
        self.dense_3 = tf.keras.layers.Dense(max_nodes, activation='relu')

    def call(self, inputs, training=None, mask=None):
        sub_input = self.subgraph_maker(inputs)
        x = self.flatten_1(sub_input)
        x = self.dense_2(x)
        return sub_input, self.dense_3(x)


class SubgraphSearchDecoder(tf.keras.models.Model):

    def __init__(self, max_nodes, *args, **kwargs):
        super(SubgraphSearchDecoder, self).__init__(*args, **kwargs)


class PredictIsSubgraphModel(tf.keras.models.Mode):
    pass
