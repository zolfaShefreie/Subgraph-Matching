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

        self.GNN_2 = GNNBaseLayer(hidden_units, hidden_units[-1], edge_dim, dropout_rate, aggregation_type,
                                  combination_type, normalize,
                                  name="graph_embed_layer_2")

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
        self.dense_2 = tf.keras.layers.Dense(1)
        self.output_layer = ThresholdLayer()

    def call(self, inputs, mask):
        x1, x2 = inputs
        mask1, mask2 = mask
        x1_aggregated = self.aggregator(self.attention([x1, x2], mask=[mask1, mask2]),
                                        mask=self.compute_mask([x1, x2], mask=[mask1, mask2]))
        x2_aggregated = self.aggregator(self.attention([x2, x1], mask=[mask2, mask1]),
                                        mask=self.compute_mask([x2, x1], mask=[mask2, mask1]))
        combined = self.dot_layer([x1_aggregated, x2_aggregated])
        combined = self.dense_1(combined)
        return self.output_layer(self.dense_2(combined))


class SearchSubgraph(tf.keras.models.Model):

    def __init__(self, max_nodes, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.attention = tf.keras.layers.Attention()
        self.aggregator = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(max_nodes), merge_mode='ave')
        self.subgraph_maker = RowChooserLayer(units=max_nodes)

    def call(self, inputs, mask=None):
        x1, x2 = inputs
        attention_scores, attention_mask = self.attention(inputs, mask=mask), self.attention.compute_mask(inputs, mask=mask)
        x1_aggregated = self.aggregator(attention_scores, mask=attention_mask)
        return self.subgraph_maker([x1, x1_aggregated], mask=None)


class SubgraphMatchingModel(tf.keras.models.Model):

    def __init__(self, graph_embed_model, subgraph_search_model, graph_matching_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph_embed_model = graph_embed_model
        self.subgraph_search_model = subgraph_search_model
        self.graph_matching_model = graph_matching_model
        self.padding_layer = PaddingLayer()

    def call(self, inputs):
        graph_1 = inputs[:, 0:1].merge_dims(0, 1)
        graph_2 = inputs[:, 1:2].merge_dims(0, 1)

        # graph embedding
        graph_1_embed = self.graph_embed_model(graph_1)
        graph_2_embed = self.graph_embed_model(graph_2)

        # padding and masking for graph embeds
        graph_1_embed, graph_1_masking = self.padding_layer(graph_1_embed), self.padding_layer.compute_output_mask(graph_1_embed)
        graph_2_embed, graph_2_masking = self.padding_layer(graph_2_embed), self.padding_layer.compute_output_mask(graph_2_embed)

        # choose subgraph
        subgraph_embed = self.subgraph_search_model([graph_1_embed, graph_2_embed], mask=[graph_1_masking, graph_2_masking])
        subgraph_embed, subgraph_masking = self.padding_layer(subgraph_embed), self.padding_layer.compute_output_mask(subgraph_embed)

        return self.graph_matching_model((graph_1_embed, subgraph_embed))



# class SubgraphSearchEncoder(tf.keras.Model):
#
#     def __init__(self, max_nodes, *args, **kwargs):
#         super(SubgraphSearchEncoder, self).__init__(*args, **kwargs)
#
#         self.subgraph_maker = RowChooserLayer(max_nodes)
#         self.flatten_1 = tf.keras.layers.Flatten()
#         self.dense_2 = tf.keras.layers.Dense(max_nodes, activation='relu')
#         self.dense_3 = tf.keras.layers.Dense(max_nodes, activation='relu')
#
#     def call(self, inputs, training=None, mask=None):
#         sub_input = self.subgraph_maker(inputs)
#         x = self.flatten_1(sub_input)
#         x = self.dense_2(x)
#         return sub_input, self.dense_3(x)
#
#
# class SubgraphSearchDecoder(tf.keras.models.Model):
#
#     def __init__(self, max_nodes, node_feature_dim, *args, **kwargs):
#         super(SubgraphSearchDecoder, self).__init__(*args, **kwargs)
#         self.dense_1 = tf.keras.layers.Dense(max_nodes)
#         self.dense_2 = tf.keras.layers.Dense(max_nodes*node_feature_dim)
#         self.dense_3 = tf.keras.layers.Dense(max_nodes * node_feature_dim)
#         self.output_layer = tf.keras.layers.Reshape(target_shape=(max_nodes, node_feature_dim))
#
#     def call(self, inputs, training=None, mask=None):
#         x = self.dense_1(inputs)
#         x = self.dense_2(x)
#         x = self.dense_3(x)
#         return self.output_layer(x)


# class PredictIsSubgraphModel(tf.keras.models.Model):
#
#     def __init__(self, graph_embed, encoder, decoder, graph_matching_model, *args, **kwargs):
#         super(PredictIsSubgraphModel, self).__init__(*args, **kwargs)
#
#         # self.graph_embed = GraphEmbeddingModel(hidden_units, dropout_rate=dropout_rate,
#         #                                        aggregation_type=aggregation_type,
#         #                                        combination_type=combination_type,
#         #                                        normalize=normalize)
#         # self.encoder = SubgraphSearchEncoder(max_nodes)
#         # self.decoder = SubgraphSearchDecoder(max_nodes=max_nodes, node_feature_dim=hidden_units[-1])
#
#         self.graph_embed = graph_embed
#         self.encoder = encoder
#         self.decoder = decoder
#         self.graph_matching_model = graph_matching_model
#
#         self.graph_loss = tf.keras.losses.mse
#         self.label_loss = tf.keras.losses.SDG
#
#         self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
#         self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
#         self.accuracy_tracker = tf.keras.metrics.Accuracy(name="accuracy")
#
#         self.label_predict = None
#         # self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
#
#     # def call(self, inputs):
#     #     graph_1 = inputs[:, 0:1].merge_dims(0, 1)
#     #     graph_2 = inputs[:, 1:2].merge_dims(0, 1)
#     #
#     #     embed_graph_1 = self.graph_embed(graph_1)
#     #     embed_graph_2 = self.graph_embed(graph_2)
#     #     subgraph_embed, z = self.encoder(embed_graph_1)
#     #
#
#     def train_step(self, data):
#         x_train, y_train = data
#         graph_1 = x_train[:, 0:1].merge_dims(0, 1)
#         graph_2 = x_train[:, 1:2].merge_dims(0, 1)
#         with tf.GradientTape() as tape:
#             embed_graph_1 = self.graph_embed(graph_1)
#             embed_graph_2 = self.graph_embed(graph_2)
#
#             subgraph_embed, z = self.encoder(embed_graph_1)
#             graph_2_reconstruction = self.decoder(z)
#             reconstruction_loss = tf.reduce_mean(tf.reduce_sum(self.graph_loss(embed_graph_2, graph_2_reconstruction),
#                                                                axis=(1, )))
#             output = self.graph_matching_model((subgraph_embed, embed_graph_2))
#             label_loss = self.label_loss(y_train, output)
#
#             total_loss = reconstruction_loss + label_loss
#         grads = tape.gradient(total_loss, self.trainable_weights)
#         self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
#         self.total_loss_tracker.update_state(total_loss)
#         self.reconstruction_loss_tracker.update_state(reconstruction_loss)
#         return {
#             "loss": self.total_loss_tracker.result(),
#             "reconstruction_loss": self.reconstruction_loss_tracker.result(),
#         }
