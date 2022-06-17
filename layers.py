import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras


class GNNBaseLayer(tf.keras.layers.Layer):

    def __init__(self, hidden_units, dropout_rate=0.2, aggregation_type="mean", combination_type="concat",
                 normalize=False, *args, **kwargs):
        """
        :param hidden_units: units for dense layers
        :param dropout_rate:
        :param aggregation_type: how to aggregate neighbor messages.it can be sum, mean, and max.
        :param combination_type: how to combine source node and neighbor messages it can be concat, gru, and add
        :param normalize: normalize the output or not
        :param args:
        :param kwargs:
        """
        super(GNNBaseLayer, self).__init__(*args, **kwargs)

        self.aggregation_type = aggregation_type
        self.combination_type = combination_type
        self.normalize = normalize

        self.base_message_create = self.create_gnn_layers(hidden_units, dropout_rate)
        self.edge_transformer = self.create_edge_transformer_layers(hidden_units=hidden_units)
        if self.combination_type == "gru":
            self.update_fn = layers.GRU(units=hidden_units, activation="tanh", recurrent_activation="sigmoid",
                                        dropout=dropout_rate, return_state=True, recurrent_dropout=dropout_rate)
        else:
            self.update_fn = self.create_gnn_layers(hidden_units, dropout_rate)

    @staticmethod
    def create_gnn_layers(hidden_units, dropout_rate, use_attention=True, name=None):
        """
        create gnn layers (it use for node features transformation)
        :param hidden_units:
        :param dropout_rate:
        :param use_attention: add attention layer or not
        :param name: name of box
        :return:
        """
        gnn_layers = []

        for units in hidden_units:
            gnn_layers.append(layers.BatchNormalization())
            gnn_layers.append(layers.Dropout(dropout_rate))
            if use_attention:
                gnn_layers.append(layers.Attention(use_scale=True))
            gnn_layers.append(layers.Dense(units, activation=tf.nn.gelu))

        return keras.Sequential(gnn_layers, name=name)

    @staticmethod
    def create_edge_transformer_layers(hidden_units, name=None):
        """
        create layers for edge feature transformation
        :param hidden_units:
        :param name: name of box
        :return:
        """
        edge_embed_layers = list()
        for units in hidden_units:
            edge_embed_layers.append(layers.Dense(units, activation=tf.nn.gelu))
        return keras.Sequential(layers, name=name)

    def prepare_neighbour_messages(self, node_representations, edge_features=None):
        """
        get base message of nodes and apply edge features representations to base_messages
        :param node_representations:
        :param edge_features:
        :return:
        """
        messages = self.base_message_create(node_representations)
        if edge_features is not None:
            edge_representations = self.edge_transformer(edge_features)
            messages = messages * edge_representations
        return messages

    def aggregate(self, node_indices, neighbour_messages):
        """
        aggregate neighbour messages
        :param node_indices: source nodes indices is a 1d array with length of number of edge
                             that shows the with neighbour nodes must aggregate together.
        :param neighbour_messages: is a 2d array with shape of (number of edges, units)
        :return:
        """

        num_nodes = tf.math.reduce_max(node_indices) + 1
        if self.aggregation_type == "sum":
            aggregated_message = tf.math.unsorted_segment_sum(neighbour_messages, node_indices, num_segments=num_nodes)
        elif self.aggregation_type == "mean":
            aggregated_message = tf.math.unsorted_segment_mean(neighbour_messages, node_indices, num_segments=num_nodes)
        elif self.aggregation_type == "max":
            aggregated_message = tf.math.unsorted_segment_max(neighbour_messages, node_indices, num_segments=num_nodes)
        else:
            raise ValueError(f"Invalid aggregation type: {self.aggregation_type}.")

        return aggregated_message

    def update(self, node_representations, aggregated_messages):
        """
        aggregate the each node_representations with its neighbour messages
        :param node_representations:
        :param aggregated_messages:
        :return:
        """
        if self.combination_type == "gru":
            # Create a sequence of two elements for the GRU layer.
            h = tf.stack([node_representations, aggregated_messages], axis=1)
        elif self.combination_type == "concat":
            # Concatenate the node_representations and aggregated_messages.
            h = tf.concat([node_representations, aggregated_messages], axis=1)
        elif self.combination_type == "add":
            # Add node_representations and aggregated_messages.
            h = node_representations + aggregated_messages
        else:
            raise ValueError(f"Invalid combination type: {self.combination_type}.")

        # Apply the processing function.
        node_embeddings = self.update_fn(h)
        if self.combination_type == "gru":
            node_embeddings = tf.unstack(node_embeddings, axis=1)[-1]

        if self.normalize:
            node_embeddings = tf.nn.l2_normalize(node_embeddings, axis=-1)
        return node_embeddings

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def call(self, inputs, *args, **kwargs):
        """
        Process the inputs to produce the graph_embedding.
        :param inputs: graph => tuple with tree elements
                                1. node features
                                2. edges => ([source_node], [neighbor_node])
                                3. edge_features
        :return: graph embedding
        """

        node_representations, edges, edge_features = inputs
        node_indices, neighbour_indices = edges[0], edges[1]
        neighbour_representations = tf.gather(node_representations, neighbour_indices)
        neighbour_messages = self.prepare_neighbour_messages(neighbour_representations, edge_features)
        aggregated_messages = self.aggregate(node_indices, neighbour_messages)
        return self.update(node_representations, aggregated_messages)


class ThresholdLayer(tf.keras.layers.Layer):

    def __init__(self, *args, **kwargs):
        super(ThresholdLayer, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name="threshold", shape=(1,), initializer="uniform",
                                      trainable=True)
        super(ThresholdLayer, self).build(input_shape)

    def call(self, x):
        """
        :param x:
        :return:
        """
        return keras.backend.sigmoid(100 * (x - self.kernel))

    def compute_output_shape(self, input_shape):
        return input_shape


class BinaryLayer(tf.keras.layers.Layer):
    """
    binary output with trainable threshold
    """

    def __init__(self, *args, **kwargs):
        super(BinaryLayer, self).__init__(*args, **kwargs)
        self.kernel = self.add_weight(name="threshold", shape=(1,), initializer="uniform",
                                      trainable=True)

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def call(self, inputs, *args, **kwargs):
        return keras.backend.cast(keras.backend.greater(inputs, self.kernel), keras.backend.floatx())


class RowChooserLayer(tf.keras.layers.Layer):

    def __init__(self, units, *args, **kwargs):
        """
        :param units: units of Dense it must be the maximum length in input
        :param args:
        :param kwargs:
        """
        super(RowChooserLayer, self).__init__(*args, **kwargs)

        self.binary_maker = BinaryLayer()
        self.dense_layer = tf.keras.layers.Dense(units, activation="sigmoid")
    
    @staticmethod
    def get_new_shape(pre_shape):
        if len(pre_shape) == 3:
            return pre_shape[0], -1, pre_shape[2]
        if len(pre_shape) == 2:
            return -1, pre_shape[-1]
    
    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def call(self, inputs, *args, **kwargs):
        """
        :param inputs:
        :param args:
        :param kwargs:
        :return:
        """
        x = self.dense_layer(inputs)
        row_index = self.binary_maker(x)
        row_index = tf.expand_dims(row_index, -1)
        row_index = tf.repeat(row_index, repeats=inputs.shape[-1], axis=-1)
        bool_index = tf.math.not_equal(row_index, 0.0)
        shape = tf.shape(inputs)
        output = inputs[bool_index]
        return tf.reshape(output, self.get_new_shape(shape))


