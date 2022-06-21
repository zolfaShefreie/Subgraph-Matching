import tensorflow as tf
# from tensorflow import keras
import numpy as np


class GraphElementEmbedLayer(tf.keras.layers.Layer):
    def __init__(self, sequential_model, attr_dim, new_attr_dim=0, *args, **kwargs):
        super(GraphElementEmbedLayer, self).__init__(*args, **kwargs)
        self.sequential_model = sequential_model
        # self.reshape = tf.keras.layers.Reshape((old_attr_dim, ))
        self.attr_dim = attr_dim
        self.new_attr_dim = new_attr_dim

    def call(self, inputs):
        element_sizes = [len(each) for each in inputs.to_list()]
        inputs = inputs.merge_dims(0, 1)
        inputs = tf.sparse.to_dense(inputs.to_sparse(), default_value=0)
        inputs = tf.reshape(inputs, (-1, self.attr_dim))
        embedded_result = self.sequential_model(inputs)
        split_result = tf.split(embedded_result, element_sizes, 0)
        return tf.ragged.constant(list([each.numpy().tolist() for each in split_result]))


class NodeEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, units, dropout_rate, attr_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attr_dim = attr_dim
        self.batch_normalization_1 = tf.keras.layers.BatchNormalization()
        self.dropout_1 = tf.keras.layers.Dropout(dropout_rate)
        self.dense_1 = tf.keras.layers.Dense(units, activation=tf.nn.gelu)
        self.batch_normalization_2 = tf.keras.layers.BatchNormalization()
        self.dropout_2 = tf.keras.layers.Dropout(dropout_rate)
        self.dense_2 = tf.keras.layers.Dense(units, activation=tf.nn.gelu)

    def call(self, inputs):
        element_sizes = inputs.row_splits
        inputs = inputs.merge_dims(0, 1)
        inputs = tf.sparse.to_dense(inputs.to_sparse(), default_value=0)
        inputs = tf.reshape(inputs, (-1, self.attr_dim))
        embedded_result = self.batch_normalization_1(inputs)
        embedded_result = self.dropout_1(embedded_result)
        embedded_result = self.dense_1(embedded_result)
        embedded_result = self.batch_normalization_2(embedded_result)
        embedded_result = self.dropout_2(embedded_result)
        embedded_result = self.dense_2(embedded_result)
        return tf.RaggedTensor.from_row_splits(embedded_result, element_sizes)


class EdgeEmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, units, dropout_rate, attr_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attr_dim = attr_dim
        self.dense_1 = tf.keras.layers.Dense(units, activation=tf.nn.gelu)
        self.dropout_1 = tf.keras.layers.Dropout(dropout_rate)
        self.dense_2 = tf.keras.layers.Dense(units, activation=tf.nn.gelu)

    def call(self, inputs):
        element_sizes = inputs.row_splits
        inputs = inputs.merge_dims(0, 1)
        inputs = tf.sparse.to_dense(inputs.to_sparse(), default_value=0)
        inputs = tf.reshape(inputs, (-1, self.attr_dim))
        embedded_result = self.dense_1(inputs)
        embedded_result = self.dropout_1(embedded_result)
        embedded_result = self.dense_2(embedded_result)
        return tf.RaggedTensor.from_row_splits(embedded_result, element_sizes)


class GNNBaseLayer(tf.keras.layers.Layer):

    def __init__(self, hidden_units, node_dim, edge_dim, dropout_rate=0.2, aggregation_type="mean",
                 combination_type="concat", normalize=False, *args, **kwargs):
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
        self.edge_dim = edge_dim
        self.node_dim = node_dim
        self.base_message_create = NodeEmbeddingLayer(units=hidden_units, attr_dim=node_dim,
                                                      dropout_rate=dropout_rate, name="base_message_create")
        self.edge_transformer = EdgeEmbeddingLayer(units=hidden_units, attr_dim=edge_dim,
                                                   dropout_rate=dropout_rate, name="edge_transformer")
        if self.combination_type == "gru":
            self.update_fn = tf.keras.layers.GRU(units=hidden_units, activation="tanh", recurrent_activation="sigmoid",
                                                 dropout=dropout_rate, return_state=True,
                                                 recurrent_dropout=dropout_rate)
        else:
            self.update_fn = NodeEmbeddingLayer(units=hidden_units,
                                                attr_dim=node_dim + hidden_units if self.combination_type == 'concat' else hidden_units,
                                                dropout_rate=dropout_rate,
                                                name="update_fn")

    def prepare_neighbour_messages(self, node_representations, edge_features=None):
        """
        get base message of nodes and apply edge features representations to base_messages
        :param node_representations:
        :param edge_features:
        :return:
        """
        messages = self.base_message_create(node_representations)
        if edge_features is not None and self.edge_dim != 0:
            edge_representations = self.edge_transformer(edge_features)
            messages = messages * edge_representations
        return messages

    def aggregate(self, node_indices, neighbour_messages, number_nodes):
        """
        aggregate neighbour messages
        :param number_nodes: number of nodes for each input
        :param node_indices: source nodes indices is a 1d array with length of number of edge
                             that shows the with neighbour nodes must aggregate together.
        :param neighbour_messages: is a 2d array with shape of (number of edges, units)
        :return:
        """
        all_aggregated_message = None
        # all_aggregated_message = list()
        # for each in batch
        for graph_node_indices, graph_neighbour_messages, num_nodes in zip(node_indices, neighbour_messages,
                                                                           number_nodes):
            if self.aggregation_type == "sum":
                aggregated_message = tf.math.unsorted_segment_sum(graph_neighbour_messages,
                                                                  graph_node_indices,
                                                                  num_segments=num_nodes)
            elif self.aggregation_type == "mean":
                aggregated_message = tf.math.unsorted_segment_mean(graph_neighbour_messages,
                                                                   graph_node_indices,
                                                                   num_segments=num_nodes)
            elif self.aggregation_type == "max":
                aggregated_message = tf.math.unsorted_segment_max(graph_neighbour_messages,
                                                                  graph_node_indices,
                                                                  num_segments=num_nodes)
            else:
                raise ValueError(f"Invalid aggregation type: {self.aggregation_type}.")

            aggregated_message_pad = aggregated_message
            if all_aggregated_message is None:
                all_aggregated_message = tf.expand_dims(aggregated_message_pad, 0)
            else:
                all_aggregated_message = tf.ragged.stack([all_aggregated_message,
                                                          tf.expand_dims(aggregated_message_pad, 0)]).merge_dims(0, 1)
        return all_aggregated_message

    def update(self, node_representations, aggregated_messages):
        """
        aggregate the each node_representations with its neighbour messages
        :param node_representations:
        :param aggregated_messages:
        :return:
        """
        if self.combination_type == "gru":
            # Create a sequence of two elements for the GRU layer.
            h = tf.stack([node_representations, aggregated_messages], axis=2)
        elif self.combination_type == "concat":
            # Concatenate the node_representations and aggregated_messages.
            h = tf.concat([node_representations, aggregated_messages], axis=2)
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

    def call(self, inputs):
        """
        Process the inputs to produce the graph_embedding.
        :param inputs: graph => tuple with tree elements
                                1. node features
                                2. edges => ([source_node], [neighbor_node])
                                3. edge_features
        :return: graph embedding
        """

        # node_representations, edges, edge_features = inputs
        node_representations = inputs[:, 0:1].merge_dims(0, 1)
        edges = tf.cast(inputs[:, 1:2].merge_dims(0, 1), tf.int64)
        edge_features = inputs[:, 2:3].merge_dims(0, 1)
        # node_indices, neighbour_indices = edges[0], edges[1]
        node_indices = edges[:, 0:1].merge_dims(0, 1)
        neighbour_indices = edges[:, 1:2].merge_dims(0, 1)
        neighbour_representations = tf.ragged.stack([tf.gather(node_representation, neighbour_index)
                                                    for node_representation, neighbour_index in
                                                    zip(node_representations, neighbour_indices)], 0)
        neighbour_messages = self.prepare_neighbour_messages(neighbour_representations, edge_features)

        number_nodes = tf.ragged.constant([len(each.to_tensor()) for each in node_representations])
        aggregated_messages = self.aggregate(node_indices, neighbour_messages, number_nodes)

        return self.update(node_representations, aggregated_messages)


class PaddingLayer(tf.keras.layers.Layer):

    def __init__(self, shape, custom_padding_value=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_padding_value = custom_padding_value
        self.shape = shape
    
    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def adding_shape(self, input_shape):
        input_shape = tf.convert_to_tensor(input_shape)
        shape = tf.convert_to_tensor(self.shape)
        diff = shape - input_shape
        condition = tf.less_equal(diff, 0)
        if tf.reduce_all(condition):
            shape = list(shape)
            shape[-2] = 0
            return tf.convert_to_tensor(shape)
        return tf.where(condition, shape, diff)

    def call(self, inputs):

        if self.custom_padding_value is None:
            min_value = tf.reduce_min(inputs)
            padding_value = min_value - 0.01
        else:
            padding_value = self.custom_padding_value

        if hasattr(inputs, '__len__'):
            input_plus_pad = tf.keras.preprocessing.sequence.pad_sequences(inputs, value=padding_value)
        else:
            input_plus_pad = tf.sparse.to_dense(inputs.to_sparse(), default_value=padding_value)

        if tf.reduce_all(self.shape == input_plus_pad.shape):
            return input_plus_pad
        return tf.concat([input_plus_pad, tf.ones(self.adding_shape(input_plus_pad.shape)) * padding_value], 1)

    def compute_output_mask(self, inputs):
        """
        compute mask based on padding value in call() method
        :param inputs: 
        :return: 
        """
        if self.custom_padding_value is None:
            min_value = tf.reduce_min(inputs)
            padding_value = min_value - 1
        else:
            padding_value = self.custom_padding_value
        output = self.call(inputs)
        return tf.not_equal(output, padding_value)


class ThresholdLayer(tf.keras.layers.Layer):

    def __init__(self, *args, **kwargs):
        super(ThresholdLayer, self).__init__(*args, **kwargs)
        self.kernel = self.add_weight(name="threshold", shape=(1,), initializer="uniform",
                                      trainable=True)

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def call(self, x):
        """
        :param x:
        :return:
        """
        return tf.keras.backend.sigmoid(100 * (x - self.kernel))

    def compute_output_shape(self, input_shape):
        return input_shape


class BinaryLayer(tf.keras.layers.Layer):
    """
    binary output with trainable threshold
    """

    def __init__(self, *args, **kwargs):
        super(BinaryLayer, self).__init__(*args, **kwargs)
        self.kernel = self.add_weight(name="thresholds", shape=(1,), initializer="uniform",
                                      trainable=True)

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def call(self, inputs, *args, **kwargs):
        return tf.keras.backend.cast(tf.keras.backend.greater(inputs, self.kernel), tf.keras.backend.floatx())


class RowChooserLayer(tf.keras.layers.Layer):
    """
    this layer compute with row must be delete with deletion. it compute new mask based of result without changing input.
    the main method is compute_output_mask method
    """

    def __init__(self, units, *args, **kwargs):
        """
        :param units: units of Dense it must be the maximum length in input
        :param args:
        :param kwargs:
        """
        super(RowChooserLayer, self).__init__(*args, **kwargs)

        self.binary_maker = BinaryLayer()
        self.dense_layer = tf.keras.layers.Dense(units, activation="relu")
    
    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def call(self, inputs, mask=None, return_binary_result=False):
        """
        compute binary result
        :param inputs:
        :param mask:
        :param return_binary_result: if set to True returns binary_result too
        :return: raw input again
        """
        raw_input, input_plus_attention = inputs[0], inputs[1]
        x = self.dense_layer(input_plus_attention)
        row_index = self.binary_maker(x)
        if return_binary_result:
            return raw_input, row_index
        return raw_input

    def compute_output_mask(self, inputs, mask):
        """
        comput output mask with raw_input mask, attention mask and result of binary output
        that shows with rows must be without effect
        :param inputs:
        :param mask:
        :return:
        """
        raw_input, input_plus_attention = inputs[0], inputs[1]
        mask_raw_input, attention_mask = mask
        _, row_index = self(inputs, mask, True)
        row_index = tf.expand_dims(row_index, -1)
        row_index = tf.repeat(row_index, repeats=raw_input.shape[-1], axis=-1)
        bool_index = tf.math.not_equal(row_index, 0.0)
        attention_mask = tf.expand_dims(attention_mask, -1)
        attention_mask = tf.repeat(attention_mask, repeats=raw_input.shape[-1], axis=-1)
        return tf.math.logical_and(tf.math.logical_and(mask_raw_input, bool_index), attention_mask)
