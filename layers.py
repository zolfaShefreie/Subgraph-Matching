import tensorflow as tf
from tensorflow import keras
import numpy as np


class GraphElementEmbedLayer(tf.keras.layers.Layer):
    def __init__(self, sequential_model, old_attr_dim, new_attr_dim=0, *args, **kwargs):
        super(GraphElementEmbedLayer, self).__init__(*args, **kwargs)
        self.sequential_model = sequential_model
        # self.reshape = tf.keras.layers.Reshape((old_attr_dim, ))
        self.old_attr_dim = old_attr_dim
        self.new_attr_dim = new_attr_dim

    def call(self, inputs):
        element_sizes = [len(each) for each in inputs.to_list()]
        inputs = inputs.merge_dims(0, 1)
        inputs = tf.sparse.to_dense(inputs.to_sparse(), default_value=0)
        inputs = tf.reshape(inputs, (-1, self.old_attr_dim))
        embedded_result = self.sequential_model(inputs)
        split_result = tf.split(embedded_result, element_sizes, 0)
        return tf.ragged.constant(list([each.numpy().tolist() for each in split_result]))


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
        self.n = node_dim
        self.base_message_create = GraphElementEmbedLayer(self.create_gnn_layers(hidden_units, node_dim, dropout_rate),
                                                          node_dim)
        self.edge_transformer = GraphElementEmbedLayer(self.create_edge_transformer_layers(hidden_units=hidden_units,
                                                                                           edge_dim=edge_dim), edge_dim)
        if self.combination_type == "gru":
            self.update_fn = tf.keras.layers.GRU(units=hidden_units, activation="tanh", recurrent_activation="sigmoid",
                                                 dropout=dropout_rate, return_state=True,
                                                 recurrent_dropout=dropout_rate)
        else:
            self.update_fn = GraphElementEmbedLayer(self.create_gnn_layers(hidden_units, node_dim, dropout_rate),
                                                    node_dim + hidden_units[-1] if self.combination_type == 'concat' else hidden_units[-1])

    @staticmethod
    def create_gnn_layers(hidden_units, node_dim, dropout_rate, use_attention=False, name=None):
        """
        create gnn layers (it use for node features transformation)
        :param node_dim: 
        :param hidden_units:
        :param dropout_rate:
        :param use_attention: add attention layer or not
        :param name: name of box
        :return:
        """
        # gnn_layers = [tf.keras.layers.Reshape((-1, node_dim))]
        gnn_layers = list()
        for units in hidden_units:
            gnn_layers.append(tf.keras.layers.BatchNormalization())
            gnn_layers.append(tf.keras.layers.Dropout(dropout_rate))
            if use_attention:
                gnn_layers.append(tf.keras.layers.Attention(use_scale=True))
            gnn_layers.append(tf.keras.layers.Dense(units, activation=tf.nn.gelu))

        return keras.Sequential(gnn_layers, name=name)

    @staticmethod
    def create_edge_transformer_layers(hidden_units, edge_dim, name=None):
        """
        create layers for edge feature transformation
        :param edge_dim: 
        :param hidden_units:
        :param name: name of box
        :return:
        """
        edge_embed_layers = list()
        # edge_embed_layers.append(tf.keras.layers.Reshape((-1, edge_dim)))
        for units in hidden_units:
            edge_embed_layers.append(tf.keras.layers.Dense(units, activation=tf.nn.gelu))
        return keras.Sequential(edge_embed_layers, name=name)

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

    def aggregate(self, node_indices, neighbour_messages, number_nodes):
        """
        aggregate neighbour messages
        :param number_nodes: number of nodes for each input
        :param node_indices: source nodes indices is a 1d array with length of number of edge
                             that shows the with neighbour nodes must aggregate together.
        :param neighbour_messages: is a 2d array with shape of (number of edges, units)
        :return:
        """
        all_aggregated_message = list()
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
            all_aggregated_message.append(tf.sparse.to_dense(aggregated_message.to_sparse(), default_value=0))

        # type formatting
        all_aggregated_message = tf.ragged.constant(list([each.numpy().tolist() for each in all_aggregated_message]))
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
        # neighbour_representations = tf.gather(node_representations, neighbour_indices)
        neighbour_representations = tf.ragged.constant([tf.gather(node_representations[i], neighbour_indices[i]).to_list() 
                                                        for i in range(len(neighbour_indices.to_list()))])
        neighbour_messages = self.prepare_neighbour_messages(neighbour_representations, edge_features)

        number_nodes = tf.ragged.constant([len(each.numpy()) for each in node_representations])
        aggregated_messages = self.aggregate(node_indices, neighbour_messages, number_nodes)

        return self.update(node_representations, aggregated_messages)


class PaddingLayer(tf.keras.layers.Layer):

    def __init__(self, shape, custom_padding_value=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_padding_value = custom_padding_value
        self.shape = tf.convert_to_tensor(shape)
    
    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    def adding_shape(self, input_shape):
        input_shape = tf.convert_to_tensor(input_shape)
        diff = self.shape - input_shape
        condition = tf.less_equal(diff, 0)
        if tf.reduce_all(condition):
            shape = self.shape.numpy()
            shape[-2] = 0
            return shape
        return tf.where(condition, self.shape, diff)

    def call(self, inputs):

        if self.custom_padding_value is None:
            min_value = tf.reduce_min(inputs)
            padding_value = min_value - 1
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
        # row_index = tf.expand_dims(row_index, -1)
        # row_index = tf.repeat(row_index, repeats=raw_input.shape[-1], axis=-1)
        # bool_index = tf.math.not_equal(row_index, 0.0)

        # output = raw_input[bool_index]
        #
        # # reshape process for 1-d output
        # element_new_sizes = [len(each) for each in bool_index.numpy() if each.all()]
        # if len(element_new_sizes) == 0:
        #     return None
        # output = tf.reshape(output, (-1, raw_input.shape[-1]))
        # split_output = tf.split(output, element_new_sizes, 0)
        # return tf.ragged.constant(list([each.numpy().tolist() for each in split_output]))

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



# class SubMatrixChooser(tf.keras.layers.Layer):
#
#     def __init__(self, units, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.binary_maker = BinaryLayer()
#
#     def call(self, inputs, *args, **kwargs):
#         """
#         :param inputs:
#         :param args:
#         :param kwargs:
#         :return:
#         """
#         raw_input, input_plus_attention = inputs
#         row_index = self.binary_maker(input_plus_attention)
#         row_index = tf.expand_dims(row_index, -1)
#         row_index = tf.repeat(row_index, repeats=raw_input.shape[-1], axis=-1)
#         bool_index = tf.math.not_equal(row_index, 0.0)
#         shape = tf.shape(raw_input)
#         output = raw_input[bool_index]
#         return tf.reshape(output, self.get_new_shape(shape))