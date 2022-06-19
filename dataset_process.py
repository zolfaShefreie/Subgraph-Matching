import os
import requests
import ast
import networkx as nx
import gzip
import numpy as np
import tensorflow as tf


def download_file(url: str, file_path: str):
    """
        download file and save on file path
    """
    file_content = requests.get(url).content
    file = open(file_path, 'wb')
    file.write(file_content)
    file.close()


class Dataset:

    PATH = "./datasets"
    DATASET_LINKS = {'AIDS': {"link": "https://github.com/zolfaShefreie/Subgraph-Matching-Dataset-Generation/raw/main/subgraph_matching_dataset/AIDS/AIDS.txt.gz",
                              "info_link": "https://raw.githubusercontent.com/zolfaShefreie/Subgraph-Matching-Dataset-Generation/main/subgraph_matching_dataset/AIDS/AIDS_info",
                              "dataset_dir": f"{PATH}/AIDS"},
                     "BZR": {"link": "https://github.com/zolfaShefreie/Subgraph-Matching-Dataset-Generation/raw/main/subgraph_matching_dataset/BZR/BZR.txt.gz",
                             "info_link": "https://raw.githubusercontent.com/zolfaShefreie/Subgraph-Matching-Dataset-Generation/main/subgraph_matching_dataset/BZR/BZR_info",
                             "dataset_dir": f"{PATH}/BZR"},
                     "Cuneiform": {"link": "https://github.com/zolfaShefreie/Subgraph-Matching-Dataset-Generation/raw/main/subgraph_matching_dataset/Cuneiform/Cuneiform.txt.gz",
                                   "info_link": "https://raw.githubusercontent.com/zolfaShefreie/Subgraph-Matching-Dataset-Generation/main/subgraph_matching_dataset/Cuneiform/Cuneiform_info",
                                   "dataset_dir": f"{PATH}/Cuneiform"},
                     "IMDB-MULTI": {"link": "https://github.com/zolfaShefreie/Subgraph-Matching-Dataset-Generation/raw/main/subgraph_matching_dataset/IMDB-MULTI/IMDB-MULTI.txt.gz",
                                    "info_link": "https://raw.githubusercontent.com/zolfaShefreie/Subgraph-Matching-Dataset-Generation/main/subgraph_matching_dataset/IMDB-MULTI/IMDB-MULTI_info",
                                    "dataset_dir": f"{PATH}/IMDB-MULTI"}}

    def __init__(self, dataset_name: str):
        """
        :param dataset_name:
        """
        if dataset_name in self.DATASET_LINKS:
            self.dataset_name = dataset_name
            if not os.path.exists(self.PATH):
                os.mkdir(self.PATH)
            self.info = self._get_dataset_info(dataset_name)
        else:
            raise Exception("Dataset Not Found")

    @property
    def max_graph_nodes(self):
        return self.info['max_nodes']

    @property
    def node_attr_dim(self):
        return self.info['node_attr_dim']

    @property
    def edge_attr_dim(self):
        return self.info['edge_attr_dim']

    @classmethod
    def _get_dataset_info(cls, dataset_name: str) -> dict:
        """
        manage to check dataset exists or not and return dict info of dataset
        :param dataset_name: name of dataset
        :return:
        """
        info_path = cls.DATASET_LINKS[dataset_name]['dataset_dir'] + "/" + dataset_name + "_info"
        dataset_path = cls.DATASET_LINKS[dataset_name]['dataset_dir'] + "/" + dataset_name + ".txt.gz"
        if not os.path.exists(info_path) or not os.path.exists(dataset_path):
            cls._download_dataset(dataset_name)
        file = open(info_path, 'r')
        return ast.literal_eval(file.read())

    @classmethod
    def _download_dataset(cls, dataset_name: str):
        """
        download dataset
        :param dataset_name: name of dataset
        :return:
        """
        if not os.path.exists(cls.DATASET_LINKS[dataset_name]['dataset_dir']):
            os.mkdir(cls.DATASET_LINKS[dataset_name]['dataset_dir'])
        download_file(cls.DATASET_LINKS[dataset_name]['link'],
                      cls.DATASET_LINKS[dataset_name]['dataset_dir'] + "/" + dataset_name + ".txt.gz")
        download_file(cls.DATASET_LINKS[dataset_name]['info_link'],
                      cls.DATASET_LINKS[dataset_name]['dataset_dir'] + "/" + dataset_name + "_info")

    @classmethod
    def _reindex_graph_nodes(cls, graph_data: dict):
        """
        reindex node_ids and change all data based on
        :param graph_data:
        :return:
        """
        node_index = {key: index for index, key in enumerate(list(graph_data['nodes'].keys()))}
        edges = dict()
        nodes = dict()
        for node_id, value in graph_data['nodes'].items():
            nodes.update({node_index[node_id]: value})
        for edge_id, value in graph_data['edges'].items():
            edges.update({(node_index[edge_id[0]], node_index[edge_id[1]]): value})
        return {'nodes': nodes, "edges": edges}

    @classmethod
    def _add_degree_node_feature(cls, graph_data: dict) -> dict:
        """
        make node feature with node degree
        :param graph_data:
        :return:
        """
        graph = nx.DiGraph()
        graph.add_nodes_from([(key, value) for key, value in graph_data['nodes'].items()])
        graph.add_edges_from([(key[0], key[1], value) for key, value in graph_data['edges'].items()])
        node_attr = {key: {'degree': graph.degree} for key, _ in graph_data['nodes'].items()}
        nx.set_node_attributes(graph, node_attr)
        return {
            'nodes': {item[0]: item[1] for item in list(graph.nodes(data=True))},
            "edges": {(item[0], item[1]): item[2] for item in list(graph.edges(data=True))}
        }

    @classmethod
    def _graph_dict_format(cls, graph_dict: dict) -> (list, list, list):
        """
        format the graph data
        convert to (nodes, edges, edges_attr):
            nodes: list of node attributes
            edges: list of edges (node_id, node_id)
            edges_attr: list of edge attributes
        :param graph_dict:
        :return:
        """

        nodes = [None for i in range(len(graph_dict['nodes']))]
        for node_id, node_attr in graph_dict['nodes'].items():
            nodes[node_id] = list(node_attr.values())

        edges, edges_attr = list(), list()
        for edge, attr in graph_dict['edges'].items():
            edges.append(np.array(edge))
            edges_attr.append(list(attr.values()))

        # graph_data = np.empty(3, dtype=object)
        # graph_data[0] = np.array(nodes)
        # graph_data[1] = np.array(edges).T
        # graph_data[2] = np.array(edges_attr)
        return [nodes, np.array(edges).T.tolist(), edges_attr]

    def load_dataset(self) -> (list, list):
        """
        load dataset from file and preprocess on elements
        :return:
        """
        file = gzip.open(self.DATASET_LINKS[self.dataset_name]['dataset_dir'] + "/" + self.dataset_name + ".txt.gz",
                         'rb')
        file_content = file.read().decode("utf-8")
        file_content = file_content.split('\n')
        dataset_y, dataset_x = list(), list()
        for element in file_content:
            if "{" in element:
                element_dict = ast.literal_eval(element)
                src_graph = self._reindex_graph_nodes(element_dict['source_graph'])
                query_graph = self._reindex_graph_nodes(element_dict['query_graph'])
                if self.info['node_attr_dim'] == 0:
                    src_graph = self._add_degree_node_feature(src_graph)
                    query_graph = self._add_degree_node_feature(query_graph)
                dataset_y.append(element_dict['label'])

                dataset_x.append([self._graph_dict_format(src_graph), self._graph_dict_format(query_graph)])

        return tf.ragged.constant(dataset_x), tf.constant(dataset_y)


if __name__ == "__main__":
    x, y = Dataset('AIDS').load_dataset()
    # print(y.shape, x.shape, x[:2, 0:1].merge_dims(0, 1).shape)


