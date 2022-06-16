import os
import requests
import ast
import networkx as nx


def download_file(url: str, file_path: str):
    """
        download file and save on file path
    """
    file_content = requests.get(url).text
    file = open(file_path, 'w')
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
            self.info = self._get_dataset_info(dataset_name)
        else:
            raise Exception("Dataset Not Found")

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

    def load_dataset(self):
        pass

    def load_generator(self):
        pass
