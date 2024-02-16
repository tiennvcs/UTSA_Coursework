import numpy as np
import networkx as nx


def create_graph_from_array(arr: np.ndarray):
    G = nx.from_numpy_array(arr)
    return G