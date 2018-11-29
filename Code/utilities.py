import numpy as np

def graph_to_edge_matrix(G):
    """Convert a networkx graph into an edge matrix.
    https://www.learndatasci.com/tutorials/k-means-clustering-algorithms-python-intro/

    Parameters
    ----------
    G : networkx graph
    """
    # Initialize edge matrix with zeros
    edge_mat = np.zeros((len(G), len(G)), dtype=int)

    # Loop to set 0 or 1 (diagonal elements are set to 1)
    for node in G:
        for neighbor in G.neighbors(node):
            edge_mat[node][neighbor] = 1
        edge_mat[node][node] = 1

    return edge_mat