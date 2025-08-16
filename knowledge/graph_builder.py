"""
graph_builder.py

This module is a fundamental component for constructing knowledge graphs from raw data,
specifically designed for integration with Graph Neural Networks (GNNs). It provides
the `build_knowledge_graph` function, which takes a list of knowledge triples (head,
relation, tail) and constructs a NetworkX graph. This graph serves as the foundational
structure that can then be processed and converted into a PyTorch Geometric (PyG)
format for GNN training and inference. The module's primary role is to bridge the
gap between a simple list of facts and a structured graph representation.
"""

import networkx as nx
from typing import List, Tuple

def build_knowledge_graph(triples: List[Tuple[str, str, str]], directed: bool = True) -> nx.Graph:
    """
    Constructs a knowledge graph from a list of (head, relation, tail) triples.

    This function iterates through a list of knowledge triples and builds a
    NetworkX graph. Each unique `head` and `tail` becomes a node, and the
    `relation` is stored as an attribute of the edge connecting them. The
    function can create either a directed or an undirected graph.

    Args:
        triples (List[Tuple[str, str, str]]): A list of tuples, where each
                                               tuple contains a head, a relation,
                                               and a tail.
        directed (bool): If True, a directed graph (DiGraph) is created.
                         If False, an undirected graph (Graph) is created.
                         Defaults to True.

    Returns:
        nx.Graph: A NetworkX graph representing the knowledge. It will be
                  either a `nx.DiGraph` or a `nx.Graph` based on the `directed`
                  parameter.
    """
    # Initialize the graph based on the `directed` parameter.
    G = nx.DiGraph() if directed else nx.Graph()
    
    # Iterate through each triple to add nodes and edges to the graph.
    for head, relation, tail in triples:
        # Add the head and tail as nodes. NetworkX handles duplicates automatically.
        G.add_node(head)
        G.add_node(tail)
        # Add a directed edge from head to tail with the relation as an attribute.
        G.add_edge(head, tail, relation=relation)
        
    return G