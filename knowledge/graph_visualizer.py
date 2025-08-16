"""
graph_visualizer.py

This module provides essential tools for visualizing knowledge graphs, which is crucial
for debugging, analysis, and understanding the behavior of the DynCogNLI system.
It defines the `visualize_subgraph` function, which leverages the NetworkX and
Matplotlib libraries to render a graphical representation of a graph or a specific
subgraph. This allows developers to inspect the graph structure, identify relationships,
and verify that the correct common-sense knowledge is being retrieved and processed
for a given user query.
"""

import matplotlib.pyplot as plt
import networkx as nx
from typing import Optional

def visualize_subgraph(G: nx.Graph, center_node: Optional[str] = None, num_hops: int = 1, title: str = "Graph"):
    """
    Visualizes a subgraph centered around a specified node, or the entire graph if no
    center node is provided.

    The function uses a spring layout algorithm to arrange the nodes, which tends to
    place highly connected nodes closer together, making the graph structure more
    intuitive. It also includes node labels, node colors, and edge labels to
    display the relationships between concepts.

    Args:
        G (nx.Graph): The NetworkX graph to be visualized.
        center_node (Optional[str]): The node from which to start the visualization.
                                     If provided, a subgraph of `num_hops` around
                                     this node is drawn.
        num_hops (int): The number of hops (distance) from the `center_node` to
                        include in the subgraph. This is ignored if `center_node`
                        is not specified. Defaults to 1.
        title (str): The title for the visualization plot. Defaults to "Graph".
    """
    # Create a subgraph based on the `center_node` and `num_hops`.
    if center_node and center_node in G:
        nodes_in_subgraph = {center_node}
        current_nodes = {center_node}
        for _ in range(num_hops):
            neighbors = set()
            for node in current_nodes:
                # Find neighbors of the current set of nodes.
                neighbors.update(G.neighbors(node))
            current_nodes = neighbors
            nodes_in_subgraph.update(current_nodes)
        
        subgraph = G.subgraph(nodes_in_subgraph)
    else:
        # If no center node is specified, visualize a small subset of the graph.
        subgraph = G.subgraph(list(G.nodes)[:50])

    # Set up the plot for visualization.
    plt.figure(figsize=(12, 8))
    
    # Use a spring layout for an aesthetically pleasing and informative node arrangement.
    pos = nx.spring_layout(subgraph, seed=42)  # `seed` ensures consistent layouts.
    
    # Draw the nodes and edges.
    nx.draw(
        subgraph,
        pos,
        with_labels=True,
        node_size=500,
        node_color='lightblue',
        font_size=8,
        arrows=True
    )
    
    # Get and draw the edge labels (the 'relation' attribute).
    edge_labels = nx.get_edge_attributes(subgraph, 'relation')
    nx.draw_networkx_edge_labels(
        subgraph,
        pos,
        edge_labels=edge_labels,
        font_size=7
    )
    
    # Set the title and display the plot.
    plt.title(title)
    plt.show()