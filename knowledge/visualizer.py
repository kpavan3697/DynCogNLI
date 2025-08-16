"""
visualizer.py

This module provides utility functions for visualizing knowledge graphs and, crucially, 
the attention scores derived from a Graph Attention Network (GAT) model. The `visualize_subgraph` 
function uses the NetworkX and Matplotlib libraries to render a graphical representation of a 
subgraph, with the added capability of coloring nodes based on their attention scores. 
This is an invaluable tool for debugging, analysis, and gaining insights into how the 
DynCogNLI model weights the importance of different concepts during its reasoning process. 
It allows developers to visually inspect which parts of the knowledge graph are being 
prioritized by the model for a given query.
"""

import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, Optional

def visualize_subgraph(graph: nx.Graph, node_mapping: Dict[int, str], attention_scores: Optional[Dict[str, float]] = None, save_path: str = "subgraph.png"):
    """
    Visualizes a NetworkX graph, optionally with nodes colored according to their attention scores.

    This function generates a plot of a given graph, drawing nodes and edges. If attention
    scores are provided, it uses a color map to visually represent the importance of each node.
    This provides an intuitive way to understand the GNN's reasoning process. The resulting
    plot is saved as a PNG file.

    Args:
        graph (nx.Graph): The NetworkX graph to be visualized.
        node_mapping (Dict[int, str]): A dictionary mapping GNN node indices to their 
                                       original concept names. (Note: this is not directly
                                       used in the visualization logic but is often part of
                                       the pipeline and good to include for context.)
        attention_scores (Optional[Dict[str, float]]): A dictionary mapping node names 
                                                       (concepts) to their attention scores.
                                                       Nodes with higher scores will appear
                                                       in a more saturated color.
        save_path (str): The file path where the generated plot will be saved.
                         Defaults to "subgraph.png".
    """
    # Use a spring layout to arrange nodes in an aesthetically pleasing manner.
    pos = nx.spring_layout(graph)
    
    # Set up the plot.
    plt.figure(figsize=(10, 7))

    # Determine node colors based on attention scores.
    node_colors = []
    for node in graph.nodes():
        if attention_scores and node in attention_scores:
            # Use the provided attention score.
            node_colors.append(attention_scores[node])
        else:
            # Default color for nodes without a score.
            node_colors.append(0.5)

    # Draw the nodes, coloring them based on the `node_colors` list.
    nx.draw_networkx_nodes(
        graph, 
        pos, 
        node_color=node_colors, 
        cmap=plt.cm.Blues, 
        node_size=800
    )
    
    # Draw the edges.
    nx.draw_networkx_edges(
        graph, 
        pos, 
        arrows=True
    )
    
    # Draw the node labels (the concept names).
    nx.draw_networkx_labels(
        graph, 
        pos, 
        font_size=10
    )

    # Draw the edge labels, which represent the relationships between concepts.
    edge_labels = nx.get_edge_attributes(graph, 'relation')
    nx.draw_networkx_edge_labels(
        graph, 
        pos, 
        edge_labels=edge_labels, 
        font_color='red'
    )

    # Finalize the plot with a title and clean up the axes.
    plt.title("Concept Subgraph with Attention Weights")
    plt.axis("off")
    plt.tight_layout()
    
    # Save the figure to the specified file path.
    plt.savefig(save_path)
    print(f"Graph saved to {save_path}")