"""
graph_plotter.py

This module provides utility functions for visualizing graph structures, particularly
the attention weights learned by a GNN model. It uses the NetworkX and Matplotlib
libraries to create visual representations of the graph, helping to inspect
the GNN's reasoning process and connectivity patterns. This is an essential tool
for debugging and interpreting the model's behavior.
"""
import networkx as nx
import matplotlib.pyplot as plt
import torch
from typing import Dict, Any, List

def visualize_attention(
    edge_index: torch.Tensor,
    attention_weights: torch.Tensor,
    node_labels: Dict[int, str]
):
    """
    Visualizes a graph with edges colored according to their attention weights.

    This function takes the raw output from a GATConv layer (edge index and attention
    weights) and a mapping of node indices to their labels. It builds a NetworkX
    graph and then plots it, using a color map to represent the attention scores
    on each edge. This makes it easy to see which relationships the model is
    focusing on.

    Args:
        edge_index (torch.Tensor): A tensor of shape (2, num_edges) containing the
                                   source and destination indices for each edge.
        attention_weights (torch.Tensor): A tensor of shape (num_edges,) containing
                                         the attention weight for each corresponding edge.
        node_labels (Dict[int, str]): A dictionary mapping node indices (int) to their
                                      original string labels.
    """
    # Create a directed graph from the edge index
    G = nx.DiGraph()
    
    # Iterate through edges and add them to the graph with their attention weights
    for i in range(edge_index.shape[1]):
        src = int(edge_index[0][i])
        dst = int(edge_index[1][i])
        weight = float(attention_weights[i])
        G.add_edge(src, dst, weight=weight)

    # Use a spring layout for node positioning
    pos = nx.spring_layout(G, seed=42)
    
    # Extract edge weights and normalize them for color mapping
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    # Normalize weights to a 0-1 range for the color map
    norm_weights = [(w - min(weights)) / (max(weights) - min(weights) + 1e-6) for w in weights]

    # Draw the nodes and edges
    # The `edge_cmap` and `edge_color` arguments color the edges based on the normalized weights
    nx.draw(G, pos, labels=node_labels, edge_color=norm_weights, edge_cmap=plt.cm.Blues,
            with_labels=True, node_size=600, font_size=10, font_color="black")
            
    # Add labels to the edges to show the exact attention weight
    edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    plt.title("GAT Attention Weights")
    plt.show()
