"""
interactive_visualizer.py

This module provides a function for creating dynamic, interactive visualizations of knowledge
graphs using Plotly. The `visualize_interactive_graph` function takes a NetworkX graph as input
and generates a rich, web-based plot that allows users to zoom, pan, and hover over nodes
and edges to inspect their properties. This is particularly valuable for exploring complex
graph structures and understanding the relationships between concepts within the DynCogNLI
pipeline, offering a more engaging and informative alternative to static images.
"""

import plotly.graph_objects as go
import networkx as nx
from typing import Optional

def visualize_interactive_graph(G: nx.Graph, center_node: Optional[str] = None, num_hops: int = 1, title: str = "Interactive Graph"):
    """
    Creates and displays an interactive visualization of a graph using Plotly.

    The function first generates a subgraph centered around a specified node (if provided),
    then uses NetworkX to compute a spatial layout. This layout is then translated into
    Plotly `Scatter` traces for edges and nodes, which are then combined into a `Figure`
    object and displayed. The resulting visualization is interactive, allowing for
    dynamic exploration.

    Args:
        G (nx.Graph): The NetworkX graph to be visualized.
        center_node (Optional[str]): The node from which to begin the subgraph traversal.
                                     If None, a small subset of the full graph is used.
        num_hops (int): The number of hops to include in the subgraph from the center node.
                        Defaults to 1.
        title (str): The title to be displayed on the interactive plot.
    """
    # Create a subgraph based on the center node and number of hops.
    if center_node and center_node in G:
        nodes_in_subgraph = {center_node}
        current_nodes = {center_node}
        for _ in range(num_hops):
            neighbors = set()
            for node in current_nodes:
                neighbors.update(G.neighbors(node))
            current_nodes = neighbors
            nodes_in_subgraph.update(current_nodes)
        subgraph = G.subgraph(nodes_in_subgraph)
    else:
        # If no center node is given, use a small subset of the graph for a default view.
        subgraph = G.subgraph(list(G.nodes)[:50])

    # Compute the node positions using a force-directed layout algorithm.
    pos = nx.spring_layout(subgraph, seed=42)

    # Prepare data for Plotly edge traces.
    edge_x = []
    edge_y = []
    for edge in subgraph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])  # None is used to separate segments
        edge_y.extend([y0, y1, None])

    # Create the Plotly scatter trace for the edges.
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='none',  # Edges are not interactive on hover
        mode='lines',
        name='Edges'
    )

    # Prepare data for Plotly node traces.
    node_x = []
    node_y = []
    node_text = []
    for node in subgraph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(str(node))

    # Create the Plotly scatter trace for the nodes.
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        text=node_text,
        mode='markers+text',
        hoverinfo='text',  # Displays node name on hover
        marker=dict(
            showscale=False,
            color='lightblue',
            size=20,
            line=dict(width=2)
        ),
        name='Nodes'
    )

    # Combine the traces and layout into a Plotly Figure.
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=title,
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False)
        )
    )

    # Display the interactive plot in the default browser or notebook.
    fig.show()