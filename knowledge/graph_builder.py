#knowledge/graph_builder.py
"""
graph_builder.py

Builds knowledge graphs from ConceptNet and converts them to PyTorch Geometric format.
Handles graph construction and feature preparation for GNN input.
"""
import networkx as nx

def build_knowledge_graph(triples, directed=True):
    G = nx.DiGraph() if directed else nx.Graph()
    for head, relation, tail in triples:
        G.add_node(head)
        G.add_node(tail)
        G.add_edge(head, tail, relation=relation)
    return G