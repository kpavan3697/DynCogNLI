"""
common_sense_client.py

This module simulates the process of retrieving relevant subgraphs from a common sense 
knowledge base. The `get_subgraph_for_query` function acts as a placeholder for a more 
complex system that would perform entity linking, semantic search, and graph traversal on a 
real-world knowledge graph (e.g., ConceptNet or a custom database). The function generates 
a simplified NetworkX graph based on keywords in the user's query, providing a structured 
input for the downstream GNN-based reasoning model.
"""

import networkx as nx
from typing import Set

def get_subgraph_for_query(query: str, max_nodes: int = 50) -> nx.Graph:
    """
    Simulates the retrieval of a relevant subgraph from a common sense knowledge graph.

    This function generates a simple NetworkX graph where nodes represent concepts 
    and edges represent relationships. The graph's content is determined by a keyword-based
    matching system. In a production environment, this would be replaced with a robust
    knowledge graph retrieval mechanism.

    Args:
        query (str): The user's natural language input query.
        max_nodes (int): The maximum number of nodes to include in the generated subgraph.

    Returns:
        nx.Graph: A NetworkX graph containing common sense concepts related to the query.
                  Each node is guaranteed to have a 'feature' attribute.
    """
    graph = nx.Graph()
    # Extract base concepts from the query for matching.
    base_concepts: Set[str] = set(query.lower().split())

    # --- Keyword-based Subgraph Generation ---
    # This section adds predefined nodes and edges based on keywords found in the query.
    # It mimics a retrieval process from a knowledge base.
    if "laptop" in base_concepts or "computer" in base_concepts:
        graph.add_nodes_from(["laptop", "computer", "screen", "keyboard", "device", "technology", "broken", "repair", "data"])
        graph.add_edges_from([("laptop", "screen"), ("laptop", "keyboard"), ("laptop", "device"),
                              ("computer", "device"), ("computer", "laptop"), ("broken", "laptop"),
                              ("repair", "broken"), ("data", "laptop")])
    if "car" in base_concepts or "vehicle" in base_concepts:
        graph.add_nodes_from(["car", "vehicle", "engine", "wheel", "road", "travel", "broken", "repair", "accident"])
        graph.add_edges_from([("car", "engine"), ("car", "wheel"), ("car", "vehicle"),
                              ("broken", "car"), ("repair", "car"), ("car", "road"), ("travel", "car"),
                              ("accident", "car")])
    if "health" in base_concepts or "sick" in base_concepts or "doctor" in base_concepts:
        graph.add_nodes_from(["health", "sick", "doctor", "medicine", "symptoms", "treatment", "pain", "wellbeing"])
        graph.add_edges_from([("sick", "health"), ("doctor", "sick"), ("medicine", "sick"),
                              ("symptoms", "sick"), ("treatment", "sick"), ("pain", "sick"),
                              ("wellbeing", "health"), ("doctor", "treatment")])
    if "bored" in base_concepts or "activity" in base_concepts:
        graph.add_nodes_from(["bored", "activity", "entertainment", "hobby", "fun", "suggestions", "time"])
        graph.add_edges_from([("bored", "activity"), ("activity", "entertainment"),
                              ("entertainment", "fun"), ("hobby", "fun"), ("suggestions", "activity"),
                              ("time", "bored")])
    if "money" in base_concepts or "financial" in base_concepts:
        graph.add_nodes_from(["money", "financial", "budget", "debt", "investment", "income", "expense", "urgent"])
        graph.add_edges_from([("money", "financial"), ("money", "budget"), ("debt", "financial"),
                              ("investment", "money"), ("income", "money"), ("expense", "money"),
                              ("urgent", "debt")])

    # Add the exact query words as nodes if they are not already in the graph.
    for word in base_concepts:
        if word not in graph:
            graph.add_node(word)
        
        # Link query words to general concepts if they exist in the graph.
        if word in ["cracked", "broken", "damaged"]:
            if "repair" in graph: graph.add_edge(word, "repair")
            if "issue" in graph: graph.add_edge(word, "issue")
        if word in ["stressed", "anxious", "worried"]:
            if "emotional_distress" in graph: graph.add_edge(word, "emotional_distress")
        if word in ["bored", "lonely"]:
            if "emotional_distress" in graph: graph.add_edge(word, "emotional_distress")
    
    # Add a default, generic subgraph if the initial graph is too small or empty.
    if graph.number_of_nodes() < 5:
        graph.add_nodes_from(["person", "user", "need", "problem", "solution", "help"])
        graph.add_edges_from([("person", "user"), ("user", "need"), ("need", "problem"),
                              ("problem", "solution"), ("solution", "help")])

    # Ensure the graph is connected by adding a default edge if it only contains nodes.
    if not graph.edges and graph.nodes:
        nodes_list = list(graph.nodes)
        if len(nodes_list) > 1:
            graph.add_edge(nodes_list[0], nodes_list[1])
        elif len(nodes_list) == 1:
            graph.add_node("context")
            graph.add_edge(list(graph.nodes)[0], "context")

    # Limit the size of the graph to `max_nodes` to prevent excessive computation.
    if graph.number_of_nodes() > max_nodes:
        nodes_to_keep = list(graph.nodes)[:max_nodes]
        graph = graph.subgraph(nodes_to_keep).copy()

    # Ensure every node has a 'feature' attribute. This is crucial for the GNN.
    # The initial feature is a placeholder and will be replaced by real embeddings later.
    for node in graph.nodes():
        if 'feature' not in graph.nodes[node]:
            # A placeholder list of a fixed size.
            graph.nodes[node]['feature'] = [0.0] * 30

    return graph