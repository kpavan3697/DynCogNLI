# common_sense_client.py
import networkx as nx

def get_subgraph_for_query(query: str, max_nodes: int = 50) -> nx.Graph:
    """
    Simulates retrieving a subgraph from a common sense knowledge graph
    based on the user's query. In a real system, this would involve querying
    a database like ConceptNet or a custom knowledge base.

    Args:
        query (str): The user's input query.
        max_nodes (int): Maximum number of nodes in the subgraph.

    Returns:
        nx.Graph: A NetworkX graph representing the relevant subgraph.
    """
    graph = nx.Graph()
    # Simple keyword-based subgraph generation for demonstration
    # In a real system, this would be much more sophisticated, e.g.,
    # using entity linking, semantic search, and graph traversal.

    # Example nodes and edges based on common concepts
    base_concepts = set(query.lower().split())
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

    # Add query words as nodes if not already present
    for word in base_concepts:
        if word not in graph:
            graph.add_node(word)
        # Link query words to general concepts
        if word in ["cracked", "broken", "damaged"]:
            if "repair" in graph: graph.add_edge(word, "repair")
            if "issue" in graph: graph.add_edge(word, "issue")
        if word in ["stressed", "anxious"]:
            if "emotional_distress" in graph: graph.add_edge(word, "emotional_distress")
        if word in ["bored", "lonely"]:
            if "emotional_distress" in graph: graph.add_edge(word, "emotional_distress")

    # Add some general nodes if the graph is too small
    if graph.number_of_nodes() < 5:
        graph.add_nodes_from(["person", "user", "need", "problem", "solution", "help"])
        graph.add_edges_from([("person", "user"), ("user", "need"), ("need", "problem"),
                              ("problem", "solution"), ("solution", "help")])

    # To ensure it's not empty for simple queries, add a default link
    if not graph.edges and graph.nodes:
        if len(graph.nodes) > 1:
            nodes_list = list(graph.nodes)
            graph.add_edge(nodes_list[0], nodes_list[1])
        elif len(graph.nodes) == 1:
            graph.add_node("context")
            graph.add_edge(list(graph.nodes)[0], "context")

    # Limit graph size if it exceeds max_nodes (simple truncation)
    if graph.number_of_nodes() > max_nodes:
        nodes_to_keep = list(graph.nodes)[:max_nodes]
        graph = graph.subgraph(nodes_to_keep).copy() # Ensure it's a copy

    # Ensure all nodes have some features, even if dummy
    for node in graph.nodes():
        if 'feature' not in graph.nodes[node]:
            graph.nodes[node]['feature'] = [0.0] * 30 # Placeholder for initial features

    return graph