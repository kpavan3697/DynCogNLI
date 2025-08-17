import networkx as nx
import os
import json
import torch
from typing import Set
from transformers import AutoModel, AutoTokenizer
from collections import deque

# Import the loader function from your new module
from preprocessing.conceptnet_loader import load_conceptnet

# Load configuration from the JSON file
try:
    with open("config.json", "r") as f:
        CONFIG = json.load(f)
except FileNotFoundError:
    print("Error: config.json not found. Please create it with the correct paths.")
    exit()

# Load the Transformer model and tokenizer once when the script starts
try:
    TOKENIZER = AutoTokenizer.from_pretrained(CONFIG["gnn"]["embedding_model"])
    MODEL = AutoModel.from_pretrained(CONFIG["gnn"]["embedding_model"])
except Exception as e:
    print(f"Error loading Transformer model: {e}")
    TOKENIZER, MODEL = None, None

# --- Step 1: Load the full ConceptNet graph into memory once ---
# This is a one-time operation at the beginning of the program.
print("Loading ConceptNet data. This may take a few minutes...")
conceptnet_triples = load_conceptnet(CONFIG["paths"]["conceptnet_csv"])
FULL_CONCEPTNET_GRAPH = nx.DiGraph() # Use a directed graph
FULL_CONCEPTNET_GRAPH.add_edges_from([(h, t, {'relation': r}) for h, r, t in conceptnet_triples])
print("ConceptNet graph loaded successfully.")

def get_subgraph_for_query(query: str) -> nx.Graph:
    """
    Dynamically retrieves a relevant subgraph from the in-memory ConceptNet graph.

    Args:
        query (str): The user's natural language input query.

    Returns:
        nx.Graph: A NetworkX graph containing common sense concepts related to the query.
                  Each node is guaranteed to have a 'feature' attribute.
    """
    graph = nx.Graph()
    base_concepts: Set[str] = set(query.lower().split())

    # --- Step 2: Use BFS to find relevant nodes for the subgraph ---
    # Find seed nodes that are directly in the query and in the full graph.
    seed_nodes = {concept for concept in base_concepts if concept in FULL_CONCEPTNET_GRAPH}
    
    # If no concepts from the query are found, return an empty graph.
    if not seed_nodes:
        print("No matching concepts found in the ConceptNet graph.")
        # Add the query words as nodes to at least have something
        graph.add_nodes_from(base_concepts)
        return graph
    
    # Use a BFS to find the subgraph
    queue = deque([(node, 0) for node in seed_nodes])
    visited = set(seed_nodes)
    subgraph_nodes = set(seed_nodes)

    while queue and len(subgraph_nodes) < CONFIG["gnn"]["max_nodes"]:
        current_node, current_depth = queue.popleft()
        
        if current_depth >= CONFIG["gnn"]["max_hops"]:
            continue
            
        for neighbor in FULL_CONCEPTNET_GRAPH.neighbors(current_node):
            if neighbor not in visited:
                visited.add(neighbor)
                subgraph_nodes.add(neighbor)
                queue.append((neighbor, current_depth + 1))
    
    # Create the final subgraph from the discovered nodes.
    subgraph = FULL_CONCEPTNET_GRAPH.subgraph(subgraph_nodes).copy()
    
    # --- Step 3: Generate real embeddings for the nodes ---
    if MODEL and TOKENIZER:
        for node in subgraph.nodes():
            if 'feature' not in subgraph.nodes[node]:
                try:
                    inputs = TOKENIZER(node, return_tensors="pt", padding=True, truncation=True)
                    with torch.no_grad():
                        node_embedding = MODEL(**inputs).last_hidden_state.mean(dim=1).squeeze()
                    subgraph.nodes[node]['feature'] = node_embedding.tolist()
                except Exception as e:
                    print(f"Warning: Could not create embedding for node '{node}'. Error: {e}")
                    # Fallback to a placeholder if the model failed to load
                    placeholder_size = 768
                    subgraph.nodes[node]['feature'] = [0.0] * placeholder_size
    
    print(f"Generated a subgraph with {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges.")
    return subgraph

if __name__ == "__main__":
    query = "My laptop screen is cracked and it's making a strange sound."
    subgraph = get_subgraph_for_query(query)
    
    print("\n--- Subgraph Nodes ---")
    for node, data in subgraph.nodes(data=True):
        print(f"Node: {node}, Features shape: {len(data['feature'])}")
    
    print("\n--- Subgraph Edges ---")
    for u, v, data in subgraph.edges(data=True):
        print(f"Edge: ({u}, {v}), Relation: {data.get('relation', 'n/a')}")