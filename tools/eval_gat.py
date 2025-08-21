"""
evaluate_model.py

This script provides a framework for evaluating a trained GAT model
on a set of terms. It loads a pre-trained model checkpoint, constructs a knowledge graph
from ConceptNet and ATOMIC data, and then uses the model to perform inference on subgraphs
related to a list of user-provided terms. This helps in understanding how the model
reasons about different concepts.
"""
import argparse
import torch
import os
import networkx as nx
import peewee
from typing import List, Dict, Tuple

# --- ConceptNet Lite Integration ---
import conceptnet_lite
from conceptnet_lite import Label, Relation, Language
from preprocessing.atomic_loader import load_atomic_tsv
from reasoning.gat_model import GATModel
from reasoning.graph_builder import nx_to_pyg_data

# Function to fetch ConceptNet triples from the database
def fetch_conceptnet_relations_db(term, depth=1, max_nodes=50, max_edges=200):
    """
    Fetches a subgraph from the ConceptNet database around a given term.
    This function uses conceptnet_lite to perform the search.
    """
    triples = set()
    nodes_seen = set()
    queue = [(term.lower(), 0)]
    
    while queue and len(nodes_seen) < max_nodes:
        current_term, current_depth = queue.pop(0)
        
        if current_depth > depth or current_term in nodes_seen:
            continue
        
        nodes_seen.add(current_term)
        
        try:
            label_obj = Label.get(current_term, language='en')
            relations = list(label_obj.relations)
            
            for rel in relations:
                h = rel.start.text.lower()
                t = rel.end.text.lower()
                r = rel.relation.name
                triples.add((h, r, t))
                
                if h not in nodes_seen and len(nodes_seen) < max_nodes:
                    queue.append((h, current_depth + 1))
                if t not in nodes_seen and len(nodes_seen) < max_nodes:
                    queue.append((t, current_depth + 1))
        except peewee.DoesNotExist:
            continue
        
    return list(triples)

def main(args: argparse.Namespace):
    """
    Main function to run the GAT model evaluation.
    """
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    print(f"Using device: {device}")

    # --- Model Loading ---
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        input_dim = checkpoint.get("input_dim", 128)
        hidden_dim = checkpoint.get("hidden_dim", 64)
        output_dim = checkpoint.get("output_dim", 4)
    except FileNotFoundError:
        print(f"Error: Checkpoint file not found at {args.checkpoint}")
        return
    except Exception as e:
        print(f"An error occurred while loading the checkpoint: {e}")
        return

    model = GATModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # --- Knowledge Graph Construction ---
    os.makedirs("data/conceptnet", exist_ok=True)
    db_path = "data/conceptnet/conceptnet.db"
    print(f"Connecting to ConceptNet Lite at {db_path}")
    conceptnet_lite.connect(db_path)
    
    atomic_triples = load_atomic_tsv(args.atomic)
    print(f"[ATOMIC] Loaded {len(atomic_triples)} triples from {args.atomic}")
    
    atomic_graph = nx.DiGraph()
    for h, r, t in atomic_triples:
        atomic_graph.add_node(h)
        atomic_graph.add_node(t)
        atomic_graph.add_edge(h, t, rel=r)

    # --- Term-based Evaluation ---
    terms = args.terms
    if not terms:
        print("No terms provided. Please pass --terms term1 term2 ...")
        return

    for term in terms:
        print(f"\nEvaluating term: {term}")
        
        # New: Split the term into keywords and remove stopwords
        keywords = [word for word in term.lower().split() if word not in ["i", "am", "a", "an", "the", "feeling"]]
        
        subgraph = nx.DiGraph()
        found_triples = False

        # Loop through each keyword to find relevant ConceptNet and ATOMIC triples
        for keyword in keywords:
            # Fetch ConceptNet triples for the keyword
            concept_triples = fetch_conceptnet_relations_db(keyword, depth=1, max_nodes=20, max_edges=50)
            if concept_triples:
                found_triples = True
            
            for h, r, t in concept_triples:
                subgraph.add_node(h)
                subgraph.add_node(t)
                subgraph.add_edge(h, t, rel=r)
            
            # Find and add ATOMIC triples that contain the keyword
            for h, r, t in atomic_triples:
                if keyword in h.lower() or keyword in t.lower():
                    subgraph.add_node(h)
                    subgraph.add_node(t)
                    subgraph.add_edge(h, t, rel=r)
                    found_triples = True
        
        if not found_triples or subgraph.number_of_nodes() == 0:
            print(f"No subgraph found for {term}")
            continue

        pyg_data, _ = nx_to_pyg_data(subgraph, feature_dim=input_dim, query_embedding=None, context_embedding=None)
        pyg_data = pyg_data.to(device)

        with torch.no_grad():
            output = model(pyg_data)
            print(f"Model output for term '{term}': {output.cpu().numpy()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained GAT model on specific terms.")
    parser.add_argument("--atomic", type=str, required=True, help="Path to the ATOMIC TSV file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained model checkpoint (.pth file).")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for evaluation (e.g., 'cpu' or 'cuda').")
    parser.add_argument("--terms", nargs="+", help="List of terms to evaluate (e.g., 'car', 'laptop').")
    args = parser.parse_args()

    main(args)