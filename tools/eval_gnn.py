import argparse
import torch
from reasoning.gat_model import GATModel
from preprocessing.conceptnet_loader import load_conceptnet
from preprocessing.atomic_loader import load_atomic_tsv
from reasoning.graph_builder import nx_to_pyg_data, fetch_conceptnet_relations
import networkx as nx

def main(args):
    """
    Main function to run the GNN model evaluation.

    Args:
        args: Command-line arguments from argparse.
    """
    device = torch.device(args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu')
    print(f"Using device: {device}")

    # --- Model Loading ---
    # Load the trained model checkpoint from the specified path.
    # It dynamically retrieves the model's dimensions to ensure the architecture matches.
    checkpoint = torch.load(args.checkpoint, map_location=device)
    input_dim = checkpoint.get("input_dim", 128)
    hidden_dim = checkpoint.get("hidden_dim", 64)
    output_dim = checkpoint.get("output_dim", 4)

    # Instantiate the GATModel with the loaded dimensions.
    model = GATModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    # Load the saved model weights into the new model instance.
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    # Set the model to evaluation mode, which disables dropout and other training-specific layers.
    model.eval()

    # --- Knowledge Graph Construction ---
    # Load triples from both ConceptNet and ATOMIC datasets.
    concept_triples = load_conceptnet(args.conceptnet)
    atomic_triples = load_atomic_tsv(args.atomic)

    # Combine the triples and build a single NetworkX graph.
    all_triples = [(h, r, t) for h, r, t in concept_triples] + [(h, r, t) for h, r, t in atomic_triples]
    G = nx.DiGraph()
    for h, r, t in all_triples:
        G.add_node(h)
        G.add_node(t)
        G.add_edge(h, t, rel=r)

    # --- Term-based Evaluation ---
    # The script evaluates the model on a list of terms provided by the user.
    terms = args.terms
    if not terms:
        print("No terms provided. Please pass --terms term1 term2 ...")
        return

    for term in terms:
        print(f"\nEvaluating term: {term}")
        # Retrieve a relevant subgraph centered around the term from the knowledge graph.
        subgraph = fetch_conceptnet_relations(term, depth=1, max_nodes=20, max_edges=50)
        if subgraph.number_of_nodes() == 0:
            print(f"No subgraph found for {term}")
            continue

        # Convert the NetworkX subgraph into a PyTorch Geometric (PyG) data object.
        # Placeholder feature vectors are assigned since no text/context embeddings are used here.
        pyg_data, _ = nx_to_pyg_data(subgraph, feature_dim=input_dim, query_embedding=None, context_embedding=None)
        pyg_data = pyg_data.to(device)

        # Run the model without calculating gradients to get the final output scores.
        with torch.no_grad():
            output = model(pyg_data)
            # Print the final output tensor, which contains the persona dimension scores.
            print(f"Model output for term '{term}': {output.cpu().numpy()}")

if __name__ == "__main__":
    # --- Command-line Argument Parser ---
    # This block defines the arguments required to run the script from the command line.
    parser = argparse.ArgumentParser(description="Evaluate a trained GNN model on specific terms.")
    parser.add_argument("--conceptnet", type=str, required=True, help="Path to the ConceptNet CSV file.")
    parser.add_argument("--atomic", type=str, required=True, help="Path to the ATOMIC TSV file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained model checkpoint (.pth file).")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use for evaluation (e.g., 'cpu' or 'cuda').")
    parser.add_argument("--terms", nargs="+", help="List of terms to evaluate (e.g., 'car', 'laptop').")
    args = parser.parse_args()

    # Call the main function to start the evaluation process.
    main(args)