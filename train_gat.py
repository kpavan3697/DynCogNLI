"""
train_gat.py

This script is the main training pipeline for the Graph Attention Network (GAT)
model. It demonstrates how to leverage external knowledge graphs like ConceptNet
and ATOMIC to perform multi-hop reasoning. The script loads these knowledge
graphs, builds a combined graph, generates mock data with subgraphs centered
around a query, and then trains a GAT model.
"""
import os
import sys
import json
import random
import time
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
import networkx as nx


# local imports (ensure repo root on sys.path)
# This handles the case where the script is run from a sub-directory
# For the purpose of this example, we assume `reasoning` and `context`
# are at the same level as `train_gat.py`'s parent directory.
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.join(__file__, '..'))))

from reasoning.gat_model import GATModel
from context.transformer_encoder import TransformerEncoder
from context.context_encoder import ContextEncoder

# loaders
from preprocessing.conceptnet_loader import load_conceptnet
from preprocessing.atomic_loader import load_atomic_tsv

# small helper: fetch_conceptnet_relations fallback if needed
try:
    from reasoning.graph_builder import nx_to_pyg_data, fetch_conceptnet_relations
except Exception:
    nx_to_pyg_data = None
    fetch_conceptnet_relations = None

# ---------- default config ----------
DEFAULT_MAX_CONCEPTNET = 100000  # safe default for dev; set higher for full data
DEFAULT_MAX_ATOMIC = 50000
DEFAULT_EPOCHS = 3
DEFAULT_BATCH = 8
DEFAULT_LR = 1e-3
HIDDEN_DIM = 64
OUTPUT_DIM = 4
BASE_NODE_FEATURE_DIM = 16  # small base features
DEFAULT_TRANSFORMER_EMBED_DIM = 384

# ---------- utility functions ----------
def build_merged_kg(concept_triples, atomic_triples):
    """
    Combines ConceptNet and ATOMIC triples into a single NetworkX DiGraph.
    
    Args:
        concept_triples (list): A list of (head, relation, tail) tuples from ConceptNet.
        atomic_triples (list): A list of (head, relation, tail) tuples from ATOMIC.
    
    Returns:
        tuple: A tuple containing the merged NetworkX graph, a node-to-ID mapping,
               and a relation-to-ID mapping.
    """
    all_triples = []
    all_triples.extend([(h, r, t, 'conceptnet') for (h, r, t) in concept_triples])
    all_triples.extend([(h, r, t, 'atomic') for (h, r, t) in atomic_triples])

    nodes = set()
    rels = set()
    for h, r, t, src in all_triples:
        nodes.add(h)
        nodes.add(t)
        rels.add(r)

    node2id = {n: i for i, n in enumerate(sorted(nodes))}
    rel2id = {r: i for i, r in enumerate(sorted(rels))}

    G = nx.DiGraph()
    for h, r, t, src in all_triples:
        G.add_node(h)
        G.add_node(t)
        G.add_edge(h, t, rel=r, rel_id=rel2id[r], source=src)
    return G, node2id, rel2id

def fetch_combined_relations(merged_graph, term, depth=1, max_nodes=50, max_edges=200):
    """
    Fetches a subgraph from the merged knowledge graph around a given term.
    
    This function performs a breadth-first search to find relevant nodes and edges
    up to a specified depth.
    
    Args:
        merged_graph (nx.DiGraph): The combined ConceptNet and ATOMIC graph.
        term (str): The starting node (e.g., a word from the user's query).
        depth (int): The number of hops to search from the starting term.
        max_nodes (int): The maximum number of nodes to include in the subgraph.
        max_edges (int): The maximum number of edges to include.
    
    Returns:
        nx.DiGraph: The extracted subgraph.
    """
    term = term.lower()
    if merged_graph is None or term not in merged_graph:
        return nx.DiGraph()
    nodes_seen = set([term])
    queue = [term]
    for d in range(depth):
        next_queue = []
        for n in queue:
            for nbr in merged_graph.successors(n):
                if nbr not in nodes_seen:
                    nodes_seen.add(nbr)
                    next_queue.append(nbr)
                    if len(nodes_seen) >= max_nodes:
                        break
            for nbr in merged_graph.predecessors(n):
                if nbr not in nodes_seen:
                    nodes_seen.add(nbr)
                    next_queue.append(nbr)
                    if len(nodes_seen) >= max_nodes:
                        break
            if len(nodes_seen) >= max_nodes:
                break
        queue = next_queue
        if len(nodes_seen) >= max_nodes:
            break
    sub = merged_graph.subgraph(nodes_seen).copy()
    if sub.number_of_edges() > max_edges:
        # Prune nodes with the lowest degrees to stay within max_edges limit
        degs = sorted(sub.degree, key=lambda x: x[1], reverse=True)
        keep_nodes = set([n for n, _ in degs[:max_nodes]])
        sub = sub.subgraph(keep_nodes).copy()
    return sub

def convert_nx_to_pyg_or_fallback(nx_graph, query_embedding, context_embedding, feature_dim):
    """
    Converts a NetworkX graph to a PyTorch Geometric data object.
    
    This function uses a helper function from `reasoning.graph_builder` but includes
    a fallback mechanism in case that import fails, ensuring the script can still run.
    
    Args:
        nx_graph (nx.DiGraph): The input NetworkX graph.
        query_embedding (torch.Tensor): The encoded query vector.
        context_embedding (torch.Tensor): The encoded context vector.
        feature_dim (int): The target dimension for the node features.
    
    Returns:
        tuple: A tuple with the PyTorch Geometric data object and a node mapping.
    """
    if nx_to_pyg_data is not None:
        try:
            pyg, mapping = nx_to_pyg_data(nx_graph, feature_dim=feature_dim,
                                            query_embedding=query_embedding,
                                            context_embedding=context_embedding)
            return pyg, mapping
        except Exception as e:
            print(f"[train] nx_to_pyg_data failed: {e}. Falling back to helper converter.")
    # fallback: broadcast query+context into node features
    try:
        from torch_geometric.data import Data
        import torch
        nodes = list(nx_graph.nodes())
        n = len(nodes)
        if n == 0:
            return None, None
        q_vec = query_embedding if query_embedding is not None else torch.zeros(DEFAULT_TRANSFORMER_EMBED_DIM)
        c_vec = context_embedding if context_embedding is not None else torch.zeros(context_encoder.total_context_dim)
        q_vec = q_vec.cpu() if hasattr(q_vec, 'cpu') else q_vec
        c_vec = c_vec.cpu() if hasattr(c_vec, 'cpu') else c_vec
        
        # Create a feature vector for each node by combining query and context embeddings
        node_feat = torch.zeros((n, feature_dim), dtype=torch.float)
        # Note: This is a simple fallback. The actual `nx_to_pyg_data` might be more sophisticated.
        for i in range(n):
            node_feat[i, :q_vec.shape[0]] = q_vec
            node_feat[i, q_vec.shape[0]:q_vec.shape[0]+c_vec.shape[0]] = c_vec
        
        # Build the edge index
        edge_index = [[], []]
        for u, v in nx_graph.edges():
            edge_index[0].append(nodes.index(u))
            edge_index[1].append(nodes.index(v))
        if len(edge_index[0]) > 0:
            edge_index = torch.tensor(edge_index, dtype=torch.long)
        else:
            edge_index = torch.empty((2,0), dtype=torch.long)
        data = Data(x=node_feat, edge_index=edge_index)
        mapping = {n: i for i, n in enumerate(nodes)}
        return data, mapping
    except Exception as e:
        print(f"[train] fallback nx->pyg failed: {e}")
        return None, None

# ---------- generate mock dataset ----------
mock_queries = [
    "My laptop screen cracked.",
    "I lost my pet cat.",
    "My car broke down on the highway.",
    "I successfully finished my project!",
    "My internet is not working.",
    "I received a surprise gift.",
    "My flight was delayed due to bad weather.",
    "I'm feeling a bit under the weather.",
    "Just got a promotion at work!",
    "Stuck in a massive traffic jam.",
    "Celebrating my birthday today!",
    "My pipes burst and flooded the basement."
]

persona_labels = {
    "My laptop screen cracked.": (0.8, 0.6, 0.9, 0.5), # urgency, empathy, proactiveness, emotional
    "I lost my pet cat.": (0.9, 0.9, 0.2, 0.9),
    "My car broke down on the highway.": (0.9, 0.7, 0.95, 0.4),
    "I successfully finished my project!": (0.1, 0.1, 0.1, 0.2),
    "My internet is not working.": (0.7, 0.4, 0.8, 0.3),
    "I received a surprise gift.": (0.05, 0.1, 0.05, 0.1),
    "My flight was delayed due to bad weather.": (0.75, 0.65, 0.7, 0.6),
    "I'm feeling a bit under the weather.": (0.3, 0.5, 0.2, 0.8),
    "Just got a promotion at work!": (0.05, 0.05, 0.05, 0.1),
    "Stuck in a massive traffic jam.": (0.6, 0.7, 0.3, 0.5),
    "Celebrating my birthday today!": (0.01, 0.05, 0.01, 0.1),
    "My pipes burst and flooded the basement.": (0.95, 0.8, 0.99, 0.7)
}

def adjust_label_for_context(label_tuple, mood, time_of_day, weather):
    """
    Adjusts a persona label based on contextual cues.
    
    Args:
        label_tuple (tuple): The base persona label (urgency, empathy, proactiveness, emotional).
        mood (str): The user's mood.
        time_of_day (str): The time of day.
        weather (str): The weather condition.
    
    Returns:
        tuple: The contextually adjusted persona label.
    """
    u, e, p, emp = label_tuple
    # small heuristics
    if mood.lower() in ["stressed", "anxious", "frustrated"]:
        emp = min(1.0, emp + 0.2); e = min(1.0, e + 0.1)
    if mood.lower() in ["happy", "excited"]:
        e = max(0.0, e - 0.2); emp = max(0.0, emp - 0.1)
    if time_of_day.lower() == "night":
        if u > 0.5: u = min(1.0, u + 0.1)
        emp = min(1.0, emp + 0.05)
    if weather.lower() in ["stormy", "rainy"]:
        if p > 0.5: p = min(1.0, p + 0.1)
        e = min(1.0, e + 0.05)
    return (u, e, p, emp)

def generate_mock_data(merged_graph, transformer_encoder, context_encoder, num_samples=100, max_nodes=20):
    """
    Generates a mock dataset of PyTorch Geometric graphs for training.
    
    For each sample, it picks a query, fetches a subgraph from the merged KG,
    encodes the query and context, and creates a PyG data object with labels.
    
    Args:
        merged_graph (nx.DiGraph): The combined ConceptNet and ATOMIC graph.
        transformer_encoder (TransformerEncoder): Encoder for the query text.
        context_encoder (ContextEncoder): Encoder for the context.
        num_samples (int): The number of samples to generate.
        max_nodes (int): The maximum number of nodes in each subgraph.
    
    Returns:
        list: A list of PyTorch Geometric Data objects.
    """
    dataset = []
    skipped = 0
    for _ in tqdm(range(num_samples), desc="Generating mock graphs"):
        query_text = random.choice(mock_queries)
        mood = random.choice(list(context_encoder.mood_map.keys()))
        time_of_day = random.choice(list(context_encoder.time_of_day_map.keys()))
        weather = random.choice(list(context_encoder.weather_condition_map.keys()))
        seed = next((w for w in query_text.split() if len(w) > 2), query_text).lower()
        
        # Build the subgraph for the query
        if merged_graph is not None:
            nx_sub = fetch_combined_relations(merged_graph, seed, depth=1, max_nodes=max_nodes, max_edges=200)
        elif fetch_conceptnet_relations is not None:
            nx_sub = fetch_conceptnet_relations(seed, depth=1, max_nodes=max_nodes, max_edges=200)
        else:
            nx_sub = nx.DiGraph()
            
        if not nx_sub or nx_sub.number_of_nodes() == 0:
            skipped += 1
            continue
            
        # Encode query and context
        q_emb = transformer_encoder.encode(query_text)
        c_emb = context_encoder.encode(mood, time_of_day, weather)
        
        # Convert to PyTorch Geometric format
        pyg, mapping = convert_nx_to_pyg_or_fallback(nx_sub, q_emb, c_emb, TOTAL_INPUT_DIM)
        if pyg is None or getattr(pyg, 'x', None) is None or pyg.x.numel() == 0:
            skipped += 1
            continue
            
        # Get and adjust labels
        lbl = adjust_label_for_context(persona_labels[query_text], mood, time_of_day, weather)
        pyg.y = torch.tensor([lbl[0], lbl[1], lbl[2], lbl[3]], dtype=torch.float)
        dataset.append(pyg)
        
    print(f"[train] Generated {len(dataset)} samples. Skipped {skipped}.")
    return dataset

# ---------- main ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GAT model for persona inference.")
    parser.add_argument("--conceptnet", type=str, default="data/conceptnet/conceptnet-assertions-5.7.0.csv.gz", help="Path to the ConceptNet file.")
    parser.add_argument("--atomic", type=str, default="data/atomic2020/v4_atomic_trn.csv", help="Path to the ATOMIC file.")
    parser.add_argument("--max-conceptnet", type=int, default=DEFAULT_MAX_CONCEPTNET, help="Max triples from ConceptNet.")
    parser.add_argument("--max-atomic", type=int, default=DEFAULT_MAX_ATOMIC, help="Max triples from ATOMIC.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH, help="Batch size for training.")
    parser.add_argument("--debug", action="store_true", help="Run a short, small run for debug.")
    parser.add_argument("--save_path", type=str, default="models/persona_gat_model.pth", help="Path to save the trained model checkpoint.")
    args = parser.parse_args()

    if args.debug:
        print("[train] DEBUG mode: using tiny sizes")
        args.epochs = max(1, min(3, args.epochs))
        args.max_conceptnet = min(5000, args.max_conceptnet)
        args.max_atomic = min(2000, args.max_atomic)

    # load triples
    concept_triples = []
    atomic_triples = []
    if os.path.exists(args.conceptnet):
        concept_triples = load_conceptnet(args.conceptnet, max_triples=args.max_conceptnet)
    else:
        print(f"[train] ConceptNet file not found at {args.conceptnet}. Will try live API fallback.")

    if os.path.exists(args.atomic):
        atomic_triples = load_atomic_tsv(args.atomic, max_triples=args.max_atomic)
    else:
        print(f"[train] ATOMIC file not found at {args.atomic}. Continuing without ATOMIC.")

    merged_graph = None
    node2id = {}
    rel2id = {}
    if concept_triples or atomic_triples:
        merged_graph, node2id, rel2id = build_merged_kg(concept_triples, atomic_triples)
        os.makedirs("data/kg_mappings", exist_ok=True)
        with open("data/kg_mappings/node2id.json", "w", encoding="utf-8") as fh:
            json.dump(node2id, fh)
        with open("data/kg_mappings/rel2id.json", "w", encoding="utf-8") as fh:
            json.dump(rel2id, fh)
        print("[train] Saved node/rel mappings to data/kg_mappings/")

    # encoders
    transformer_encoder = TransformerEncoder()  # will fallback to deterministic embedding if model not available
    context_encoder = ContextEncoder()

    TRANSFORMER_EMBEDDING_DIM = transformer_encoder.embedding_dim if hasattr(transformer_encoder, 'embedding_dim') and transformer_encoder.embedding_dim > 0 else DEFAULT_TRANSFORMER_EMBED_DIM
    CONTEXT_DIM = context_encoder.total_context_dim
    TOTAL_INPUT_DIM = TRANSFORMER_EMBEDDING_DIM + CONTEXT_DIM + BASE_NODE_FEATURE_DIM
    print(f"[train] TOTAL_INPUT_DIM={TOTAL_INPUT_DIM}")

    # model init
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GATModel(input_dim=TOTAL_INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=DEFAULT_LR)
    criterion = nn.MSELoss()

    best_val = float('inf')
    for epoch in range(1, args.epochs + 1):
        print(f"[train] Epoch {epoch}/{args.epochs}")
        dataset = generate_mock_data(merged_graph, transformer_encoder, context_encoder, num_samples=50 if args.debug else 200, max_nodes=20)
        if not dataset:
            print("[train] No data generated; exit.")
            break
        # small split
        random.shuffle(dataset)
        val_split = max(1, int(0.1 * len(dataset)))
        val_set = dataset[:val_split]
        train_set = dataset[val_split:]
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc="batches"):
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            pred = torch.sigmoid(out)
            # out shape: (batch_size, OUTPUT_DIM)
            bs = pred.shape[0]
            try:
                target = batch.y.view(bs, OUTPUT_DIM).to(device)
            except Exception as e:
                print(f"[train] shape error: {e}")
                continue
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_train_loss = epoch_loss / max(1, len(train_loader))
        
        # validation
        model.eval()
        val_loss = 0.0
        if val_set:
            val_loader = DataLoader(val_set, batch_size=max(1, args.batch_size), shuffle=False)
            for batch in val_loader:
                batch = batch.to(device)
                with torch.no_grad():
                    out = model(batch)
                    pred = torch.sigmoid(out)
                    bs = pred.shape[0]
                    try:
                        target = batch.y.view(bs, OUTPUT_DIM).to(device)
                    except:
                        continue
                    val_loss += criterion(pred, target).item()
            val_loss = val_loss / max(1, len(val_loader))
        print(f"[train] epoch_loss={avg_train_loss:.4f}, val_loss={val_loss:.4f}")
        
        # checkpoint
        os.makedirs("models", exist_ok=True)
        ckpt_path = f"models/persona_gat_epoch{epoch}.pth"
        torch.save({
            "model_state_dict": model.state_dict(),
            "input_dim": TOTAL_INPUT_DIM,
            "hidden_dim": HIDDEN_DIM,
            "output_dim": OUTPUT_DIM,
            "epoch": epoch
        }, ckpt_path)
        print(f"[train] saved checkpoint: {ckpt_path}")
        
        # Save best model
        if val_loss < best_val:
            best_val = val_loss
            best_path = "models/persona_gat_model.pth"
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_dim": TOTAL_INPUT_DIM,
                "hidden_dim": HIDDEN_DIM,
                "output_dim": OUTPUT_DIM,
                "epoch": epoch
            }, best_path)
            print(f"[train] saved best model to {best_path} (val_loss improved)")
            
    print("[train] training complete")
