# reasoning/graph_builder.py
import networkx as nx
import requests
import json
import collections
import time
import torch
import numpy as np
from torch_geometric.data import Data
import certifi # Keep import even if not strictly used for verify=False
from typing import Optional, Tuple, Dict

# Suppress InsecureRequestWarning if verify=False is used
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Set to False to explicitly disable certificate verification
# If you want to enable verification, change to True and ensure certifi is installed
_requests_ca_bundle = False

def fetch_conceptnet_relations(start_node: str, depth: int = 2, max_nodes: int = 50, max_edges: int = 100) -> nx.DiGraph:
    graph = nx.DiGraph()
    visited_nodes = set()
    queue = collections.deque([(start_node.lower(), 0)])
    
    last_request_time = 0
    # Minimum interval between API requests to avoid rate limiting
    # 0.2 seconds means max 5 requests per second
    # Increase this if you still hit rate limits (e.g., to 0.5 or 1.0)
    min_interval = 0.2

    print(f"DEBUG: Starting ConceptNet fetch for '{start_node}' up to depth {depth}...")

    while queue and len(graph.nodes()) < max_nodes and len(graph.edges()) < max_edges:
        current_node, current_depth = queue.popleft()

        if current_node in visited_nodes:
            continue

        visited_nodes.add(current_node)
        # Only add node to graph if it's new and we haven't hit max_nodes
        if len(graph.nodes()) < max_nodes:
            graph.add_node(current_node)
        else:
            # If we've hit max_nodes, don't process this node or its edges further
            # but ensure existing nodes are processed from queue if possible
            continue

        if current_depth >= depth:
            # print(f"DEBUG: Max depth ({depth}) reached for '{current_node}'.")
            continue

        url = f"http://api.conceptnet.io/c/en/{current_node}"
        
        current_time = time.time()
        if current_time - last_request_time < min_interval:
            sleep_duration = min_interval - (current_time - last_request_time)
            # print(f"DEBUG: Sleeping for {sleep_duration:.2f}s before next request.")
            time.sleep(sleep_duration)
        last_request_time = time.time()

        print(f"DEBUG: Fetching URL: {url} (Current node: '{current_node}', Depth: {current_depth}, Graph nodes: {len(graph.nodes())}, Graph edges: {len(graph.edges())})")

        try:
            # Added a timeout to prevent requests from hanging indefinitely
            # 10 seconds is usually reasonable. Increase if your network is very slow.
            response = requests.get(url, params={"limit": 50}, verify=_requests_ca_bundle, timeout=10)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            data = response.json()
            # print(f"DEBUG: Successfully fetched data for {current_node}. Edges found: {len(data.get('edges', []))}")

            for edge in data.get('edges', []):
                if len(graph.edges()) >= max_edges:
                    # print(f"DEBUG: Max edges ({max_edges}) reached. Stopping edge processing for {current_node}.")
                    break

                relation = edge['rel']['label']
                start_label = edge['start']['label'].lower()
                end_label = edge['end']['label'].lower()

                # Only consider English relations
                if not (edge['start']['language'] == 'en' and edge['end']['language'] == 'en'):
                    continue
                
                # Add edge and potential new nodes if within limits
                if len(graph.nodes()) < max_nodes:
                    graph.add_edge(start_label, end_label, relation=relation, weight=edge['weight'])
                else:
                    # If max_nodes reached, don't add new nodes from edges, but existing nodes can still form edges
                    if start_label in graph and end_label in graph:
                         graph.add_edge(start_label, end_label, relation=relation, weight=edge['weight'])
                    continue # Stop processing this edge if it would add new nodes past limit
                    
                if start_label not in visited_nodes and len(graph.nodes()) < max_nodes:
                    queue.append((start_label, current_depth + 1))
                if end_label not in visited_nodes and len(graph.nodes()) < max_nodes:
                    queue.append((end_label, current_depth + 1))
        
        except requests.exceptions.Timeout:
            print(f"ERROR: Request to ConceptNet timed out for '{current_node}' after 10 seconds.")
            # Continue to next node in queue rather than crashing
            continue
        except requests.exceptions.RequestException as e:
            print(f"ERROR: Network or API error fetching ConceptNet data for '{current_node}': {e}")
            continue
        except json.JSONDecodeError as e:
            print(f"ERROR: JSON decoding error for '{current_node}' (invalid response from API): {e}")
            continue
        except Exception as e: # Catch any other unexpected errors
            print(f"CRITICAL ERROR: Unexpected error during ConceptNet fetch for '{current_node}': {e}")
            continue
        
    print(f"DEBUG: Completed ConceptNet fetch for '{start_node}'. Final graph has {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
    return graph

def nx_to_pyg_data(
    nx_graph: nx.Graph,
    feature_dim: int, # This is the total expected dimension by the GNN
    query_embedding: Optional[torch.Tensor] = None,
    context_embedding: Optional[torch.Tensor] = None
) -> Tuple[Data, Dict[int, str]]:
    """
    Converts a NetworkX graph to a PyTorch Geometric Data object.
    Includes options to inject query and context embeddings into node features.

    Args:
        nx_graph (nx.Graph): The input NetworkX graph.
        feature_dim (int): The desired total dimension for node features.
                           This should match the input_dim of your GNN model.
        query_embedding (Optional[torch.Tensor]): A pre-computed embedding of the user query.
                                                   If provided, it will be integrated into node features.
        context_embedding (Optional[torch.Tensor]): A pre-computed embedding of the context
                                                     (mood, time, weather). If provided, it will
                                                     be integrated into node features.

    Returns:
        Tuple[Data, Dict[int, str]]: A PyG Data object and a mapping from PyG index to original node name.
    """
    if not nx_graph or nx_graph.number_of_nodes() == 0:
        # Return an empty Data object with correctly shaped x, even if empty
        return Data(x=torch.empty((0, feature_dim)), edge_index=torch.empty((2,0), dtype=torch.long)), {}

    node_list = list(nx_graph.nodes())
    node_to_idx = {node: i for i, node in enumerate(node_list)}
    idx_to_node = {i: node for node, i in node_to_idx.items()}

    x = []
    edge_indices = []

    # Calculate sizes of query and context embeddings
    query_embed_len = query_embedding.shape[0] if query_embedding is not None else 0
    context_embed_len = context_embedding.shape[0] if context_embedding is not None else 0
    
    # The remaining space after query and context is for base node features
    base_feature_size = feature_dim - query_embed_len - context_embed_len
    if base_feature_size < 0:
        print(f"WARNING in nx_to_pyg_data: feature_dim ({feature_dim}) is smaller than combined embedding dimensions ({query_embed_len + context_embed_len}). "
              "Adjusting base_feature_size to 0 and potentially truncating embeddings.")
        base_feature_size = 0
        # If combined embeddings are larger than feature_dim, truncate.
        # This case implies an incorrect feature_dim passed or model mismatch.
        # It's better to ensure feature_dim is calculated correctly upstream.

    for i, node in enumerate(node_list):
        node_features_parts = []

        # 1. Base Node Features
        node_base_features = np.zeros(base_feature_size)
        if 'feature' in nx_graph.nodes[node] and nx_graph.nodes[node]['feature'] is not None:
            existing_features = np.array(nx_graph.nodes[node]['feature'])
            copy_len = min(base_feature_size, len(existing_features))
            node_base_features[:copy_len] = existing_features[:copy_len]
        elif base_feature_size > 0:
            # Add a small random component to distinguish nodes if no specific features
            node_base_features = np.random.rand(base_feature_size) * 0.1
        
        if base_feature_size > 0: # Only append if there's actual space/features
            node_features_parts.append(torch.from_numpy(node_base_features).float())

        # 2. Query Embedding (if provided and fits)
        if query_embedding is not None and query_embed_len > 0:
            node_features_parts.append(query_embedding)

        # 3. Context Embedding (if provided and fits)
        if context_embedding is not None and context_embed_len > 0:
            node_features_parts.append(context_embedding)

        # Concatenate all parts
        if node_features_parts:
            node_features_tensor = torch.cat(node_features_parts)
        else:
            # If no features from any source, create a zero tensor of target dimension
            node_features_tensor = torch.zeros(feature_dim)


        # Ensure final feature vector matches the target feature_dim
        if node_features_tensor.shape[0] < feature_dim:
            padding = torch.zeros(feature_dim - node_features_tensor.shape[0])
            node_features_tensor = torch.cat((node_features_tensor, padding), dim=0)
        elif node_features_tensor.shape[0] > feature_dim:
            node_features_tensor = node_features_tensor[:feature_dim]

        x.append(node_features_tensor)

    for u, v in nx_graph.edges():
        if u in node_to_idx and v in node_to_idx:
            edge_indices.append([node_to_idx[u], node_to_idx[v]])
            if not nx_graph.is_directed(): # Add reverse for undirected interpretation
                edge_indices.append([node_to_idx[v], node_to_idx[u]])

    x_tensor = torch.stack(x) if x else torch.empty((0, feature_dim))
    edge_index_tensor = torch.tensor(edge_indices, dtype=torch.long).t().contiguous() if edge_indices else torch.empty((2, 0), dtype=torch.long)

    data = Data(x=x_tensor, edge_index=edge_index_tensor)

    return data, idx_to_node