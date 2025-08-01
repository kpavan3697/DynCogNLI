# reasoning/multi_hop_reasoner.py
"""
multi_hop_reasoner.py

Implements multi-hop reasoning over knowledge graphs using the trained GNN model.
Supports inference, persona scoring, and context-aware reasoning for user queries.
"""
import torch
import networkx as nx
import os
import matplotlib.pyplot as plt
import numpy as np
import collections
import time
from typing import Tuple, Dict, Any, Optional, List

# Ensure the parent directory of 'reasoning' is in sys.path for sibling imports
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from reasoning.gnn_model import GATModel
from reasoning.graph_builder import nx_to_pyg_data, fetch_conceptnet_relations
from context.transformer_encoder import TransformerEncoder
from context.context_encoder import ContextEncoder

# Global variables for model and encoder to avoid re-loading
_gnn_model = None
_transformer_encoder = None
_context_encoder = None
_model_expected_input_dim = None

# Constants for persona dimensions
OUTPUT_DIM = 4 # Urgency, Emotional Distress, Practical Need, Empathy Requirement

# Default dimensions for encoders if they fail to load (should match actual encoder dimensions)
DEFAULT_TRANSFORMER_EMBEDDING_DIM = 384 # Matches MiniLM-L6-v2 default
DEFAULT_CONTEXT_TOTAL_DIM = 8 + 5 + 6 # Mood (8) + Time (5) + Weather (6)

BASE_NODE_FEATURE_SIZE = 0 

def _load_gnn_model(model_path="models/persona_gnn_model.pth"):
    """
    Loads the GNN model from the specified path, or initializes a dummy model if not found.
    Caches the loaded model and its expected input dimension.
    """
    global _gnn_model, _model_expected_input_dim
    if _gnn_model is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=device)

                input_dim = state_dict.get('input_dim')
                if input_dim is None:
                    raise ValueError(f"Model state dict at {model_path} does not contain 'input_dim'. "
                                     "Please retrain your GNN with the updated train_gnn.py script "
                                     "which saves 'input_dim' to the state_dict.")

                hidden_dim = state_dict.get('hidden_dim', 64) # Default if not found

                _gnn_model = GATModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=OUTPUT_DIM).to(device)
                _gnn_model.load_state_dict(state_dict['model_state_dict'])
                _gnn_model.eval()
                _model_expected_input_dim = input_dim
                print(f"GNN Model loaded successfully from {model_path} with input_dim={input_dim}.")
            except Exception as e:
                print(f"Error loading GNN model from {model_path}: {e}")
                _gnn_model = None
        else:
            print(f"WARNING: GNN model not found at {model_path}. "
                  f"Please run `python train_gnn.py` first. Initializing dummy model configuration.")
            
            dummy_input_dim = (DEFAULT_TRANSFORMER_EMBEDDING_DIM +
                               DEFAULT_CONTEXT_TOTAL_DIM +
                               BASE_NODE_FEATURE_SIZE)
            
            _gnn_model = GATModel(input_dim=dummy_input_dim, hidden_dim=64, output_dim=OUTPUT_DIM).to(device)
            _gnn_model.eval()
            _model_expected_input_dim = dummy_input_dim
            print(f"Dummy GNN Model initialized with input_dim={dummy_input_dim}.")
    return _gnn_model

def get_transformer_encoder():
    """Returns the cached TransformerEncoder instance, initializing it if necessary."""
    global _transformer_encoder
    if _transformer_encoder is None:
        try:
            _transformer_encoder = TransformerEncoder()
        except Exception as e:
            print(f"Error initializing TransformerEncoder: {e}")
            _transformer_encoder = None
    return _transformer_encoder

def get_context_encoder():
    """Returns the cached ContextEncoder instance, initializing it if necessary."""
    global _context_encoder
    if _context_encoder is None:
        try:
            _context_encoder = ContextEncoder()
        except Exception as e:
            print(f"Error initializing ContextEncoder: {e}")
            _context_encoder = None
    return _context_encoder

def visualize_subgraph(nx_graph: nx.Graph, query: str, output_path="subgraph_explanation.png") -> Optional[str]:
    """
    Visualizes the NetworkX subgraph and saves it as a PNG image.
    
    Args:
        nx_graph (nx.Graph): The input NetworkX graph.
        query (str): The initial user query (used for highlighting and title).
        output_path (str): The path to save the visualization image.

    Returns:
        Optional[str]: The path to the saved image if successful, else None.
    """
    if not nx_graph or nx_graph.number_of_nodes() == 0:
        print("No graph to visualize.")
        return None
    
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(nx_graph, k=0.7, iterations=50, seed=42) 
    
    nx.draw_networkx_nodes(nx_graph, pos, node_color='skyblue', node_size=2500, alpha=0.7)
    nx.draw_networkx_edges(nx_graph, pos, edge_color='gray', alpha=0.7)
    
    nx.draw_networkx_labels(nx_graph, pos, font_size=10, font_weight='bold')
    
    # Highlight query nodes
    query_terms = set(word.lower() for word in query.lower().split() if len(word) > 2)
    highlight_nodes = [node for node in nx_graph.nodes() if any(term in node.lower() for term in query_terms)]
    if highlight_nodes:
        nx.draw_networkx_nodes(nx_graph, pos, nodelist=highlight_nodes, node_color='lightcoral', node_size=3000)
    
    plt.title("Relevant Subgraph for Query: " + query)
    plt.axis('off')
    try:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        return output_path
    except Exception as e:
        print(f"Error saving subgraph visualization to {output_path}: {e}")
        plt.close()
        return None

def interpret_gnn_output_for_persona(scores: torch.Tensor, user_query: str, user_mood: str, time_of_day: str, weather_condition: str) -> List[str]:
    """
    Interprets the GNN's output (4 persona scores, already 0-1 scaled) into human-readable insights.
    
    Args:
        scores (torch.Tensor): A tensor of shape (4,) containing the 0-1 scaled persona scores.
        user_query (str): The user's original query.
        user_mood (str): User's mood.
        time_of_day (str): Time of day.
        weather_condition (str): Weather condition.

    Returns:
        List[str]: A list of insight strings.
    """
    if scores.numel() != OUTPUT_DIM:
        print(f"WARNING: interpret_gnn_output_for_persona received a tensor of shape {scores.shape}, expected ({OUTPUT_DIM},). Using default interpretation.")
        scores = torch.zeros(OUTPUT_DIM)
    
    urgency, emotional_distress, practical_need, empathy_requirement = scores.tolist()

    insight = []
    
    insight.append(f"Based on the query '{user_query}' and the contextual factors ({user_mood}, {time_of_day}, {weather_condition}):")
    insight.append("") # Spacer

    # Detailed interpretation of scores
    insight.append(f"**Urgency:** {urgency:.2f} - " + 
                     ("Highly urgent, requiring immediate attention." if urgency > 0.7 else
                      "Some urgency to the situation." if urgency > 0.4 else
                      "Low urgency."))

    insight.append(f"**Emotional Distress:** {emotional_distress:.2f} - " +
                     ("Significant emotional distress likely." if emotional_distress > 0.7 else
                      "Some emotional discomfort possible." if emotional_distress > 0.4 else
                      "Appears emotionally stable."))

    insight.append(f"**Practical Need:** {practical_need:.2f} - " +
                     ("Immediate, concrete steps needed to resolve the core issue." if practical_need > 0.7 else
                      "Practical assistance or information needed." if practical_need > 0.4 else
                      "Not primarily looking for direct practical help."))

    insight.append(f"**Empathy Requirement:** {empathy_requirement:.2f} - " +
                     ("High degree of empathy and reassurance required." if empathy_requirement > 0.7 else
                      "Moderate empathy and understanding needed." if empathy_requirement > 0.4 else
                      "A direct and straightforward approach might be suitable."))
    
    insight.append("\n**Recommended Approach:**")
    summary_points = []
    if urgency > 0.7: summary_points.append("act swiftly")
    elif urgency > 0.4: summary_points.append("address with attention to time")
    else: summary_points.append("take a considered approach")

    if emotional_distress > 0.7: summary_points.append("be highly empathetic and reassuring")
    elif emotional_distress > 0.4: summary_points.append("show understanding for potential discomfort")
    else: summary_points.append("maintain a calm and objective tone")

    if practical_need > 0.7: summary_points.append("offer concrete solutions and actionable advice")
    elif practical_need > 0.4: summary_points.append("provide useful information or guidance")
    else: summary_points.append("focus on general discussion or informational support")

    if empathy_requirement > 0.7: summary_points.append("prioritize emotional support")
    elif empathy_requirement > 0.4: summary_points.append("balance practicality with emotional awareness")
    else: summary_points.append("be direct and efficient")

    insight.append(f"    - To effectively respond, you should: {', '.join(summary_points)}.")
    
    return insight

def run_gnn_reasoning(
    query: str,
    mood: str,
    time_of_day: str,
    weather_condition: str,
    ignore_context: bool = False
) -> Tuple[List[str], Dict[str, float], Optional[str]]:
    """
    Main function to orchestrate the GNN reasoning process and get persona insights.
    
    Args:
        query (str): The user's input query.
        mood (str): The selected mood context.
        time_of_day (str): The selected time of day context.
        weather_condition (str): The selected weather condition context.
        ignore_context (bool): If True, context embeddings are zeroed out for GNN input.

    Returns:
        Tuple[List[str], Dict[str, float], Optional[str]]: 
        - A list of natural language persona insight strings.
        - A dictionary of persona scores (Urgency, Emotional Distress, Practical Need, Empathy Requirement).
        - Optional[str]: Path to the generated subgraph visualization image, or None if not generated.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = _load_gnn_model()
    if model is None:
        return ["Error: GNN model could not be loaded. Please train the model first by running `python train_gnn.py`."], \
               {"Urgency": 0.0, "Emotional Distress": 0.0, "Practical Need": 0.0, "Empathy Requirement": 0.0}, \
               None

    # Get encoders
    transformer_encoder = get_transformer_encoder()
    context_encoder = get_context_encoder()

    query_embedding_tensor = None
    if transformer_encoder:
        query_embedding_tensor = transformer_encoder.encode(query).squeeze(0).to(device)
    else:
        print("WARNING: TransformerEncoder not available. Using zero embedding for query.")
        query_embedding_tensor = torch.zeros(DEFAULT_TRANSFORMER_EMBEDDING_DIM).to(device)

    context_embedding_tensor = None
    if not ignore_context:
        if context_encoder:
            context_embedding_tensor = context_encoder.encode(mood, time_of_day, weather_condition).to(device)
        else:
            print("WARNING: ContextEncoder not available. Using zero embedding for context.")
            context_embedding_tensor = torch.zeros(DEFAULT_CONTEXT_TOTAL_DIM).to(device)
    else:
        print("INFO: Context is being ignored for GNN reasoning.")
        context_embedding_tensor = torch.zeros(DEFAULT_CONTEXT_TOTAL_DIM).to(device)


    total_input_feature_dim = _model_expected_input_dim 
    if total_input_feature_dim is None or total_input_feature_dim == 0:
        total_input_feature_dim = (BASE_NODE_FEATURE_SIZE + 
                                (query_embedding_tensor.shape[0] if query_embedding_tensor is not None else 0) +
                                (context_embedding_tensor.shape[0] if context_embedding_tensor is not None else 0))
        if total_input_feature_dim == 0:
            print("CRITICAL ERROR: Calculated GNN input dimension is zero. Check BASE_NODE_FEATURE_SIZE and encoder dimensions.")
            return ["Error: GNN input dimension is zero. Please check console for details."], {
                "Urgency": 0.0, "Emotional Distress": 0.0, "Practical Need": 0.0, "Empathy Requirement": 0.0
            }, None

    # --- MODIFIED GRAPH GENERATION LOGIC ---
    # Attempt to build a graph from multiple relevant terms in the query
    keywords_to_search = [word.lower() for word in query.split() if len(word) > 2] # Simple heuristic: words > 2 chars
    
    # Initialize combined_graph as a DiGraph (directed graph)
    combined_graph = nx.DiGraph() 
    fetched_any_graph = False

    if not keywords_to_search:
        keywords_to_search = [query.lower()] # Fallback to full query if no suitable keywords

    for keyword in keywords_to_search:
        temp_graph = fetch_conceptnet_relations(keyword, depth=1, max_nodes=5, max_edges=5) # Smaller sub-graphs per keyword
        if temp_graph and temp_graph.number_of_nodes() > 0:
            # nx.compose works with DiGraphs and Graphs, but they must be consistent
            combined_graph = nx.compose(combined_graph, temp_graph)
            fetched_any_graph = True
            if combined_graph.number_of_nodes() > 50: # Cap the total graph size for performance
                print("INFO: Combined graph reached max node limit for efficiency.")
                break
    
    # If no graph could be built for any keyword, try with the full query as a single node
    if not fetched_any_graph and query.strip():
        print(f"WARNING: No graph could be built for individual keywords. Attempting with full query as a single entity: '{query}'.")
        # Ensure fetch_conceptnet_relations also produces a DiGraph for consistency
        combined_graph = fetch_conceptnet_relations(query.lower().strip(), depth=1, max_nodes=10, max_edges=10)
        if combined_graph and combined_graph.number_of_nodes() > 0:
            fetched_any_graph = True

    nx_graph = combined_graph if fetched_any_graph else nx.DiGraph() # Use the combined graph or an empty DiGraph
    # --- END MODIFIED GRAPH GENERATION LOGIC ---

    graph_level_scores_tensor = torch.zeros(OUTPUT_DIM).to(device)
    explanation_path = None

    if not nx_graph or nx_graph.number_of_nodes() == 0:
        print(f"WARNING: No valid ConceptNet graph could be built for query '{query}'. Returning default scores.")
    else:
        pyg_data, original_node_mapping = nx_to_pyg_data(
            nx_graph,
            feature_dim=total_input_feature_dim,
            query_embedding=query_embedding_tensor,
            context_embedding=context_embedding_tensor
        )

        if pyg_data is None or pyg_data.x is None or pyg_data.x.numel() == 0:
            print("WARNING: PyTorch Geometric Data object is empty after conversion or features are missing. Returning default scores.")
        else:
            data = pyg_data.to(device)
            model.eval()
            with torch.no_grad():
                out = model(data)

                if out.dim() == 2 and out.size(0) == 1 and out.size(1) == OUTPUT_DIM:
                    graph_level_logits = out.squeeze(0)
                elif out.dim() == 2 and out.size(1) == OUTPUT_DIM:
                    graph_level_logits = out.mean(dim=0)
                elif out.dim() == 1 and out.size(0) == OUTPUT_DIM:
                    graph_level_logits = out
                else:
                    print(f"WARNING: Unexpected GNN output shape: {out.shape}. Defaulting logits for interpretation.")
                    graph_level_logits = torch.zeros(OUTPUT_DIM).to(device)

                graph_level_scores_tensor = torch.sigmoid(graph_level_logits)

            explanation_path = visualize_subgraph(
                nx_graph, 
                query, 
                output_path=os.path.join("explanation_images", f"subgraph_{time.time()}.png")
            )

    insight_list = interpret_gnn_output_for_persona(
        graph_level_scores_tensor.cpu(),
        query,
        mood,
        time_of_day,
        weather_condition
    )

    scores_dict = {
        "Urgency": graph_level_scores_tensor[0].item(),
        "Emotional Distress": graph_level_scores_tensor[1].item(),
        "Practical Need": graph_level_scores_tensor[2].item(),
        "Empathy Requirement": graph_level_scores_tensor[3].item()
    }
    
    return insight_list, scores_dict, explanation_path