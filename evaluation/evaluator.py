# evaluation/evaluator.py
"""
evaluator.py

Evaluates the persona inference system by comparing GNN-derived scores and persona insights
against expected values from a test dataset. Calculates metrics such as MSE and semantic similarity
(with and without context) and saves detailed results for analysis.
"""
import json
import os
import sys
import numpy as np
from sklearn.metrics import mean_squared_error, cosine_similarity
import torch

# Add project root to sys.path to allow imports from subdirectories
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common_sense_client import get_subgraph_for_query
from reasoning.multi_hop_reasoner import run_gnn_reasoning # This is the function we're testing
from context.transformer_encoder import TransformerEncoder
# You might need an LLM for semantic similarity if evaluating 'persona_insight' text
# from llm.llm_responder import LLMResponder # If you want to use an LLM for semantic similarity eval

# Ensure output directories exist
os.makedirs("evaluation", exist_ok=True)
os.makedirs("models", exist_ok=True)

def calculate_text_similarity(text1: str, text2: str, encoder: TransformerEncoder) -> float:
    """
    Calculates semantic similarity between two texts using a TransformerEncoder.
    This requires a TransformerEncoder initialized for the evaluation.
    """
    if not encoder or encoder.embedding_dim == 0:
        print("WARNING: TransformerEncoder not available for semantic similarity. Returning 0.")
        return 0.0
    
    try:
        embed1 = encoder.encode(text1).unsqueeze(0) # Add batch dimension
        embed2 = encoder.encode(text2).unsqueeze(0) # Add batch dimension
        
        # Cosine similarity expects inputs to be normalized or it handles it.
        # sentence-transformers embeddings are usually normalized.
        similarity = cosine_similarity(embed1.numpy(), embed2.numpy())[0][0]
        return float(similarity)
    except Exception as e:
        print(f"Error calculating semantic similarity: {e}. Returning 0.")
        return 0.0

def evaluate_persona_inference(data_path="evaluation/evaluation_data.json"):
    """
    Evaluates the persona inference system using a predefined dataset.
    Compares GNN-derived scores and potentially semantic similarity of insights.
    """
    with open(data_path, 'r') as f:
        test_cases = json.load(f)

    results = []
    # Initialize a TransformerEncoder for semantic similarity calculation if needed
    # (This is separate from the one used in the GNN pipeline, or can be shared if carefully managed)
    text_encoder_for_eval = TransformerEncoder() if TransformerEncoder else None

    total_mse_scores_with_context = []
    total_mse_scores_without_context = []
    total_text_sim_with_context = []
    total_text_sim_without_context = []

    score_labels = ["Urgency", "Emotional Distress", "Practical Need", "Empathy Requirement"]

    for i, case in enumerate(test_cases):
        print(f"\n--- Evaluating Test Case {i+1}: {case['query']} ---")
        query = case['query']
        mood = case.get('mood', 'Neutral')
        time_of_day = case.get('time_of_day', 'Day')
        weather_condition = case.get('weather_condition', 'Clear')
        expected_scores = np.array(list(case['expected_scores'].values()))

        current_case_result = {
            "query": query,
            "mood": mood,
            "time_of_day": time_of_day,
            "weather_condition": weather_condition,
            "expected_scores": case['expected_scores'],
            "expected_persona_insight_with_context": case['expected_persona_insight_with_context'],
            "expected_persona_insight_without_context": case['expected_persona_insight_without_context']
        }

        # --- Run with Context ---
        print("  Running with context...")
        subgraph_with_context = get_subgraph_for_query(query)
        if subgraph_with_context and subgraph_with_context.number_of_nodes() > 0:
            try:
                # We expect run_gnn_reasoning to return the raw GNN scores as the 5th element
                _, _, _, persona_insight_with_context, predicted_scores_with_context = run_gnn_reasoning(
                    subgraph_with_context, query, mood, time_of_day, weather_condition,
                    ignore_context_in_interpretation=False
                )
                
                # Check if predicted_scores_with_context is not None and is a numpy array
                if predicted_scores_with_context is not None and isinstance(predicted_scores_with_context, np.ndarray):
                    mse_scores_with_context = mean_squared_error(expected_scores, predicted_scores_with_context)
                    total_mse_scores_with_context.append(mse_scores_with_context)
                    current_case_result['predicted_scores_with_context'] = predicted_scores_with_context.tolist()
                    current_case_result['mse_scores_with_context'] = mse_scores_with_context
                else:
                    print(f"    WARNING: No valid predicted scores from GNN with context for query: {query}")
                    current_case_result['predicted_scores_with_context'] = None
                    current_case_result['mse_scores_with_context'] = None

                current_case_result['persona_insight_with_context_generated'] = persona_insight_with_context

                # Evaluate semantic similarity of generated insight with expected insight
                if text_encoder_for_eval:
                    sim_with_context = calculate_text_similarity(
                        persona_insight_with_context,
                        case['expected_persona_insight_with_context'],
                        text_encoder_for_eval
                    )
                    total_text_sim_with_context.append(sim_with_context)
                    current_case_result['text_similarity_with_context'] = sim_with_context
                else:
                    current_case_result['text_similarity_with_context'] = "N/A (Encoder not loaded)"

            except Exception as e:
                print(f"    Error during context-aware reasoning for query '{query}': {e}")
                current_case_result['persona_insight_with_context_generated'] = f"ERROR: {e}"
                current_case_result['mse_scores_with_context'] = None
                current_case_result['text_similarity_with_context'] = None
        else:
            print(f"    Skipping with context: No subgraph for query '{query}'")
            current_case_result['persona_insight_with_context_generated'] = "No subgraph generated."
            current_case_result['mse_scores_with_context'] = None
            current_case_result['text_similarity_with_context'] = None


        # --- Run without Context ---
        print("  Running without context...")
        subgraph_without_context = get_subgraph_for_query(query) # Subgraph stays the same
        if subgraph_without_context and subgraph_without_context.number_of_nodes() > 0:
            try:
                _, _, _, persona_insight_without_context, predicted_scores_without_context = run_gnn_reasoning(
                    subgraph_without_context, query, "Neutral", "Day", "Clear", # Pass default/neutral context
                    ignore_context_in_interpretation=True # Crucially, ignore in interpretation
                )

                if predicted_scores_without_context is not None and isinstance(predicted_scores_without_context, np.ndarray):
                    mse_scores_without_context = mean_squared_error(expected_scores, predicted_scores_without_context)
                    total_mse_scores_without_context.append(mse_scores_without_context)
                    current_case_result['predicted_scores_without_context'] = predicted_scores_without_context.tolist()
                    current_case_result['mse_scores_without_context'] = mse_scores_without_context
                else:
                    print(f"    WARNING: No valid predicted scores from GNN without context for query: {query}")
                    current_case_result['predicted_scores_without_context'] = None
                    current_case_result['mse_scores_without_context'] = None

                current_case_result['persona_insight_without_context_generated'] = persona_insight_without_context

                if text_encoder_for_eval:
                    sim_without_context = calculate_text_similarity(
                        persona_insight_without_context,
                        case['expected_persona_insight_without_context'],
                        text_encoder_for_eval
                    )
                    total_text_sim_without_context.append(sim_without_context)
                    current_case_result['text_similarity_without_context'] = sim_without_context
                else:
                    current_case_result['text_similarity_without_context'] = "N/A (Encoder not loaded)"

            except Exception as e:
                print(f"    Error during context-agnostic reasoning for query '{query}': {e}")
                current_case_result['persona_insight_without_context_generated'] = f"ERROR: {e}"
                current_case_result['mse_scores_without_context'] = None
                current_case_result['text_similarity_without_context'] = None
        else:
            print(f"    Skipping without context: No subgraph for query '{query}'")
            current_case_result['persona_insight_without_context_generated'] = "No subgraph generated."
            current_case_result['mse_scores_without_context'] = None
            current_case_result['text_similarity_without_context'] = None

        results.append(current_case_result)

    # --- Aggregate and Report ---
    print(f"\n--- Evaluation Summary for {len(results)} Test Cases ---")

    if total_mse_scores_with_context:
        avg_mse_with = np.mean(total_mse_scores_with_context)
        print(f"Average MSE for GNN Scores (With Context): {avg_mse_with:.4f}")
    else:
        print("No valid MSE data for GNN Scores (With Context).")

    if total_mse_scores_without_context:
        avg_mse_without = np.mean(total_mse_scores_without_context)
        print(f"Average MSE for GNN Scores (Without Context): {avg_mse_without:.4f}")
    else:
        print("No valid MSE data for GNN Scores (Without Context).")

    if total_text_sim_with_context and not all(s == "N/A (Encoder not loaded)" for s in total_text_sim_with_context):
        avg_text_sim_with = np.mean([s for s in total_text_sim_with_context if isinstance(s, float)])
        print(f"Average Text Similarity (With Context): {avg_text_sim_with:.4f} (Cosine Similarity)")
    else:
        print("No valid Text Similarity data (With Context).")

    if total_text_sim_without_context and not all(s == "N/A (Encoder not loaded)" for s in total_text_sim_without_context):
        avg_text_sim_without = np.mean([s for s in total_text_sim_without_context if isinstance(s, float)])
        print(f"Average Text Similarity (Without Context): {avg_text_sim_without:.4f} (Cosine Similarity)")
    else:
        print("No valid Text Similarity data (Without Context).")

    print("\nDetailed results saved to evaluation_results.json")

    with open("evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    # To run this, ensure your GNN model is in models/persona_gnn_model.pth
    # Or, if you don't have one trained, it will use a dummy model.
    evaluate_persona_inference()
