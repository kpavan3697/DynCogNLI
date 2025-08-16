"""
evaluator.py

This module provides a comprehensive framework for evaluating the persona inference system, 
with a specific focus on the contributions of the Graph Neural Network (GNN) and dynamic context. 
It defines the `evaluate_persona_inference` function, which systematically compares the system's 
outputs against expected values from a predefined test dataset. The evaluation measures include 
Mean Squared Error (MSE) for GNN-derived persona scores and semantic similarity for natural 
language insights. This allows for a quantitative assessment of how effectively the system 
identifies and processes a user's situational and emotional context.
"""

import json
import os
import sys
import numpy as np
from sklearn.metrics import mean_squared_error, cosine_similarity
import torch
from typing import List, Dict, Any, Optional

# Add the project root to the system path to allow for imports from other modules.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import necessary components from the project.
from knowledge.common_sense_client import get_subgraph_for_query
from reasoning.multi_hop_reasoner import run_gnn_reasoning
from context.transformer_encoder import TransformerEncoder

# Ensure that the necessary output directories exist.
os.makedirs("evaluation", exist_ok=True)
os.makedirs("models", exist_ok=True)

def calculate_text_similarity(text1: str, text2: str, encoder: TransformerEncoder) -> float:
    """
    Calculates the semantic similarity between two text strings using a TransformerEncoder.

    This function encodes both texts into vector embeddings and then computes the cosine
    similarity between them. A similarity score of 1.0 indicates identical meaning, while
    0.0 indicates no semantic relationship.

    Args:
        text1 (str): The first text string.
        text2 (str): The second text string.
        encoder (TransformerEncoder): An initialized TransformerEncoder instance.

    Returns:
        float: The cosine similarity score, ranging from -1.0 to 1.0. Returns 0.0
               if the encoder is not available or an error occurs.
    """
    if not encoder or encoder.embedding_dim == 0:
        print("WARNING: TransformerEncoder not available for semantic similarity. Returning 0.")
        return 0.0
    
    try:
        # Encode both texts to get their embeddings.
        embed1 = encoder.encode(text1).unsqueeze(0)  # Add a batch dimension.
        embed2 = encoder.encode(text2).unsqueeze(0)  # Add a batch dimension.
        
        # Calculate cosine similarity using numpy.
        similarity = cosine_similarity(embed1.numpy(), embed2.numpy())[0][0]
        return float(similarity)
    except Exception as e:
        print(f"Error calculating semantic similarity: {e}. Returning 0.")
        return 0.0

def evaluate_persona_inference(data_path: str = "evaluation/evaluation_data.json"):
    """
    Evaluates the persona inference system's performance on a predefined test dataset.

    This is the main evaluation function. It iterates through a series of test cases,
    running the GNN-based reasoning system both with and without dynamic context.
    It then compares the system's output (persona scores and natural language insights)
    against the ground truth provided in the test data. The function calculates and
    reports key metrics like average MSE for scores and average cosine similarity for insights.

    Args:
        data_path (str): The file path to the JSON test data.
    """
    # Load the evaluation test cases from the specified JSON file.
    try:
        with open(data_path, 'r') as f:
            test_cases: List[Dict[str, Any]] = json.load(f)
    except FileNotFoundError:
        print(f"ERROR: Evaluation data file not found at '{data_path}'. Please create it.")
        return

    results: List[Dict[str, Any]] = []
    # Initialize a TransformerEncoder for calculating text similarity.
    text_encoder_for_eval: Optional[TransformerEncoder] = None
    try:
        text_encoder_for_eval = TransformerEncoder()
    except Exception as e:
        print(f"Failed to initialize TransformerEncoder for evaluation: {e}")

    total_mse_scores_with_context: List[float] = []
    total_mse_scores_without_context: List[float] = []
    total_text_sim_with_context: List[float] = []
    total_text_sim_without_context: List[float] = []

    score_labels = ["Urgency", "Emotional Distress", "Practical Need", "Empathy Requirement"]

    for i, case in enumerate(test_cases):
        print(f"\n--- Evaluating Test Case {i+1}: {case['query']} ---")
        query: str = case['query']
        mood: str = case.get('mood', 'Neutral')
        time_of_day: str = case.get('time_of_day', 'Day')
        weather_condition: str = case.get('weather_condition', 'Clear')
        expected_scores: np.ndarray = np.array(list(case['expected_scores'].values()))

        current_case_result: Dict[str, Any] = {
            "query": query,
            "mood": mood,
            "time_of_day": time_of_day,
            "weather_condition": weather_condition,
            "expected_scores": case['expected_scores'],
            "expected_persona_insight_with_context": case['expected_persona_insight_with_context'],
            "expected_persona_insight_without_context": case['expected_persona_insight_without_context']
        }

        # --- Run the Reasoning Pipeline with Context ---
        print("  Running with context...")
        subgraph_with_context = get_subgraph_for_query(query)
        if subgraph_with_context and subgraph_with_context.number_of_nodes() > 0:
            try:
                # The run_gnn_reasoning function is expected to return the GNN scores.
                _, _, _, persona_insight_with_context, predicted_scores_with_context = run_gnn_reasoning(
                    subgraph_with_context, query, mood, time_of_day, weather_condition,
                    ignore_context_in_interpretation=False
                )
                
                if predicted_scores_with_context is not None and isinstance(predicted_scores_with_context, np.ndarray):
                    # Calculate MSE for the predicted scores.
                    mse_scores_with_context = mean_squared_error(expected_scores, predicted_scores_with_context)
                    total_mse_scores_with_context.append(mse_scores_with_context)
                    current_case_result['predicted_scores_with_context'] = predicted_scores_with_context.tolist()
                    current_case_result['mse_scores_with_context'] = mse_scores_with_context
                else:
                    print(f"    WARNING: No valid predicted scores from GNN with context for query: {query}")
                    current_case_result['predicted_scores_with_context'] = None
                    current_case_result['mse_scores_with_context'] = None

                current_case_result['persona_insight_with_context_generated'] = persona_insight_with_context

                # Evaluate semantic similarity of the generated insight.
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

        # --- Run the Reasoning Pipeline without Context ---
        print("  Running without context...")
        subgraph_without_context = get_subgraph_for_query(query)
        if subgraph_without_context and subgraph_without_context.number_of_nodes() > 0:
            try:
                # Use default/neutral context and instruct the function to ignore it.
                _, _, _, persona_insight_without_context, predicted_scores_without_context = run_gnn_reasoning(
                    subgraph_without_context, query, "Neutral", "Day", "Clear",
                    ignore_context_in_interpretation=True
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

    # --- Aggregate and Report Final Results ---
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

    if total_text_sim_with_context and any(isinstance(s, float) for s in total_text_sim_with_context):
        avg_text_sim_with = np.mean([s for s in total_text_sim_with_context if isinstance(s, float)])
        print(f"Average Text Similarity (With Context): {avg_text_sim_with:.4f} (Cosine Similarity)")
    else:
        print("No valid Text Similarity data (With Context).")

    if total_text_sim_without_context and any(isinstance(s, float) for s in total_text_sim_without_context):
        avg_text_sim_without = np.mean([s for s in total_text_sim_without_context if isinstance(s, float)])
        print(f"Average Text Similarity (Without Context): {avg_text_sim_without:.4f} (Cosine Similarity)")
    else:
        print("No valid Text Similarity data (Without Context).")

    # Save detailed results to a JSON file for further analysis.
    print("\nDetailed results saved to evaluation_results.json")
    with open("evaluation_results.json", 'w') as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    # The main entry point for running the evaluation script.
    # It assumes the necessary GNN model checkpoint and evaluation data are in place.
    evaluate_persona_inference()