import json
import pandas as pd
import numpy as np
from typing import Dict, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def get_performance_metrics(file_path: str = "evaluation/evaluation_results.json") -> Dict[str, Dict[str, Any]]:
    """
    Loads and processes the raw evaluation results to calculate aggregate performance metrics.

    Args:
        file_path (str): The path to the JSON file containing the evaluation results.

    Returns:
        Dict[str, Dict[str, Any]]: A dictionary containing aggregated metrics for the GNN
                                    and the baseline model, ready for display.
    """
    try:
        with open(file_path, 'r') as f:
            results = json.load(f)
    except FileNotFoundError:
        print(f"Warning: Evaluation results file not found at '{file_path}'. Using placeholder data.")
        return {
            'GNN Model': {
                'Mean Squared Error (MSE)': "N/A",
                'Mean Absolute Error (MAE)': "N/A",
                'R-squared Score ($R^2$)': "N/A"
            },
            'Baseline FFN': {
                'Mean Squared Error (MSE)': "N/A",
                'Mean Absolute Error (MAE)': "N/A",
                'R-squared Score ($R^2$)': "N/A"
            }
        }
    except json.JSONDecodeError:
        print(f"Error: The file '{file_path}' contains invalid JSON. Using placeholder data.")
        return {
            'GNN Model': {
                'Mean Squared Error (MSE)': "N/A",
                'Mean Absolute Error (MAE)': "N/A",
                'R-squared Score ($R^2$)': "N/A"
            },
            'Baseline FFN': {
                'Mean Squared Error (MSE)': "N/A",
                'Mean Absolute Error (MAE)': "N/A",
                'R-squared Score ($R^2$)': "N/A"
            }
        }

    # Lists to store metrics for each test case
    gnn_metrics = {'mse': [], 'mae': [], 'r2': []}
    baseline_metrics = {'mse': [], 'mae': [], 'r2': []}

    for case in results:
        expected = case.get('expected_scores')
        predicted_gnn = case.get('predicted_scores_with_context')
        predicted_baseline = case.get('predicted_scores_without_context')
        
        # Skip this case if there's no ground truth data
        if not expected:
            continue

        # Helper function to process metrics for a given model
        def process_model(predicted_scores, metrics_dict):
            # Check if there are any predicted scores for this model
            if not predicted_scores:
                return

            try:
                # Find labels that are present in BOTH expected and predicted scores
                valid_labels = sorted([label for label in expected.keys() if label in predicted_scores.keys()])

                # If no common labels are found, raise an error to skip this case
                if not valid_labels:
                    raise ValueError(f"No common labels found between expected and predicted scores.")
                
                # Create the ordered lists of values based on the valid labels
                predicted_values = [predicted_scores[label] for label in valid_labels]
                expected_values = [expected[label] for label in valid_labels]
                
                # Calculate metrics and append to the lists
                metrics_dict['mse'].append(mean_squared_error(expected_values, predicted_values))
                metrics_dict['mae'].append(mean_absolute_error(expected_values, predicted_values))
                metrics_dict['r2'].append(r2_score(expected_values, predicted_values))
            
            except (ValueError, KeyError) as e:
                # Print a more specific warning and skip this case
                print(f"Warning: Skipping a case due to data error: {e}")

        # Process GNN Model
        process_model(predicted_gnn, gnn_metrics)
        
        # Process Baseline FFN
        process_model(predicted_baseline, baseline_metrics)

    # Calculate and format average values, handling empty lists
    def format_metric(metric_list):
        return f"{np.mean(metric_list):.4f}" if metric_list else "N/A"

    return {
        'GNN Model': {
            'Mean Squared Error (MSE)': format_metric(gnn_metrics['mse']),
            'Mean Absolute Error (MAE)': format_metric(gnn_metrics['mae']),
            'R-squared Score ($R^2$)': format_metric(gnn_metrics['r2'])
        },
        'Baseline FFN': {
            'Mean Squared Error (MSE)': format_metric(baseline_metrics['mse']),
            'Mean Absolute Error (MAE)': format_metric(baseline_metrics['mae']),
            'R-squared Score ($R^2$)': format_metric(baseline_metrics['r2'])
        }
    }
