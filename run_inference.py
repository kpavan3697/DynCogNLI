"""
run_inference.py

This script provides a quick, command-line interface to run the persona inference
logic. It's useful for testing the model outside of the Streamlit application,
running batch inferences, or simply for a quick debug session. It parses command-line
arguments for the query and contextual factors, then calls the core reasoning
function to get and print the results.
"""
import argparse
from reasoning.multi_hop_reasoner import run_gnn_reasoning

# Set up the command-line argument parser
parser = argparse.ArgumentParser(
    description="Run persona inference from the command line."
)
parser.add_argument(
    "--query",
    type=str,
    default="My laptop screen cracked.",
    help="The user's query or situation to analyze."
)
parser.add_argument(
    "--mood",
    type=str,
    default="Neutral",
    help="The contextual mood (e.g., 'Stressed', 'Happy')."
)
parser.add_argument(
    "--time",
    type=str,
    default="Day",
    help="The time of day (e.g., 'Day', 'Night')."
)
parser.add_argument(
    "--weather",
    type=str,
    default="Clear",
    help="The weather condition (e.g., 'Rainy', 'Cloudy')."
)
parser.add_argument(
    "--ignore_context",
    action="store_true",
    help="A flag to ignore all contextual factors during inference."
)

# Parse the arguments from the command line
args = parser.parse_args()

# Run the core GNN reasoning function with the provided arguments
# Note: The `run_gnn_reasoning` function needs to be implemented and available
# in `reasoning/multi_hop_reasoner.py` for this to work with your actual model.
insights, scores, image = run_gnn_reasoning(
    args.query,
    args.mood,
    args.time,
    args.weather,
    ignore_context=args.ignore_context
)

# Print the results to the console
print("\nINSIGHTS:\n")
for line in insights:
    print(line)
print("\nSCORES:\n", scores)
print("\nGRAPH IMAGE:", image)
