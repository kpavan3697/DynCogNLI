# run_inference.py
"""
Quick inference script: loads saved model and runs run_gnn_reasoning
Usage:
    python run_inference.py --query "My car broke down" --mood Stressed --time Night --weather Rainy
"""
import argparse
from reasoning.multi_hop_reasoner import run_gnn_reasoning

parser = argparse.ArgumentParser()
parser.add_argument("--query", type=str, default="My laptop screen cracked.")
parser.add_argument("--mood", type=str, default="Neutral")
parser.add_argument("--time", type=str, default="Day")
parser.add_argument("--weather", type=str, default="Clear")
parser.add_argument("--ignore_context", action="store_true")
args = parser.parse_args()

insights, scores, image = run_gnn_reasoning(args.query, args.mood, args.time, args.weather, ignore_context=args.ignore_context)
print("\nINSIGHTS:\n")
for line in insights:
    print(line)
print("\nSCORES:\n", scores)
print("\nGRAPH IMAGE:", image)
