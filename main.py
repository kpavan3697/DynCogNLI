# main.py
import networkx as nx
import os
import torch
from datetime import datetime # Import datetime for current time
import collections # Import collections for deque in build_concept_graph

# Imports from your project structure
from preprocessing.conceptnet_loader import load_conceptnet
from preprocessing.atomic_loader import load_atomic_tsv
from knowledge.graph_builder import build_knowledge_graph
from knowledge.retriever import extract_concepts, retrieve_combined_subgraph
from utils.graph_utils import normalize_and_match_concepts
from reasoning.multi_hop_reasoner import run_gnn_reasoning
from context.context_manager import ContextManager
from context.real_time_updater import RealTimeContextUpdater, build_prompt, get_model_response

# Load knowledge graphs once
conceptnet_path = 'data/conceptnet/conceptnet-assertions-5.7.0.csv'
atomic_path = 'data/atomic2020/train.tsv'

# Limit graph size for faster loading and execution, especially during development
print("Loading ConceptNet...")
conceptnet_triples = load_conceptnet(conceptnet_path)
conceptnet_graph = build_knowledge_graph(conceptnet_triples[:100000]) # Increased for more concepts
print(f"ConceptNet loaded with {conceptnet_graph.number_of_nodes()} nodes and {conceptnet_graph.number_of_edges()} edges.")

print("Loading ATOMIC...")
atomic_triples = load_atomic_tsv(atomic_path)
atomic_graph = build_knowledge_graph(atomic_triples[:50000]) # Increased for more concepts
print(f"ATOMIC loaded with {atomic_graph.number_of_nodes()} nodes and {atomic_graph.number_of_edges()} edges.")

# Initialize context managers
context_manager = ContextManager(max_history=20)
real_time_updater = RealTimeContextUpdater(context_manager)


def get_user_mood_input():
    mood_options = ["Neutral", "Happy", "Sad", "Angry", "Stressed"]
    print("\nSelect your current mood:")
    for i, mood in enumerate(mood_options):
        print(f"{i + 1}. {mood}")
    while True:
        try:
            choice = int(input(f"Enter the number for your mood (1-{len(mood_options)}): "))
            if 1 <= choice <= len(mood_options):
                return mood_options[choice - 1]
            else:
                print("Invalid choice. Please enter a number within the range.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def get_user_time_input():
    time_options = ["Morning", "Afternoon", "Evening", "Night"]
    print("\nSelect the time of day:")
    for i, time_str in enumerate(time_options):
        print(f"{i + 1}. {time_str}")
    while True:
        try:
            choice = int(input(f"Enter the number for time of day (1-{len(time_options)}): "))
            if 1 <= choice <= len(time_options):
                return time_options[choice - 1]
            else:
                print("Invalid choice. Please enter a number within the range.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def get_user_weather_input():
    weather_options = ["Sunny", "Cloudy", "Rainy", "Snowy", "Stormy", "Foggy", "Windy"]
    print("\nSelect the current weather:")
    for i, weather_str in enumerate(weather_options):
        print(f"{i + 1}. {weather_str}")
    while True:
        try:
            choice = int(input(f"Enter the number for weather (1-{len(weather_options)}): "))
            if 1 <= choice <= len(weather_options):
                return weather_options[choice - 1]
            else:
                print("Invalid choice. Please enter a number within the range.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def process_query(user_input, weather, mood, time_of_day):
    # Update context in the shared ContextManager
    context_manager.add_user_utterance(user_input)
    real_time_updater.update_manual(weather, mood, time_of_day) # Updates context_manager's real_time_data

    # Extract and match concepts
    extracted_concepts = extract_concepts(user_input)
    print("Extracted concepts from query:", extracted_concepts)

    concept_matches = normalize_and_match_concepts(conceptnet_graph, extracted_concepts)
    # Filter out None values and collect matched node names for retrieval
    matched_nodes_for_retrieval = [node for node in concept_matches.values() if node is not None]
    # Collect all matched nodes (even if None) for debug display
    matched_nodes_for_debug = ', '.join([node if node is not None else f"'{orig_c}' (no match)" for orig_c, node in concept_matches.items()])
    
    print("Matched nodes for subgraph retrieval:", matched_nodes_for_retrieval)

    # Retrieve combined subgraph with increased neighborhood k=3
    # Only try to retrieve if there are actual matched nodes
    subgraph = nx.Graph() # Initialize an empty graph
    if matched_nodes_for_retrieval:
        subgraph = retrieve_combined_subgraph(conceptnet_graph, atomic_graph, matched_nodes_for_retrieval, k=3, is_concept_list=True)
    
    print(f"Subgraph retrieved with {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges.")

    gnn_output, node_mapping, explanation_path, gnn_persona_insight = None, None, None, "No specific GNN insights." # Initialize with defaults

    # Only run GNN if the subgraph is meaningful (has nodes and edges for GAT)
    if subgraph.number_of_nodes() > 0:
        try:
            # Pass user_input, mood, time_of_day, and weather_condition to GNN
            gnn_output, node_mapping, explanation_path, gnn_persona_insight = \
                run_gnn_reasoning(subgraph, user_query=user_input, user_mood=mood, time_of_day=time_of_day, weather_condition=weather)
        except Exception as e:
            print(f"Error during GNN reasoning: {e}")
            gnn_persona_insight = f"GNN processing encountered an error: {e}"


    # Prepare current context for LLM prompt
    current_llm_context = real_time_updater.get_context()
    
    # Build prompt for OpenAI
    prompt = build_prompt(user_input, current_llm_context, gnn_insight=gnn_persona_insight)
    
    # Get response from OpenAI
    response = get_model_response(prompt)
    context_manager.add_system_response(response) # Store system response in history

    # Placeholder for actual confidence score (e.g., from a classifier or LLM's own confidence)
    accuracy = round(torch.rand(1).item() * 10 + 90, 2)

    return {
        "response": response,
        "accuracy": accuracy,
        "explanation": explanation_path,
        "raw_gnn_output": gnn_output.tolist() if gnn_output is not None else None, # For debugging/analysis
        "gnn_persona_insight": gnn_persona_insight,
        "weather_context": current_llm_context.get('weather', 'N/A'),
        "mood_context": current_llm_context.get('user_mood', 'N/A'),
        "time_context": current_llm_context.get('time_of_day', 'N/A'),
        "extracted_concepts": ', '.join(extracted_concepts),
        "matched_nodes_for_debug": matched_nodes_for_debug
    }


if __name__ == "__main__":
    print("\n--- Starting Interactive Persona AI Demo ---")
    print("Type 'exit' or 'quit' to end the session.")

    while True:
        user_query_input = input("\nEnter your query: ")
        if user_query_input.lower() in ["exit", "quit"]:
            break

        selected_mood = get_user_mood_input()
        selected_time = get_user_time_input()
        selected_weather = get_user_weather_input() # New: Get weather input

        print(f"\nProcessing query: '{user_query_input}' with Mood: {selected_mood}, Time: {selected_time}, Weather: {selected_weather}")

        output = process_query(user_query_input, selected_weather, selected_mood, selected_time)

        print("\n--- Persona AI Response ---")
        print("Model Response:", output["response"])
        print("\n--- Reasoning Insights ---")
        print("GNN Persona Insight:", output["gnn_persona_insight"])
        print("System Confidence (Accuracy):", output["accuracy"], "%")
        if output["explanation"]:
            print("Graph Attention Visualization saved to:", output["explanation"])
        else:
            print("Graph Attention Visualization: Not available (subgraph too small or error during visualization).")
        print("\n--- Contextual Data Used ---")
        print("Weather Condition:", output["weather_context"])
        print("Time of Day:", output["time_context"])
        print("User Mood:", output["mood_context"])
        print("\n--- Technical Details ---")
        print("Extracted Concepts:", output["extracted_concepts"])
        print("Matched Nodes in KG:", output["matched_nodes_for_debug"])
        # print("Raw GNN Output (for debug):", output["raw_gnn_output"]) # Uncomment for detailed GNN output

    print("\n--- Session Ended ---")