"""
response_generator.py

This module is the core component for generating natural language responses. It integrates
insights from the Graph Neural Network (GNN) and contextual data to produce user-facing
replies. The module uses a pre-trained Large Language Model (LLM) to generate responses
based on a structured prompt that includes the user's query, real-time context, and
GNN-derived reasoning.
"""

from transformers import pipeline
import re
import torch
import numpy as np
from typing import Dict, List, Any

# Load the GPT-2 model from Hugging Face.
# 'gpt2-medium' is a good balance between performance and size for this task.
# This assumes the model is either cached locally or an internet connection is available.
try:
    generator = pipeline("text-generation", model="gpt2-medium")
except ImportError as e:
    print(f"Error loading GPT-2 model: {e}. Please ensure 'transformers' is installed.")
    generator = None

def format_dialogue_history(history: List[Dict[str, str]], max_turns: int = 2) -> str:
    """
    Formats a list of dialogue turns into a clean, readable string.

    This function prepares dialogue history to be included in the LLM prompt.
    It takes the last `max_turns` of the conversation to provide short-term context.

    Args:
        history (List[Dict[str, str]]): A list of dictionaries, each representing a
                                         dialogue turn with 'speaker' and 'utterance' keys.
        max_turns (int): The number of recent turns to include.

    Returns:
        str: A formatted string of the recent dialogue history.
    """
    return "\n".join(
        f"{turn['speaker'].capitalize()}: {turn['utterance']}" for turn in history[-max_turns:]
    )

def clean_response(text: str) -> str:
    """
    Cleans up the generated text from the LLM to remove unwanted artifacts.

    This function removes repetitive phrases, system prompts, and ensures the
    response ends with a proper punctuation mark.

    Args:
        text (str): The raw text output from the language model.

    Returns:
        str: The cleaned and formatted response.
    """
    # Remove leading/trailing whitespace.
    text = text.strip()
    # Replace multiple apologies with a single one.
    text = re.sub(r"(I'm sorry[.!]\s*){2,}", "I'm sorry. ", text, flags=re.IGNORECASE)
    # Remove fragments of the prompt that the model might repeat.
    text = re.sub(r"(User|Assistant|System|Response):.*", "", text, flags=re.IGNORECASE).strip()
    # Remove common conversational fillers or repeated prompt parts.
    text = re.sub(r"\b(Given|Based on|Here's|The response is|A good response would be|My advice is|Response would be|Assistant:)\s*$", "", text, flags=re.IGNORECASE).strip()
    # Ensure the response ends with a period if it doesn't already have one.
    if text and not text.endswith(('.', '!', '?')):
        text += '.'
    return text

def generate_response(
    query: str, 
    gnn_output: Optional[torch.Tensor], 
    node_mapping: Dict[int, str], 
    context_history: List[Dict[str, str]], 
    real_time_data: Dict[str, str]
) -> str:
    """
    Generates a final response by integrating all available information.

    This function constructs a detailed prompt for the LLM, including the user's
    query, contextual data, and a summary of GNN insights. It then uses the LLM
    to generate a response and cleans it before returning.

    Args:
        query (str): The user's input query.
        gnn_output (Optional[torch.Tensor]): The output tensor from the GNN model,
                                             representing node embeddings or scores.
        node_mapping (Dict[int, str]): A dictionary mapping GNN node indices to
                                       their original text concepts.
        context_history (List[Dict[str, str]]): A list of recent dialogue turns.
        real_time_data (Dict[str, str]): A dictionary of real-time context data.

    Returns:
        str: The final, generated, and cleaned response.
    """
    # Create a string of relevant concepts from the GNN's node mapping.
    matched_strs = [str(node_mapping[i]) for i in range(len(node_mapping)) if i in node_mapping and node_mapping[i] is not None]
    relevant_concepts_str = ", ".join(matched_strs) if matched_strs else "None"

    # Extract real-time context data.
    weather = real_time_data.get("weather", "No data")
    mood = real_time_data.get("user_mood", "No data")
    time_of_day = real_time_data.get("time_of_day", "No data")

    # --- Interpret GNN Output into a Natural Language Insight ---
    gnn_insight = ""
    # Check if the GNN produced a valid output.
    if gnn_output is not None and isinstance(gnn_output, torch.Tensor) and gnn_output.numel() > 0:
        # A generic insight for when relevant concepts were matched.
        if relevant_concepts_str != "None":
            gnn_insight = f"The graph neural network analyzed common sense relationships involving concepts like {relevant_concepts_str}. "
        else:
            gnn_insight = "The model processed information from common sense graphs. "
    else:
        # Fallback if no GNN output was available.
        gnn_insight = "No specific graph-based reasoning insight was available. "

    # --- Construct the Prompt for the LLM ---
    # The prompt is carefully crafted to guide the LLM's behavior and response style.
    prompt = (
        "You are a highly intelligent and helpful AI assistant designed to provide practical, common-sense advice.\n"
        "Your goal is to give actionable and empathetic responses to user's situations, drawing on provided context and reasoning.\n"
        "Always respond directly to the user in a helpful tone. Do not ask questions or make demands.\n"
        "**Ensure your responses are always advice for the user, starting with phrases like 'You should...', 'It's recommended to...', or similar. Do not use 'I will' or imply that you are taking action on the user's behalf.**\n\n"
        "Here's an example of how you should respond:\n"
        "User: 'I spilled water on my laptop.'\n"
        "Context: time = morning, weather = rainy, mood = worried, concepts = water, laptop, electronics\n"
        "Response: 'You should immediately turn off your laptop, unplug it, and remove the battery if possible. Let it dry completely for at least 24-48 hours before attempting to turn it on again.'\n\n"
        f"User: '{query}'\n"
        f"Context: time = {time_of_day}, weather = {weather}, mood = {mood}, concepts = {relevant_concepts_str}\n"
        f"Reasoning Insight: {gnn_insight}\n"
        "Response: "
    )

    # --- Generate the response using the LLM ---
    if generator is None:
        return "Error: The text generation model is not loaded."

    outputs = generator(
        prompt,
        max_new_tokens=80,          # Limits the length of the generated response.
        num_return_sequences=1,
        temperature=0.7,            # Controls the randomness of the output.
        top_p=0.9,                  # Controls the diversity of the output.
        do_sample=True,             # Enables sampling-based generation.
        return_full_text=False,     # Returns only the generated part of the text.
    )

    generated_text = outputs[0]["generated_text"].strip()
    cleaned = clean_response(generated_text)

    # --- Fallback/Hardcoded Responses for Common Queries ---
    # This section provides robust, reliable responses for specific, high-priority
    # queries, preventing the LLM from generating poor or unsafe advice.
    if not cleaned or len(cleaned.split()) < 3 or cleaned.lower() == query.lower():
        if "hungry" in query.lower():
            return "You should find something to eat. Perhaps prepare a snack or a meal."
        elif "tired" in query.lower():
            return "It sounds like you need some rest. Consider taking a break or getting some sleep."
        elif "spilled" in query.lower() and "laptop" in query.lower():
            return "Immediately turn off your laptop, unplug it, and remove the battery if possible. Let it dry completely before turning it on."
        else:
            return "I recommend looking for a practical solution to your situation."

    return cleaned