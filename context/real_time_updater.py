"""
real_time_updater.py

This module provides the core logic for dynamically updating context variables
based on user input and system data. It defines the `RealTimeContextUpdater` class,
which is responsible for extracting context from a user's query or manual input,
and a `build_prompt` function that formats this context for a large language model.

This file is a key part of the dynamic contextual system, enabling real-time
adaptation for both inference and training of the GNN model. It also includes
a placeholder `get_model_response` function for development and testing without
an external API key.
"""

import os
from datetime import datetime
from typing import Dict, Any, Optional

# NOTE: The following ContextManager is a placeholder for demonstration purposes
# within this file. The canonical ContextManager is defined in context_manager.py.
# This placeholder class is for testing `real_time_updater.py` in isolation.
class ContextManager:
    """
    A placeholder class for ContextManager, used for local testing within this file.
    """
    def __init__(self):
        self.context: Dict[str, Any] = {
            "weather": "unknown",
            "user_mood": "neutral",
            "time_of_day": "unknown"
        }
        self.history: list = []

    def add_user_utterance(self, utterance: str):
        self.history.append({"role": "user", "content": utterance, "timestamp": datetime.now().isoformat()})

    def add_system_response(self, response: str):
        self.history.append({"role": "system", "content": response, "timestamp": datetime.now().isoformat()})

    def update_real_time_data(self, key: str, value: Any):
        self.context[key] = value

    def get_all_context(self) -> Dict[str, Any]:
        return self.context

    def get_history(self, num_entries: int = 5) -> list:
        return self.history[-num_entries:]


class RealTimeContextUpdater:
    """
    A class to update a `ContextManager` instance with real-time data.

    This class supports two main methods for updating context: one that
    heuristically extracts context from a text query, and another for manual
    updates, typically from a user interface.

    Attributes:
        context_manager (ContextManager): An instance of the central ContextManager
                                          class to be updated.
    """
    def __init__(self, context_manager_instance: 'ContextManager'):
        """
        Initializes the updater with a reference to the main context manager.
        """
        self.context_manager = context_manager_instance

    def update_from_query(self, query: str):
        """
        Extracts and updates context data based on keywords in a user query.

        This method performs basic keyword-based extraction for weather and mood.
        Time of day is determined programmatically based on the current system time.

        Args:
            query (str): The user's input query string.
        """
        query_lower = query.lower()

        # Simple keyword-based weather detection
        weather = "rainy" if "rain" in query_lower else "sunny"

        # Simple keyword-based mood detection
        if any(w in query_lower for w in ["tired", "sleepy", "exhausted", "fatigued"]):
            mood = "tired"
        elif any(w in query_lower for w in ["happy", "joyful", "excited"]):
            mood = "happy"
        elif any(w in query_lower for w in ["sad", "unhappy", "depressed"]):
            mood = "sad"
        else:
            mood = "neutral"

        # Determine time of day based on current system time
        hour = datetime.now().hour
        if 5 <= hour < 12:
            time_of_day = "morning"
        elif 12 <= hour < 17:
            time_of_day = "afternoon"
        else:
            time_of_day = "evening"

        # Update the central context manager with the extracted data.
        self.context_manager.update_real_time_data("weather", weather)
        self.context_manager.update_real_time_data("user_mood", mood)
        self.context_manager.update_real_time_data("time_of_day", time_of_day)

    def update_manual(self, weather: str, mood: str, time_of_day: str):
        """
        Manually updates context data, typically from a UI form.

        This method provides an explicit way to set context variables, with
        handling for 'No data' or empty string values.

        Args:
            weather (str): The weather condition string.
            mood (str): The user's mood string.
            time_of_day (str): The time of day string.
        """
        # Update weather if a meaningful value is provided.
        if weather and weather.lower() != "no data":
            self.context_manager.update_real_time_data("weather", weather)
        else:
            self.context_manager.update_real_time_data("weather", "unknown")

        # Update mood if a meaningful value is provided.
        if mood and mood.lower() != "no data":
            self.context_manager.update_real_time_data("user_mood", mood)
        else:
            self.context_manager.update_real_time_data("user_mood", "neutral")

        # Update time of day. This is assumed to always have a valid value.
        if time_of_day:
            self.context_manager.update_real_time_data("time_of_day", time_of_day)

    def get_context(self) -> Dict[str, Any]:
        """
        Retrieves all current context data from the managed context manager.

        Returns:
            Dict[str, Any]: A dictionary containing all real-time context data.
        """
        return self.context_manager.get_all_context()


def build_prompt(user_query: str, context: Dict[str, Any], gnn_insight: str = "") -> str:
    """
    Constructs a detailed prompt for a large language model.

    This function combines a static system prompt with dynamic context variables
    and an optional GNN insight to guide the model's response. The prompt is
    designed to elicit actionable, empathetic, and common-sense advice.

    Args:
        user_query (str): The user's query or problem statement.
        context (Dict[str, Any]): A dictionary of context variables (e.g., weather, mood).
        gnn_insight (str): An optional insight or reasoning from the GNN model.

    Returns:
        str: The final, formatted prompt string.
    """
    prompt_parts = [
        "You are a highly intelligent and helpful AI assistant designed to provide practical, common-sense advice.",
        "Your goal is to give actionable and empathetic responses to user's situations, drawing on provided context and reasoning.",
        "Always respond directly to the user in a helpful tone. Do not ask questions or make demands.",
        "Ensure your responses are always advice for the user, starting with phrases like 'You should...', 'It's recommended to...', or similar. Do not use 'I will' or imply that you are taking action on the user's behalf.",
        "Based on the following dynamic context and reasoning insights, provide the best common-sense advice:"
    ]
    # Append dynamic context to the prompt.
    prompt_parts.append(f"- Weather: {context.get('weather', 'unknown')}")
    prompt_parts.append(f"- User Mood: {context.get('user_mood', 'neutral')}")
    prompt_parts.append(f"- Time of Day: {context.get('time_of_day', 'unknown')}")

    # Add GNN insight only if it's provided.
    if gnn_insight:
        prompt_parts.append(f"- Knowledge Graph Insight: {gnn_insight}")
    
    # Conclude the prompt with the user's query and a clear instruction.
    prompt_parts.append(f"\nThe user said: \"{user_query}\"")
    prompt_parts.append("Respond with common-sense advice:")

    return "\n".join(prompt_parts)


def get_model_response(prompt: str) -> str:
    """
    Returns a static placeholder message instead of calling a real API.

    This function simulates a model's response by parsing the prompt and
    constructing a generic, informative message. This is useful for testing
    the system's data flow without requiring an API key.

    Args:
        prompt (str): The constructed prompt string.

    Returns:
        str: A formatted placeholder response string.
    """
    print("WARNING: OpenAI API key is not set, returning placeholder response.")
    
    # Initialize variables with default values to handle parsing errors.
    user_query_extracted = "your situation"
    weather_extracted = "unknown"
    mood_extracted = "neutral"
    time_extracted = "unknown"
    gnn_insight_extracted = "No specific knowledge graph insight available for this query."

    try:
        # Helper function to safely extract values from the prompt string.
        def extract_value_from_prompt(label: str, default: str = "N/A") -> str:
            start_idx = prompt.find(label)
            if start_idx != -1:
                value_start = start_idx + len(label)
                # Find the end of the line or the next context label.
                end_idx = prompt.find('\n', value_start)
                if end_idx == -1:
                    end_idx = len(prompt)
                
                value = prompt[value_start:end_idx].strip()
                # Return default for generic or placeholder values.
                if value.lower() in ["no data", "unknown", "neutral"]:
                    return default
                return value
            return default

        # Extract the user's query from the prompt string.
        user_query_start_marker = 'The user said: "'
        if user_query_start_marker in prompt:
            query_start = prompt.find(user_query_start_marker) + len(user_query_start_marker)
            query_end = prompt.find('"', query_start)
            if query_start != -1 and query_end != -1:
                user_query_extracted = prompt[query_start:query_end]
        
        # Extract each context variable.
        weather_extracted = extract_value_from_prompt("- Weather: ")
        mood_extracted = extract_value_from_prompt("- User Mood: ")
        time_extracted = extract_value_from_prompt("- Time of Day: ")
        
        # Extract the GNN insight.
        gnn_insight_marker = "- Knowledge Graph Insight: "
        if gnn_insight_marker in prompt:
            insight_start = prompt.find(gnn_insight_marker) + len(gnn_insight_marker)
            insight_end = prompt.find('\n', insight_start)
            if insight_end == -1:
                insight_end = len(prompt)
            
            gnn_insight_extracted = prompt[insight_start:insight_end].strip()
            if not gnn_insight_extracted:
                gnn_insight_extracted = "No specific knowledge graph insight available for this query."

    except Exception as e:
        print(f"Warning: Could not fully parse prompt for placeholder response due to error: {e}. Some details might be missing.")

    # Construct the final placeholder response string.
    placeholder_response = (
        "Based on your input and the common sense knowledge the system processed: "
        f"\n- **Your Query**: \"{user_query_extracted}\""
        f"\n- **Context**: Weather - {weather_extracted}, User Mood - {mood_extracted}, Time of Day - {time_extracted}"
        f"\n- **Knowledge Graph Insight**: {gnn_insight_extracted}"
        "\n\n**Advice**: To address this situation, it's generally helpful to gather more information and consider all relevant factors. Depending on the specifics, you might want to look into common solutions for similar issues. Taking your current context (weather, mood, and time of day) into account can also help you make a more effective plan."
    )
    return placeholder_response


def main():
    """
    Main function to demonstrate and test the RealTimeContextUpdater in isolation.

    This function simulates a user interaction, updates context based on the query,
    builds a prompt, and generates a placeholder response to show the full workflow
    of this module.
    """
    user_query = "I spilled water on my laptop and I'm stressed because it's raining outside."
    
    # Use the placeholder ContextManager for local testing.
    context_manager_test = ContextManager()
    context_updater = RealTimeContextUpdater(context_manager_test)

    # Simulate updating context from a user query.
    context_updater.update_from_query(user_query)
    current_context = context_updater.get_context()
    
    # Build a prompt with the updated context.
    prompt = build_prompt(user_query, current_context, gnn_insight="Laptops are sensitive to liquids. Water can cause short circuits.")
    
    # Get a placeholder response.
    response = get_model_response(prompt)

    print("Context:", current_context)
    print("\nPrompt Sent:\n", prompt)
    print("\nModel Response:\n", response)


if __name__ == "__main__":
    main()