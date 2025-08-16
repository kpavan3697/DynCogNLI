"""
context_manager.py

This module defines the ContextManager class, a central component for handling
and maintaining all forms of context within the DynCogNLI system. It is designed
to store and manage both dialogue history and real-time contextual data, such as
mood, time, and weather. This consolidated approach ensures that all relevant
contextual information is readily available for other modules, like the GNN model,
to inform its reasoning and decision-making processes.
"""

from typing import List, Dict, Any, Optional

class ContextManager:
    """
    Manages and stores various forms of context, including dialogue history and
    real-time contextual data.

    This class serves as a central hub for all context-related information, providing
    methods to update, retrieve, and clear the stored data. It maintains a fixed-size
    dialogue history and a dictionary for flexible, real-time data.

    Args:
        max_history (int): The maximum number of dialogue turns to store in the history.
                           Defaults to 10.
    
    Attributes:
        dialogue_history (List[Dict[str, str]]): A list of dictionaries, where each
                                                 dictionary represents a dialogue turn
                                                 with 'speaker' and 'utterance' keys.
        max_history (int): The maximum size of the dialogue history queue.
        real_time_data (Dict[str, Any]): A dictionary for storing key-value pairs
                                         of real-time context, such as mood, time,
                                         and weather.
    """
    def __init__(self, max_history: int = 10):
        """
        Initializes the ContextManager with an empty dialogue history and real-time data store.
        """
        # A list to store the last `max_history` dialogue turns.
        self.dialogue_history: List[Dict[str, str]] = []
        self.max_history: int = max_history
        
        # A dictionary to store arbitrary key-value pairs for real-time context.
        self.real_time_data: Dict[str, Any] = {}

    def add_user_utterance(self, utterance: str):
        """
        Adds a new user utterance to the dialogue history.

        Args:
            utterance (str): The text of the user's utterance.
        """
        self._add_to_history({"speaker": "user", "utterance": utterance})

    def add_system_response(self, response: str):
        """
        Adds a new system response to the dialogue history.

        Args:
            response (str): The text of the system's response.
        """
        self._add_to_history({"speaker": "system", "utterance": response})

    def _add_to_history(self, turn: Dict[str, str]):
        """
        An internal method to add a new turn to the dialogue history.

        This method manages the fixed size of the history list by appending a new turn
        and removing the oldest turn if the list exceeds `max_history`.

        Args:
            turn (Dict[str, str]): A dictionary representing a single turn of dialogue.
        """
        self.dialogue_history.append(turn)
        # Enforce the maximum history size by removing the oldest item.
        if len(self.dialogue_history) > self.max_history:
            self.dialogue_history.pop(0)

    def get_dialogue_history(self) -> List[Dict[str, str]]:
        """
        Retrieves the current dialogue history.

        Returns:
            List[Dict[str, str]]: A list of dialogue turns.
        """
        return self.dialogue_history

    def update_real_time_data(self, key: str, value: Any):
        """
        Updates a specific piece of real-time context data.

        This method is used to store or update dynamic information, such as the
        current mood, time, or weather.

        Args:
            key (str): The key for the data to be updated (e.g., 'mood', 'weather').
            value (Any): The value to be stored.
        """
        self.real_time_data[key] = value

    def get_real_time_data(self, key: str) -> Optional[Any]:
        """
        Retrieves a specific piece of real-time context data.

        Args:
            key (str): The key for the data to be retrieved.

        Returns:
            Optional[Any]: The value associated with the key, or None if the key
                           is not found.
        """
        return self.real_time_data.get(key, None)

    def get_all_context(self) -> Dict[str, Any]:
        """
        Retrieves all stored real-time context data.

        Returns:
            Dict[str, Any]: A dictionary containing all real-time context data.
        """
        return self.real_time_data

    def clear_context(self):
        """
        Clears both the dialogue history and all real-time context data.

        This method is useful for resetting the system's context, for instance,
        at the beginning of a new session.
        """
        self.dialogue_history = []
        self.real_time_data = {}