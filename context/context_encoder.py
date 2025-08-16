"""
context_encoder.py

This module implements the ContextEncoder class, which is responsible for converting 
categorical context variables (e.g., mood, time, weather) into a numerical,
machine-readable format. This encoder is crucial for providing contextual features
to the Graph Neural Network (GNN) model, allowing it to incorporate environmental
and emotional states into its reasoning process.
"""

import torch

class ContextEncoder:
    """
    Encodes categorical context variables (mood, time_of_day, weather_condition)
    into a concatenated one-hot feature vector.

    The encoded vector can be used as node or edge features in a GNN to provide
    contextual information, influencing the model's predictions based on the
    user's state and environment.

    Attributes:
        mood_map (dict): A mapping from mood strings to integer indices.
        time_of_day_map (dict): A mapping from time of day strings to integer indices.
        weather_condition_map (dict): A mapping from weather strings to integer indices.
        total_context_dim (int): The total dimension of the final concatenated one-hot vector.
    """
    def __init__(self):
        """
        Initializes the ContextEncoder with predefined mappings for each context category.
        """
        # Define mappings for each categorical variable.
        # These mappings translate human-readable strings into numerical indices.
        self.mood_map = {
            "Neutral": 0, "Happy": 1, "Stressed": 2, "Sad": 3,
            "Angry": 4, "Excited": 5, "Anxious": 6, "Frustrated": 7
        }
        self.time_of_day_map = {
            "Day": 0, "Night": 1, "Morning": 2, "Afternoon": 3, "Evening": 4
        }
        self.weather_condition_map = {
            "Clear": 0, "Rainy": 1, "Cloudy": 2, "Snowy": 3,
            "Windy": 4, "Stormy": 5
        }

        # Calculate the dimension for each one-hot vector.
        # This is determined by the number of unique categories in each map.
        self.mood_dim = len(self.mood_map)
        self.time_of_day_dim = len(self.time_of_day_map)
        self.weather_condition_dim = len(self.weather_condition_map)

        # The total context dimension is the sum of the dimensions of all one-hot vectors.
        self.total_context_dim = self.mood_dim + self.time_of_day_dim + self.weather_condition_dim
        print(f"ContextEncoder initialized. Total context feature dimension: {self.total_context_dim}")

    def encode(self, mood: str, time_of_day: str, weather_condition: str) -> torch.Tensor:
        """
        Encodes the given context strings into a combined one-hot tensor.

        The method takes a string for each context variable, converts it to its
        corresponding one-hot vector, and then concatenates these vectors to form
        a single context feature tensor.

        Args:
            mood (str): The current mood, e.g., "Happy", "Stressed".
            time_of_day (str): The time of day, e.g., "Morning", "Night".
            weather_condition (str): The current weather, e.g., "Clear", "Rainy".

        Returns:
            torch.Tensor: A 1D tensor representing the concatenated one-hot
                          encoding of all context variables.
        """
        # Standardize input strings by capitalizing the first letter.
        # This ensures robustness against minor case differences in input.
        mood = mood.capitalize()
        time_of_day = time_of_day.capitalize()
        weather_condition = weather_condition.capitalize()

        # One-hot encode mood.
        # We create a zero vector and set the index corresponding to the mood to 1.
        # A default value ("Neutral") is used if the input mood is not recognized.
        mood_one_hot = torch.zeros(self.mood_dim)
        mood_idx = self.mood_map.get(mood, self.mood_map["Neutral"])
        mood_one_hot[mood_idx] = 1.0

        # One-hot encode time of day.
        # Similar to mood, a default value ("Day") is used for unknown inputs.
        time_one_hot = torch.zeros(self.time_of_day_dim)
        time_idx = self.time_of_day_map.get(time_of_day, self.time_of_day_map["Day"])
        time_one_hot[time_idx] = 1.0

        # One-hot encode weather condition.
        # A default value ("Clear") is used for unknown weather inputs.
        weather_one_hot = torch.zeros(self.weather_condition_dim)
        weather_idx = self.weather_condition_map.get(weather_condition, self.weather_condition_map["Clear"])
        weather_one_hot[weather_idx] = 1.0

        # Concatenate all one-hot vectors into a single feature vector.
        # This final vector represents the full context for the model.
        context_features = torch.cat([mood_one_hot, time_one_hot, weather_one_hot])
        return context_features