# context/context_encoder.py
"""
context_encoder.py

Implements the ContextEncoder class for encoding mood, time, and weather into context embeddings.
Provides context features for the GNN model.
"""
import torch

class ContextEncoder:
    """
    Encodes categorical context variables (mood, time_of_day, weather_condition)
    into a numerical feature vector.
    """
    def __init__(self):
        # Define mappings for each categorical variable
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

        # Calculate feature dimension for each category (max index + 1)
        self.mood_dim = max(self.mood_map.values()) + 1
        self.time_of_day_dim = max(self.time_of_day_map.values()) + 1
        self.weather_condition_dim = max(self.weather_condition_map.values()) + 1

        # Total dimension for context features
        self.total_context_dim = self.mood_dim + self.time_of_day_dim + self.weather_condition_dim
        print(f"ContextEncoder initialized. Total context feature dimension: {self.total_context_dim}")

    def encode(self, mood: str, time_of_day: str, weather_condition: str) -> torch.Tensor:
        """
        Encodes the given context strings into a combined one-hot tensor.

        Returns:
            torch.Tensor: A 1D tensor representing the concatenated one-hot
                          encoding of all context variables.
        """
        # Ensure inputs are capitalized for consistent lookup
        mood = mood.capitalize()
        time_of_day = time_of_day.capitalize()
        weather_condition = weather_condition.capitalize()

        # One-hot encode mood
        mood_one_hot = torch.zeros(self.mood_dim)
        mood_idx = self.mood_map.get(mood, self.mood_map["Neutral"]) # Default to Neutral
        if mood_idx < self.mood_dim:
            mood_one_hot[mood_idx] = 1.0

        # One-hot encode time of day
        time_one_hot = torch.zeros(self.time_of_day_dim)
        time_idx = self.time_of_day_map.get(time_of_day, self.time_of_day_map["Day"]) # Default to Day
        if time_idx < self.time_of_day_dim:
            time_one_hot[time_idx] = 1.0

        # One-hot encode weather condition
        weather_one_hot = torch.zeros(self.weather_condition_dim)
        weather_idx = self.weather_condition_map.get(weather_condition, self.weather_condition_map["Clear"]) # Default to Clear
        if weather_idx < self.weather_condition_dim:
            weather_one_hot[weather_idx] = 1.0

        # Concatenate all one-hot vectors
        context_features = torch.cat([mood_one_hot, time_one_hot, weather_one_hot])
        return context_features
