"""
simple_model.py

A baseline model for persona inference using a simple Feed-Forward Neural Network.
This model provides a point of comparison to the more complex GNN-based approach,
as it processes concatenated embeddings without leveraging any graph structure.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class SimpleModel(nn.Module):
    """
    A simple Feed-Forward Neural Network for persona inference.

    This model processes a flattened tensor of concatenated query and context embeddings.
    It serves as a straightforward baseline to evaluate the performance of graph-based
    models against.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """
        Initializes the SimpleModel with fully connected layers.

        Args:
            input_dim (int): The total dimension of the input features, which should be
                             the sum of the query and context embedding dimensions.
            hidden_dim (int): The dimension of the hidden layers.
            output_dim (int): The dimension of the final output (e.g., the number of
                              persona classes).
        """
        super(SimpleModel, self).__init__()
        # First fully connected layer with ReLU activation.
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # Second fully connected layer with ReLU activation.
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        # Final fully connected layer to produce the output scores.
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the model.

        The input `x` is expected to be a concatenated tensor of query and context embeddings.

        Args:
            x (torch.Tensor): The input tensor of concatenated features.

        Returns:
            torch.Tensor: The output tensor of scores for each persona class.
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
