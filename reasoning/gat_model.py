"""
gat_model.py

Defines the GNN architecture (e.g., GATModel) for persona inference.
This module implements the core model that performs reasoning over the
knowledge graph data. The forward pass is defined to process the graph
structure and node features, ultimately producing a graph-level representation
that can be used for downstream tasks, such as inferring a user's persona.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data

class GATModel(nn.Module):
    """
    A Graph Attention Network (GAT) model for processing graph data.

    This model uses multiple GAT layers to learn a representation of a graph.
    The GAT layers apply an attention mechanism to weigh the importance of
    neighboring nodes, which is particularly effective for tasks that require
    understanding relationships within a graph, such as persona inference.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, heads: int = 4):
        """
        Initializes the GATModel with configurable layer dimensions.

        Args:
            input_dim (int): The number of features for each input node.
            hidden_dim (int): The number of features in the hidden layers.
            output_dim (int): The dimension of the final output (e.g., the number of
                              persona classes or a regression value).
            heads (int): The number of attention heads for the GAT layers. Multiple heads
                         can help stabilize the learning process.
        """
        super(GATModel, self).__init__()
        # First GATConv layer. It takes the raw node features and learns
        # an attention-weighted representation. Dropout is applied for regularization.
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=0.6)
        
        # Second GATConv layer. It operates on the output of the first layer,
        # which has a dimension of `hidden_dim * heads`.
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=0.6)
        
        # A final linear layer to project the learned node representations
        # to the desired output dimension (e.g., a score for each persona type).
        self.node_predictor = nn.Linear(hidden_dim * heads, output_dim)

    def forward(self, data: Data):
        """
        Defines the forward pass of the GNN model.

        The forward pass consists of two GATConv layers with ReLU activation
        and dropout, followed by a linear layer to get node-level scores.
        Finally, a global mean pooling operation is applied to aggregate these
        node scores into a single graph-level score.

        Args:
            data (Data): A PyG `Data` object containing `x` (node features),
                         `edge_index` (graph connectivity), and `batch` (batch assignments).

        Returns:
            torch.Tensor: A tensor representing the aggregated graph-level score.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Apply the first GATConv layer with ReLU and dropout.
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        
        # Apply the second GATConv layer with ReLU and dropout.
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        
        # Use the linear layer to get the final score for each node.
        node_scores = self.node_predictor(x)
        
        # Perform global mean pooling to aggregate the node scores into
        # a single score for the entire graph.
        graph_score = global_mean_pool(node_scores, batch)
        
        return graph_score
