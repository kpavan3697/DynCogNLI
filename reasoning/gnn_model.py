# reasoning/gnn_model.py
"""
gnn_model.py

Defines the GNN architecture (e.g., GATModel) for persona inference.
Implements forward pass and model layers for reasoning over graph data.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool

class GATModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=1):
        super(GATModel, self).__init__()
        self.input_dim = input_dim # Store for saving/loading
        self.hidden_dim = hidden_dim # Store for saving/loading
        self.output_dim = output_dim # Store for saving/loading
        self.heads = heads # Store for saving/loading

        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=0.6)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=0.6) # output of conv1 has hidden_dim * heads
        
        # Linear layer for node-level predictions
        self.node_predictor = nn.Linear(hidden_dim * heads, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch # Batch vector for graph pooling (PyG automatically creates this for Batches)

        x = torch.relu(self.conv1(x, edge_index))
        x = torch.dropout(x, p=0.6, train=self.training)
        x = torch.relu(self.conv2(x, edge_index))
        x = torch.dropout(x, p=0.6, train=self.training)

        # Apply node predictor to get node-level scores
        node_scores = self.node_predictor(x)

        # Global pooling to get a single graph-level prediction
        # For persona scores, we want a single score per dimension for the whole graph.
        # Here we use global_mean_pool on the node_scores (after the predictor).
        # Alternatively, you could pool before the predictor if you wanted a graph embedding
        # and then a single linear layer for the final 4 dimensions.
        graph_score = global_mean_pool(node_scores, batch)

        return graph_score # This is the 4-dimensional output for the graph