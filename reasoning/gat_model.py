# reasoning/gnn_model.py
"""
gnn_model.py

Defines the GNN architecture (e.g., GATModel) for persona inference.
Implements forward pass and model layers for reasoning over graph data.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class GATModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, heads=1):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=0.6)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=0.6)
        self.node_predictor = nn.Linear(hidden_dim * heads, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        node_scores = self.node_predictor(x)
        graph_score = global_mean_pool(node_scores, batch)
        return graph_score
