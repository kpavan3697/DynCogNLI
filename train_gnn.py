"""
train_gnn.py

Trains the GNN model for persona inference using synthetic data and context features.
Handles data generation, model training, and saving the trained model for downstream reasoning.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
import networkx as nx
import os
import random
from tqdm import tqdm
import sys

# Add parent directory to sys.path to enable imports from 'reasoning' and 'context'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from reasoning.gnn_model import GATModel
from reasoning.graph_builder import fetch_conceptnet_relations, nx_to_pyg_data
from context.transformer_encoder import TransformerEncoder
from context.context_encoder import ContextEncoder

# --- Configuration ---
NUM_EPOCHS = 50
BATCH_SIZE = 16
LEARNING_RATE = 0.001
HIDDEN_DIM = 64 # Hidden dimension for GNN layers
OUTPUT_DIM = 4 # Urgency, Emotional Distress, Practical Need, Empathy Requirement

# Initialize encoders to get their dimensions
transformer_encoder = TransformerEncoder()
context_encoder = ContextEncoder()

TRANSFORMER_EMBEDDING_DIM = transformer_encoder.embedding_dim # typically 384 for MiniLM
CONTEXT_EMBEDDING_DIM = context_encoder.total_context_dim # e.g., 8+5+6 = 19 for our current setup
BASE_NODE_FEATURE_DIM = 30 # Placeholder for any other inherent node features (e.g., one-hot for node type)

# TOTAL_INPUT_DIM must be the sum of all feature types
TOTAL_INPUT_DIM = TRANSFORMER_EMBEDDING_DIM + CONTEXT_EMBEDDING_DIM + BASE_NODE_FEATURE_DIM
print(f"Calculated TOTAL_INPUT_DIM for GNN: {TOTAL_INPUT_DIM} "
      f"(Transformer: {TRANSFORMER_EMBEDDING_DIM}, Context: {CONTEXT_EMBEDDING_DIM}, Base Node: {BASE_NODE_FEATURE_DIM})")

MODEL_SAVE_PATH = "models/persona_gnn_model.pth"

# --- Device Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on device: {device}")

# --- Model Initialization ---
model = GATModel(input_dim=TOTAL_INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM).to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss() # Using MSE for regression tasks (predicting scores)

print(f"GATModel initialized: input_dim={TOTAL_INPUT_DIM}, hidden_dim={HIDDEN_DIM}, output_dim={OUTPUT_DIM}")

# --- Mock Data Generation ---
def generate_mock_data(num_samples=100):
    print("Generating mock training data...")
    mock_queries = [
        "My laptop screen cracked.",
        "I lost my pet cat.",
        "My car broke down on the highway.",
        "I successfully finished my project!",
        "My internet is not working.",
        "I received a surprise gift.",
        "My flight was delayed due to bad weather.",
        "I'm feeling a bit under the weather.",
        "Just got a promotion at work!",
        "Stuck in a massive traffic jam.",
        "Celebrating my birthday today!",
        "My pipes burst and flooded the basement."
    ]

    mock_moods = ["Neutral", "Happy", "Stressed", "Sad", "Angry", "Excited", "Anxious", "Frustrated"]
    mock_times = ["Day", "Night", "Morning", "Afternoon", "Evening"]
    mock_weathers = ["Clear", "Rainy", "Cloudy", "Snowy", "Windy", "Stormy"]

    persona_labels = {
        "My laptop screen cracked.": {"Urgency": 0.8, "Emotional Distress": 0.6, "Practical Need": 0.9, "Empathy Requirement": 0.5},
        "I lost my pet cat.": {"Urgency": 0.9, "Emotional Distress": 0.9, "Practical Need": 0.2, "Empathy Requirement": 0.9},
        "My car broke down on the highway.": {"Urgency": 0.9, "Emotional Distress": 0.7, "Practical Need": 0.95, "Empathy Requirement": 0.4},
        "I successfully finished my project!": {"Urgency": 0.1, "Emotional Distress": 0.1, "Practical Need": 0.1, "Empathy Requirement": 0.2},
        "My internet is not working.": {"Urgency": 0.7, "Emotional Distress": 0.4, "Practical Need": 0.8, "Empathy Requirement": 0.3},
        "I received a surprise gift.": {"Urgency": 0.05, "Emotional Distress": 0.1, "Practical Need": 0.05, "Empathy Requirement": 0.1},
        "My flight was delayed due to bad weather.": {"Urgency": 0.75, "Emotional Distress": 0.65, "Practical Need": 0.7, "Empathy Requirement": 0.6},
        "I'm feeling a bit under the weather.": {"Urgency": 0.3, "Emotional Distress": 0.5, "Practical Need": 0.2, "Empathy Requirement": 0.8},
        "Just got a promotion at work!": {"Urgency": 0.05, "Emotional Distress": 0.05, "Practical Need": 0.05, "Empathy Requirement": 0.1},
        "Stuck in a massive traffic jam.": {"Urgency": 0.6, "Emotional Distress": 0.7, "Practical Need": 0.3, "Empathy Requirement": 0.5},
        "Celebrating my birthday today!": {"Urgency": 0.01, "Emotional Distress": 0.05, "Practical Need": 0.01, "Empathy Requirement": 0.1},
        "My pipes burst and flooded the basement.": {"Urgency": 0.95, "Emotional Distress": 0.8, "Practical Need": 0.99, "Empathy Requirement": 0.7}
    }

    def adjust_label_for_context(label, mood, time, weather):
        adjusted_label = label.copy()
        if mood.lower() in ["stressed", "anxious", "frustrated"]:
            adjusted_label["Empathy Requirement"] = min(1.0, adjusted_label["Empathy Requirement"] + 0.2)
            adjusted_label["Emotional Distress"] = min(1.0, adjusted_label["Emotional Distress"] + 0.1)
        elif mood.lower() in ["happy", "excited"]:
            adjusted_label["Emotional Distress"] = max(0.0, adjusted_label["Emotional Distress"] - 0.2)
            adjusted_label["Empathy Requirement"] = max(0.0, adjusted_label["Empathy Requirement"] - 0.1)

        if time.lower() == "night":
            adjusted_label["Urgency"] = min(1.0, adjusted_label["Urgency"] + 0.1) if adjusted_label["Urgency"] > 0.5 else adjusted_label["Urgency"]
            adjusted_label["Empathy Requirement"] = min(1.0, adjusted_label["Empathy Requirement"] + 0.05)
        
        if weather.lower() in ["stormy", "rainy"]:
            adjusted_label["Practical Need"] = min(1.0, adjusted_label["Practical Need"] + 0.1) if adjusted_label["Practical Need"] > 0.5 else adjusted_label["Practical Need"]
            adjusted_label["Emotional Distress"] = min(1.0, adjusted_label["Emotional Distress"] + 0.05)
        
        return adjusted_label

    dataset = []
    failed_graphs = 0
    for _ in tqdm(range(num_samples), desc="Generating Mock Graphs"):
        query_text = random.choice(mock_queries)
        mock_mood = random.choice(mock_moods)
        mock_time = random.choice(mock_times)
        mock_weather = random.choice(mock_weathers)

        mock_nx_graph = fetch_conceptnet_relations(query_text.split()[0], depth=1, max_nodes=10, max_edges=10)

        if not mock_nx_graph or mock_nx_graph.number_of_nodes() == 0:
            failed_graphs += 1
            continue

        query_embedding_tensor = transformer_encoder.encode(query_text).squeeze(0)
        context_embedding_tensor = context_encoder.encode(mock_mood, mock_time, mock_weather)

        pyg_data, _ = nx_to_pyg_data(
            mock_nx_graph,
            feature_dim=TOTAL_INPUT_DIM,
            query_embedding=query_embedding_tensor,
            context_embedding=context_embedding_tensor
        )

        if pyg_data is None or pyg_data.x is None or pyg_data.x.numel() == 0:
            failed_graphs += 1
            continue

        true_labels = adjust_label_for_context(persona_labels[query_text], mock_mood, mock_time, mock_weather)
        pyg_data.y = torch.tensor([
            true_labels["Urgency"],
            true_labels["Emotional Distress"],
            true_labels["Practical Need"],
            true_labels["Empathy Requirement"]
        ], dtype=torch.float)

        dataset.append(pyg_data)

    print(f"Generated {len(dataset)} valid mock samples. {failed_graphs} samples skipped due to empty graphs or API issues.")
    return dataset

def train():
    model.train()
    total_loss = 0

    dataset = generate_mock_data(num_samples=100)
    if not dataset:
        print("No valid data generated for training epoch. Skipping training for this epoch.")
        return 0.0

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    pbar = tqdm(train_loader, desc="Training Batch")
    for data in pbar:
        data = data.to(device)
        optimizer.zero_grad()

        out = model(data)
        predicted_scores = torch.sigmoid(out)

        current_batch_size = predicted_scores.shape[0]
        try:
            target_labels = data.y.view(current_batch_size, OUTPUT_DIM)
        except RuntimeError as e:
            print(f"CRITICAL ERROR in train(): Failed to reshape data.y.")
            raise e 

        loss = criterion(predicted_scores, target_labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    return total_loss / len(train_loader)

def save_model(model, path, input_dim, hidden_dim, output_dim):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'output_dim': output_dim,
    }, path)
    print(f"Model saved to {path}")

if __name__ == "__main__":
    print(f"Starting training for {NUM_EPOCHS} epochs...")
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_pbar = tqdm(total=1, desc=f"Epoch {epoch:03d}", leave=False)
        train_loss = train()
        epoch_pbar.set_postfix(loss=f"{train_loss:.4f}")
        epoch_pbar.close()
        print(f"Epoch {epoch:03d}, Loss: {train_loss:.4f}")

    save_model(model, MODEL_SAVE_PATH, TOTAL_INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    print("Training finished.")

    # --- Persona Score Comparison: With and Without Context ---
    query_text = "My laptop screen cracked."
    real_mood, real_time, real_weather = "Stressed", "Night", "Rainy"
    query_embedding = transformer_encoder.encode(query_text).squeeze(0)
    context_embedding_real = context_encoder.encode(real_mood, real_time, real_weather)
    context_embedding_neutral = context_encoder.encode("Neutral", "Day", "Clear")

    mock_nx_graph = fetch_conceptnet_relations(query_text.split()[0], depth=1, max_nodes=10, max_edges=10)

    pyg_data_with_context, _ = nx_to_pyg_data(
        mock_nx_graph,
        feature_dim=TOTAL_INPUT_DIM,
        query_embedding=query_embedding,
        context_embedding=context_embedding_real
    )

    pyg_data_without_context, _ = nx_to_pyg_data(
        mock_nx_graph,
        feature_dim=TOTAL_INPUT_DIM,
        query_embedding=query_embedding,
        context_embedding=context_embedding_neutral
    )

    model.eval()
    with torch.no_grad():
        out_with_context = torch.sigmoid(model(pyg_data_with_context.to(device)))
        out_without_context = torch.sigmoid(model(pyg_data_without_context.to(device)))

    print("Raw Persona Scores WITH context:", out_with_context.cpu().numpy())
    print("Raw Persona Scores WITHOUT context:", out_without_context.cpu().numpy())