"""
train_baseline.py

Train a simple Feed-Forward Network (FFN) as a baseline model for persona inference.
Now includes early stopping and validation split for more robust training.
"""

import os
import sys
import random
import argparse
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Ensure local imports
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.join(__file__, '..'))))

# local imports
from reasoning.simple_model import SimpleModel
from context.transformer_encoder import TransformerEncoder
from context.context_encoder import ContextEncoder

# --- Default Config ---
DEFAULT_EPOCHS = 50
DEFAULT_BATCH = 32
DEFAULT_LR = 1e-3
HIDDEN_DIM = 64
OUTPUT_DIM = 4  # Urgency, Emotional Distress, Practical Need, Empathy Requirement

# --- Mock Data Generator ---
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

PERSONA_DIMENSIONS = ["Urgency", "Emotional Distress", "Practical Need", "Empathy Requirement"]

def generate_persona_data(transformer_encoder, context_encoder, num_samples=500, mock_graph=False):
    """
    Generate synthetic persona data for training/testing.
    """
    data = []
    for _ in range(num_samples):
        query_embedding = torch.normal(mean=0.0, std=1.0, size=(transformer_encoder.embedding_dim,))
        context_embedding = torch.empty(context_encoder.total_context_dim).uniform_(-1, 1)

        base_score = torch.sigmoid(query_embedding.mean() + context_embedding.mean()).item()
        labels = {
            dim: min(max(base_score + random.uniform(-0.2, 0.2), 0.0), 1.0)
            for dim in PERSONA_DIMENSIONS
        }

        data.append({
            "query_embedding": query_embedding,
            "context_embedding": context_embedding,
            "labels": labels
        })
    return data

# --- Early Stopping ---
class EarlyStopping:
    def __init__(self, patience=5, delta=1e-4):
        self.patience = patience
        self.delta = delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False
        self.best_state = None

    def step(self, val_loss, model):
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

# --- Main Training Loop ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a baseline FFN for persona inference.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate for the optimizer.")
    args = parser.parse_args()

    print("Loading encoders to determine dimensions...")
    transformer_encoder = TransformerEncoder()
    context_encoder = ContextEncoder()
    TRANSFORMER_EMBEDDING_DIM = transformer_encoder.embedding_dim if hasattr(transformer_encoder, 'embedding_dim') else 384
    CONTEXT_DIM = context_encoder.total_context_dim
    TOTAL_INPUT_DIM = TRANSFORMER_EMBEDDING_DIM + CONTEXT_DIM
    print(f" Baseline model input dimension: {TOTAL_INPUT_DIM}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleModel(input_dim=TOTAL_INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    print("Generating mock data...")
    all_data = generate_persona_data(transformer_encoder, context_encoder, num_samples=500)

    X = torch.stack([d['query_embedding'] for d in all_data])
    C = torch.stack([d['context_embedding'] for d in all_data])
    inputs = torch.cat((X, C), dim=1)
    labels = torch.tensor([list(d['labels'].values()) for d in all_data], dtype=torch.float32)

    # --- Train/Validation Split ---
    train_inputs, val_inputs, train_labels, val_labels = train_test_split(
        inputs, labels, test_size=0.2, random_state=42
    )
    train_dataset = TensorDataset(train_inputs, train_labels)
    val_dataset = TensorDataset(val_inputs, val_labels)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # --- Training ---
    early_stopper = EarlyStopping(patience=5)
    print("--- Starting Training with Early Stopping ---")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        for batch_inputs, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_train_loss = epoch_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_inputs, val_labels in val_loader:
                val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                outputs = model(val_inputs)
                loss = criterion(outputs, val_labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        early_stopper.step(avg_val_loss, model)
        if early_stopper.should_stop:
            print(f" Early stopping triggered at epoch {epoch+1}")
            break

    # --- Save Best Model ---
    os.makedirs("models", exist_ok=True)
    save_path = "models/baseline_ffn_model.pth"
    torch.save({
        "model_state_dict": early_stopper.best_state if early_stopper.best_state else model.state_dict(),
        "input_dim": TOTAL_INPUT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "output_dim": OUTPUT_DIM,
        "epochs_trained": epoch+1
    }, save_path)
    print(f"\n Best Baseline FFN model saved to {save_path}")
