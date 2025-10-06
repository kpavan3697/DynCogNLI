"""
train_baseline.py

This script is for training a simple Feed-Forward Network (FFN) to act as a
baseline model for persona inference. It processes query and context embeddings
directly, without the graph-based reasoning of the GNN. This allows for a
direct performance comparison to demonstrate the value of the GNN's approach.
"""
import os
import sys
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Ensure the parent directory is in sys.path for local imports
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.join(__file__, '..'))))

# local imports
from reasoning.simple_model import SimpleModel
from context.transformer_encoder import TransformerEncoder
from context.context_encoder import ContextEncoder
from preprocessing.mock_data_generator import generate_persona_data

# --- Default Config ---
DEFAULT_EPOCHS = 20
DEFAULT_BATCH = 8
DEFAULT_LR = 1e-3
HIDDEN_DIM = 64
OUTPUT_DIM = 4  # Represents the 4 persona dimensions

# --- Main Training Loop ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a baseline FFN for persona inference.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate for the optimizer.")
    args = parser.parse_args()

    # Load encoders to get correct dimensions
    print("Loading encoders to determine dimensions...")
    transformer_encoder = TransformerEncoder()
    context_encoder = ContextEncoder()
    
    # Calculate total input dimension for the FFN
    TRANSFORMER_EMBEDDING_DIM = transformer_encoder.embedding_dim if hasattr(transformer_encoder, 'embedding_dim') else 384
    CONTEXT_DIM = context_encoder.total_context_dim
    TOTAL_INPUT_DIM = TRANSFORMER_EMBEDDING_DIM + CONTEXT_DIM
    print(f" Baseline model input dimension: {TOTAL_INPUT_DIM}")
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleModel(input_dim=TOTAL_INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    print(f"Generating mock data for training...")
    all_data = generate_persona_data(
        transformer_encoder,
        context_encoder,
        num_samples=500,
        mock_graph=True
    )
    
    # Combine query embeddings and context embeddings to form the full feature vector
    X = torch.stack([d['query_embedding'] for d in all_data])
    C = torch.stack([d['context_embedding'] for d in all_data])
    inputs = torch.cat((X, C), dim=1).to(device)
    labels = torch.tensor([list(d['labels'].values()) for d in all_data]).to(device)

    # Create a DataLoader for batching
    dataset = TensorDataset(inputs, labels)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    
    print("--- Starting Training ---")
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0
        for batch_inputs, batch_labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            optimizer.zero_grad()
            outputs = model(batch_inputs)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch+1} Loss: {avg_loss:.4f}")

    # --- Evaluation ---
    print("\n--- Evaluating Model ---")
    model.eval()
    with torch.no_grad():
        predictions = model(inputs).cpu().numpy()
        true_labels = labels.cpu().numpy()

    mse = mean_squared_error(true_labels, predictions)
    mae = mean_absolute_error(true_labels, predictions)
    r2 = r2_score(true_labels, predictions)

    print(f"\n Model Evaluation Metrics:")
    print(f"   MSE: {mse:.4f}")
    print(f"   MAE: {mae:.4f}")
    print(f"   R² : {r2:.4f}")

    # After training, save the model
    os.makedirs("models", exist_ok=True)
    save_path = "models/baseline_ffn_model.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": TOTAL_INPUT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "output_dim": OUTPUT_DIM,
        "epochs": args.epochs
    }, save_path)
    print(f"\n Baseline FFN model saved to {save_path}")
