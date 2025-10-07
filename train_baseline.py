"""
train_baseline.py

Baseline Feed-Forward Network (FFN) for persona inference.
It compares performance without graph reasoning (vs. GAT model).
"""

import os
import sys
import argparse
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Ensure parent directory in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.join(__file__, '..'))))

# Local imports
from reasoning.simple_model import SimpleModel
from context.transformer_encoder import TransformerEncoder
from context.context_encoder import ContextEncoder
from preprocessing.mock_data_generator import generate_persona_data

# ---- Config ----
DEFAULT_EPOCHS = 50
DEFAULT_BATCH = 16
DEFAULT_LR = 1e-3
HIDDEN_DIM = 64
OUTPUT_DIM = 4  # urgency, empathy, proactiveness, emotional
VAL_SPLIT = 0.1
SEED = 42


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model, dataloader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out = torch.sigmoid(model(x))
            preds.append(out.cpu().numpy())
            targets.append(y.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    mse_all = [mean_squared_error(targets[:, i], preds[:, i]) for i in range(targets.shape[1])]
    mae_all = [mean_absolute_error(targets[:, i], preds[:, i]) for i in range(targets.shape[1])]
    r2_all = [r2_score(targets[:, i], preds[:, i]) for i in range(targets.shape[1])]

    return np.mean(mse_all), np.mean(mae_all), np.mean(r2_all)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a baseline FFN for persona inference.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    args = parser.parse_args()

    set_seed(SEED)

    # ---- Load encoders ----
    print("Loading encoders...")
    transformer_encoder = TransformerEncoder()
    context_encoder = ContextEncoder()

    TRANSFORMER_EMBEDDING_DIM = getattr(transformer_encoder, "embedding_dim", 384)
    CONTEXT_DIM = context_encoder.total_context_dim
    TOTAL_INPUT_DIM = TRANSFORMER_EMBEDDING_DIM + CONTEXT_DIM
    print(f"[baseline] Total input dim: {TOTAL_INPUT_DIM}")

    # ---- Initialize model ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleModel(input_dim=TOTAL_INPUT_DIM, hidden_dim=HIDDEN_DIM, output_dim=OUTPUT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # ---- Generate mock data ----
    print("Generating mock data...")
    all_data = generate_persona_data(transformer_encoder, context_encoder, num_samples=500)

    X = torch.stack([d["query_embedding"] for d in all_data])
    C = torch.stack([d["context_embedding"] for d in all_data])
    inputs = torch.cat((X, C), dim=1)
    labels = torch.tensor([list(d["labels"].values()) for d in all_data], dtype=torch.float)

    # ---- Split train/val ----
    val_size = int(len(inputs) * VAL_SPLIT)
    train_size = len(inputs) - val_size
    train_ds, val_ds = random_split(TensorDataset(inputs, labels), [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # ---- Training ----
    print("\n--- Training Started ---")
    best_val_loss = float("inf")
    patience = 5
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        for x_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = torch.sigmoid(model(x_batch))
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        val_loss, val_mae, val_r2 = evaluate(model, val_loader, device)
        print(f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val MSE: {val_loss:.4f} | Val MAE: {val_mae:.4f} | Val R²: {val_r2:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "models/baseline_best.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # ---- Final Evaluation ----
    print("\n--- Final Evaluation on Validation Set ---")
    model.load_state_dict(torch.load("models/baseline_best.pth"))
    mse, mae, r2 = evaluate(model, val_loader, device)
    print(f"Final MSE: {mse:.4f}")
    print(f"Final MAE: {mae:.4f}")
    print(f"Final R² : {r2:.4f}")

    # Save final model
    os.makedirs("models", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": TOTAL_INPUT_DIM,
        "hidden_dim": HIDDEN_DIM,
        "output_dim": OUTPUT_DIM
    }, "models/baseline_ffn_model.pth")

    print("\nBaseline FFN model saved to models/baseline_ffn_model.pth")
