"""
inspect_checkpoint.py

This utility script is designed to help users examine the contents of a PyTorch
model checkpoint file (.pth). It's useful for debugging and verification,
allowing you to see the top-level keys, the number of parameters in the
state dictionary, and the names and shapes of each parameter. It also
checks for and displays any stored metadata, such as model dimensions
or training epoch.
"""
import sys
import os
import torch
from typing import Dict, Any

def inspect(path: str):
    """
    Inspects and prints the contents of a PyTorch model checkpoint file.

    This function loads a .pth file, lists its top-level keys, and then
    iterates through the model's state dictionary to display the names and
    shapes of all saved parameters. It also checks for and prints any
    metadata like model dimensions or epoch numbers.

    Args:
        path (str): The file path to the PyTorch checkpoint.
    """
    if not os.path.exists(path):
        print("Checkpoint not found:", path)
        return
    
    # Load the checkpoint to CPU to avoid device-specific issues
    ckpt: Dict[str, Any] = torch.load(path, map_location="cpu")

    # Display top-level keys for a high-level overview
    print("--- Checkpoint Inspection ---")
    print("Top-level keys in checkpoint:", list(ckpt.keys()))

    # Get the state dictionary, handling cases where it's a nested key or the whole file
    sd: Dict[str, torch.Tensor]
    if 'model_state_dict' in ckpt:
        sd = ckpt['model_state_dict']
    else:
        sd = ckpt
        
    print(f"\nNumber of parameters: {len(sd)}")

    # Print the name and shape of each parameter in the state dictionary
    print("\nPARAMETERS (name : shape):")
    for k, v in sd.items():
        try:
            print(f"{k} : {tuple(v.shape)}")
        except Exception:
            print(f"{k} : (couldn't read shape, type={type(v)})")

    # Print any stored metadata that might be useful
    print("\nMETADATA:")
    metadata_found = False
    for meta in ('input_dim', 'hidden_dim', 'output_dim', 'epoch'):
        if meta in ckpt:
            print(f"{meta}: {ckpt[meta]}")
            metadata_found = True
    if not metadata_found:
        print("No specific metadata (input_dim, epoch, etc.) found.")


if __name__ == "__main__":
    # Ensure a file path is provided as a command-line argument
    if len(sys.argv) < 2:
        print("Usage: python inspect_checkpoint.py <checkpoint_path>")
        # Example of how to run the script
        print("Example: python inspect_checkpoint.py my_model.pth")
    else:
        # Call the inspect function with the provided path
        inspect(sys.argv[1])
