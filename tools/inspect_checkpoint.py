import sys, os, torch

def inspect(path):
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
    ckpt = torch.load(path, map_location="cpu")

    # Display top-level keys for a high-level overview
    print("Top-level keys in checkpoint:", list(ckpt.keys()))

    # Get the state dictionary, handling cases where it's a nested key or the whole file
    if 'model_state_dict' in ckpt:
        sd = ckpt['model_state_dict']
    else:
        sd = ckpt
        
    print("Number of parameters:", len(sd))

    # Print the name and shape of each parameter in the state dictionary
    print("\nPARAMETERS (name : shape):")
    for k, v in sd.items():
        try:
            print(f"{k} : {tuple(v.shape)}")
        except Exception:
            print(f"{k} : (couldn't read shape, type={type(v)})")

    # Print any stored metadata that might be useful
    print("\nMETADATA:")
    for meta in ('input_dim', 'hidden_dim', 'output_dim', 'epoch'):
        if meta in ckpt:
            print(f"{meta}: {ckpt[meta]}")

if __name__ == "__main__":
    # Ensure a file path is provided as a command-line argument
    if len(sys.argv) < 2:
        print("Usage: python inspect_checkpoint.py <checkpoint_path>")
    else:
        # Call the inspect function with the provided path
        inspect(sys.argv[1])