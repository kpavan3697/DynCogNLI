# DynCogNLI

A research-driven NLP system for dynamic commonsense inference, enabling models to adapt in real time to changing contexts and inputs.

## Setup

The following commands help you create and activate a virtual environment.  
This is used to create an isolated Python environment for your project.

```bash
python -m venv .venv
.venv\Scripts\activate
```

Run the following command to install required libraries:

```bash
pip install -r requirements.txt
```

Small Training command

```bash
python train_gnn.py --conceptnet data/conceptnet/conceptnet-assertions-5.6.0.csv --atomic data/atomic2020/train.csv --epochs 10 --batch-size 8
```

inspect_checkpoint.py: Checkpoint Inspection Tool
The inspect_checkpoint.py script is a crucial debugging utility for examining the contents of a PyTorch model checkpoint file (.pth). It allows you to verify the structure and metadata of a saved model without needing to instantiate the full model architecture.

Key Features:
Top-level Keys: Displays the primary keys within the checkpoint (e.g., model_state_dict, epoch, optimizer_state_dict).

Parameter Verification: Lists the name and tensor shape of every parameter in the model's state_dict, which is essential for diagnosing size mismatch errors.

Metadata: Prints any additional metadata saved in the file, such as input_dim, hidden_dim, and output_dim.

Usage:
To run the script, provide the path to your checkpoint file as a command-line argument.

```bash
python tools/inspect_checkpoint.py models/persona_gnn_model.pth
```

The eval_gnn.py script is a command-line tool designed to evaluate the performance of the trained Graph Neural Network (GNN) model. It serves as the primary method for generating quantitative metrics to assess the model's effectiveness in persona inference.

Key Functions:
Model Loading: The script loads the pre-trained GNN model from a specified checkpoint file (.pth) into memory.

Data Preparation: It prepares the evaluation data, which consists of subgraphs and their corresponding ground-truth persona labels.

Inference: The script runs the loaded model on the evaluation data to generate persona predictions for each subgraph.

Metrics Calculation: It compares the model's predictions against the ground-truth labels to compute key performance metrics such as accuracy, precision, recall, and F1-score.

Results Reporting: Finally, it reports these metrics to the console or a file, providing the empirical evidence needed to validate the model's performance in your thesis.

To run eval_gnn.py use the following command

```bash
python -m tools.eval_gnn --conceptnet data/conceptnet/conceptnet-assertions-5.6.0.csv --atomic data/atomic2020/train.csv --checkpoint models/persona_gnn_epoch10.pth --device cpu --terms laptop phone car "flight delay"
```
