# DynCogNLI

A research-driven NLP system for **dynamic commonsense inference**, enabling models to adapt in real time to changing contexts and inputs.

---

## ⚙️ Setup

The following commands help you create and activate a virtual environment.  
This is used to create an isolated Python environment for your project.

```bash
python -m venv .venv
```

Activate the environment:

- **Windows (PowerShell)**
  ```powershell
  .venv\Scripts\Activate.ps1
  ```
- **Windows (CMD)**
  ```cmd
  .venv\Scripts\activate.bat
  ```
- **Linux / macOS**
  ```bash
  source .venv/bin/activate
  ```

Install required libraries:

```bash
pip install -r requirements.txt
```

---

## 🏋️ Training the Model

To train the Graph Attention Network (GAT) with ConceptNet and ATOMIC datasets:

```bash
python train_gat.py   --conceptnet data/conceptnet/conceptnet-assertions-5.6.0.csv   --atomic data/atomic2020/train.csv   --epochs 10   --batch-size 8
```

Arguments:
- `--conceptnet` → Path to ConceptNet CSV file  
- `--atomic` → Path to ATOMIC dataset  
- `--epochs` → Number of training epochs  
- `--batch-size` → Training batch size  

---

## 📊 Evaluation

The `eval_gat.py` script evaluates the performance of the trained GAT model. It loads a checkpoint, prepares evaluation data, runs inference, and computes metrics.

Key Functions:
- **Model Loading** → Loads a pre-trained checkpoint (`.pth`)  
- **Data Preparation** → Prepares subgraphs + ground-truth persona labels  
- **Inference** → Runs predictions on evaluation data  
- **Metrics** → Accuracy, Precision, Recall, F1-score  

Run evaluation with:

```bash
python -m tools.eval_gat   --atomic data/atomic2020/train.csv   --checkpoint models/persona_gat_model.pth   --device cpu   --terms laptop phone car "flight delay"
```

---

## 🔍 Checkpoint Inspection

The `inspect_checkpoint.py` script helps debug and verify PyTorch model checkpoint files (`.pth`) without needing the full model architecture.

Key Features:
- Displays **top-level keys** (`model_state_dict`, `epoch`, `optimizer_state_dict`, etc.)  
- Lists **parameter names + tensor shapes**  
- Shows **metadata** (`input_dim`, `hidden_dim`, `output_dim`)  

Usage:

```bash
python tools/inspect_checkpoint.py models/persona_gat_model.pth
```

---

## 🌐 Streamlit Interface

The project includes a **Streamlit app** for interactive exploration.

Run with:

```bash
streamlit run app.py --server.runOnSave false
```

Access in your browser:
- Local: [http://localhost:8501](http://localhost:8501)  
- Network: `http://<your-ip>:8501`  

---

## 📂 Project Structure

```
DynCogNLI/
│── app.py                     # Streamlit interface
│── train_gat.py               # Main training script
│── tools/
│   ├── eval_gat.py            # Evaluate trained model
│   ├── inspect_checkpoint.py  # Inspect model checkpoints
│── models/                    # Saved models (.pth)
│── data/                      # ConceptNet and ATOMIC datasets
│── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

---

## 📜 Available Scripts

- `train_gat.py` → Train the GAT model  
- `tools/eval_gat.py` → Evaluate the trained model  
- `tools/inspect_checkpoint.py` → Inspect checkpoint contents  
- `app.py` → Run Streamlit interface  

---

## ⚡ Example Workflow

1. **Setup environment**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Train the model**
   ```bash
   python train_gat.py --conceptnet data/conceptnet/conceptnet-assertions-5.6.0.csv --atomic data/atomic2020/train.csv --epochs 10 --batch-size 8
   ```

3. **Evaluate model performance**
   ```bash
   python -m tools.eval_gat --atomic data/atomic2020/train.csv --checkpoint models/persona_gat_model.pth --device cpu --terms laptop phone car "flight delay"
   ```

4. **Inspect a checkpoint**
   ```bash
   python tools/inspect_checkpoint.py models/persona_gat_model.pth
   ```

5. **Launch Streamlit app**
   ```bash
   streamlit run app.py --server.runOnSave false
   ```

---
