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
python train_gat.py --atomic data/atomic2020/train.csv --batch-size 32
```

Arguments:

- `--atomic` → Path to ATOMIC dataset
- `--batch-size` → Training batch size

---

To train the baseline model with ConceptNet and ATOMIC datasets:

```bash
python train_baseline.py
```

Optional Arguments:

- `--epochs` → Path to ATOMIC dataset
- `--batch-size` → Batch size for training
- `--lr` → Learning rate for the optimizer

If no arguments are provided, the default configuration below is used:

- `--epochs` = 20
- `--batch-size` = 8
- `--lr` = 1e-3

---

After training the models, run the following command. This will update the aggregate_summary.json, which is used to display the metrics on the UI.

```bash
python .\evaluation\evaluator.py
```

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
python -m tools.eval_gat --atomic data/atomic2020/train.csv --checkpoint models/persona_gat_model.pth --device cpu --terms [TERM1] [TERM2] "[MULTI-WORD TERM]"
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
├── README.md
├── requirements.txt
├── config.json
├── app.py
├── train_gat.py
├── train_baseline.py
├── run_inference.py
├── aggregate_summary.json
├── evaluated_results.json
│
├── context/
│   ├── context_encoder.py
│   ├── context_manager.py
│   ├── real_time_updater.py
│   ├── response_generator.py
│   └── transformer_encoder.py
│
├── data/
│   ├── atomic2020/
│   │   └── train.csv
│   ├── conceptnet/
│   │   └── conceptnet.db
│   └── kg_mappings/
│       ├── node2id.json
│       └── rel2id.json
│
├── evaluation/
│   ├── dialogue_metric.py
│   ├── evaluation_data.json
│   ├── evaluation_results.json
│   ├── evaluator.py
│   └── get_evaluation_stats.py
│
├── explanation_images/
│
├── knowledge/
│   ├── common_sense_client.py
│   ├── graph_builder.py
│   ├── graph_visualizer.py
│   ├── interactive_visualizer.py
│   ├── retriever.py
│   └── visualizer.py
│
├── llm/
│   └── llm_responder.py
│
├── models/
│   └── persona_gat_model.pth
│
├── preprocessing/
│   ├── atomic_loader.py
│   └── conceptnet_loader.py
│   └── mock_data_generator.py
│
├── reasoning/
│   ├── gat_model.py
│   ├── graph_builder.py
│   ├── multi_hop_reasoner.py
│   └── simple_model.py
│
├── scripts/
│   ├── run_streamlit.bat
│   ├── run_streamlit.ps1
│   └── run_streamlit.sh
│
├── tools/
│   ├── app copy.py
│   ├── eval_gat.py
│   └── inspect_checkpoint.py
│
├── utils/
│   └── graph_utils.py
│
└── visualization/
    └── graph_plotter.py

```

---

## Available Scripts

- `train_gat.py` → Train the GAT model
- `tools/eval_gat.py` → Evaluate the trained model
- `tools/inspect_checkpoint.py` → Inspect checkpoint contents
- `app.py` → Run Streamlit interface

---

## Example Workflow

1. **Setup environment**

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Train the model**

   ```bash
   python train_gat.py --atomic data/atomic2020/train.csv --batch-size 8
   ```

3. **Evaluate model performance**

   ```bash
   python -m tools.eval_gat --atomic data/atomic2020/train.csv --checkpoint models/persona_gat_model.pth --device cpu --terms [TERM1] [TERM2] "[MULTI-WORD TERM]"
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
