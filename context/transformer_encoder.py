# context/transformer_encoder.py
import torch
from transformers import AutoTokenizer, AutoModel
import os

class TransformerEncoder:
    """
    Encodes text into vector embeddings using a pre-trained Transformer model.
    """
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"TransformerEncoder using device: {self.device}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval() # Set to evaluation mode
            self.embedding_dim = self.model.config.hidden_size
            print(f"INFO: TransformerEncoder initialized with model: {model_name}, embedding_dim: {self.embedding_dim}")
        except Exception as e:
            print(f"ERROR: Failed to load Transformer model {model_name}. Please check your internet connection or model name. Error: {e}")
            self.tokenizer = None
            self.model = None
            self.embedding_dim = 0 # Indicate no embedding capability

    def encode(self, text: str) -> torch.Tensor:
        """
        Encodes a single string into an embedding.
        """
        if not self.model or not self.tokenizer:
            return torch.zeros(self.embedding_dim if self.embedding_dim > 0 else 768, device='cpu') # Return dummy if model failed

        if not text:
            return torch.zeros(self.embedding_dim, device='cpu')

        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Take the mean of the last hidden state for a simple sentence embedding
        sentence_embedding = outputs.last_hidden_state.mean(dim=1).squeeze(0)
        return sentence_embedding.cpu() # Return to CPU for numpy conversion later in graph_builder