"""
transformer_encoder.py

This module provides the `TransformerEncoder` class, which is responsible for
generating dense vector representations (embeddings) of text. It uses a pre-trained
transformer model from the Hugging Face library. A key feature of this class is its
robustness: if a model fails to load (e.g., due to a network error or missing files),
it can fall back to a deterministic, hash-based embedding method. This ensures the
system remains functional and consistent even without an external model. The generated
embeddings are used as node features in the Graph Neural Network (GNN) model.
"""

import torch
import hashlib
from typing import Optional, Dict, Any
from transformers import AutoTokenizer, AutoModel

DEFAULT_EMBED_DIM = 384  # A common fallback embedding dimension, matching 'all-MiniLM-L6-v2'.

class TransformerEncoder:
    """
    Encodes text into a dense vector representation using a pre-trained transformer model.

    This class handles the loading and inference of a transformer model for text
    embedding. It is designed with a fallback mechanism to ensure the system
    can still operate if the primary model is unavailable.

    Args:
        model_name (str): The name of the Hugging Face model to load.
                          Defaults to 'sentence-transformers/all-MiniLM-L6-v2'.
        use_fallback_on_error (bool): If True, a hash-based embedding method will
                                      be used if the transformer model fails to load.
                                      Defaults to True.

    Attributes:
        device (torch.device): The device (CPU or CUDA) on which the model will run.
        embedding_dim (int): The dimension of the output embeddings.
        tokenizer (Optional[AutoTokenizer]): The tokenizer for the model.
        model (Optional[AutoModel]): The transformer model itself.
        _use_hash_fallback (bool): Flag to indicate whether the hash fallback is in use.
    """
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2', use_fallback_on_error: bool = True):
        """
        Initializes the TransformerEncoder by attempting to load a model.
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding_dim: int = 0
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModel] = None
        self._use_hash_fallback: bool = False

        try:
            print(f"[TransformerEncoder] Loading model: {model_name} on {self.device}")
            # Load tokenizer and model from Hugging Face.
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            # Dynamically determine the embedding dimension from the model's configuration.
            self.embedding_dim = getattr(self.model.config, "hidden_size", DEFAULT_EMBED_DIM)
            print(f"[TransformerEncoder] Loaded, embedding_dim={self.embedding_dim}")
        except Exception as e:
            # Handle loading errors gracefully.
            print(f"[TransformerEncoder] Failed to load transformer model: {e}")
            if use_fallback_on_error:
                print("[TransformerEncoder] Using deterministic hash-fallback embeddings.")
                self._use_hash_fallback = True
                self.embedding_dim = DEFAULT_EMBED_DIM
            else:
                # If no fallback is desired, set dimension to 0 for error handling in other modules.
                self.embedding_dim = 0

    def _hash_to_vector(self, text: str) -> torch.Tensor:
        """
        Generates a deterministic vector from text using a cryptographic hash function.

        This method is a fallback when a real transformer model cannot be loaded.
        It converts the hash bytes into a pseudo-random floating-point vector.

        Args:
            text (str): The input string to be hashed.

        Returns:
            torch.Tensor: A 1D tensor of shape `(embedding_dim,)`.
        """
        if text is None:
            text = ""
        # Use SHA-256 to create a consistent hash of the input text.
        h = hashlib.sha256(text.encode('utf-8')).digest()
        vec = []
        idx = 0
        while len(vec) < self.embedding_dim:
            # Loop through the hash bytes, converting them to pseudo-random floats.
            chunk = h[idx % len(h):(idx % len(h)) + 4]
            val = int.from_bytes(chunk, 'little', signed=False)
            vec.append((val % 10000) / 10000.0)
            idx += 4
        return torch.tensor(vec[:self.embedding_dim], dtype=torch.float)

    def encode(self, text: str) -> torch.Tensor:
        """
        Encodes a given string into a dense vector representation.

        This is the main public method. It automatically switches between using
        the loaded transformer model and the hash-based fallback.

        Args:
            text (str): The input string to encode.

        Returns:
            torch.Tensor: A 1D tensor representing the text embedding.
        """
        if text is None:
            text = ""

        if self.model is not None and self.tokenizer is not None:
            # Use the transformer model for encoding.
            inputs: Dict[str, Any] = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                padding=True,
                max_length=128
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                # Get the hidden states from the model.
                outputs = self.model(**inputs)
            
            # Apply mean pooling to the last hidden state to get a single vector.
            # This is a common practice for sentence-level embeddings.
            emb = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu()
            return emb
        elif self._use_hash_fallback:
            # Use the deterministic hash-based fallback.
            return self._hash_to_vector(text)
        else:
            # As a final fallback, return a zero vector if no other method is available.
            return torch.zeros(
                self.embedding_dim if self.embedding_dim > 0 else DEFAULT_EMBED_DIM,
                dtype=torch.float
            )