# context/transformer_encoder.py
import torch
import hashlib
import os
from transformers import AutoTokenizer, AutoModel

DEFAULT_EMBED_DIM = 384  # fallback embedding dimension

class TransformerEncoder:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', use_fallback_on_error=True):
        """
        Loads a transformer. If loading fails (no internet, etc.) and use_fallback_on_error=True,
        the encoder will fall back to a deterministic hash-based embedding (consistent across runs).
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.embedding_dim = 0
        self.tokenizer = None
        self.model = None
        self._use_hash_fallback = False

        try:
            print(f"[TransformerEncoder] Loading model: {model_name} on {self.device}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            # some transformer models have config.hidden_size
            self.embedding_dim = getattr(self.model.config, "hidden_size", DEFAULT_EMBED_DIM)
            print(f"[TransformerEncoder] Loaded, embedding_dim={self.embedding_dim}")
        except Exception as e:
            print(f"[TransformerEncoder] Failed to load transformer model: {e}")
            if use_fallback_on_error:
                print("[TransformerEncoder] Using deterministic hash-fallback embeddings.")
                self._use_hash_fallback = True
                self.embedding_dim = DEFAULT_EMBED_DIM
            else:
                # leave embedding_dim 0 -> other code should handle
                self.embedding_dim = 0

    def _hash_to_vector(self, text: str):
        """Deterministic vector from text using sha256; returns torch.FloatTensor shape (embedding_dim,)"""
        if text is None:
            text = ""
        h = hashlib.sha256(text.encode('utf-8')).digest()
        # expand or repeat bytes to reach embedding_dim
        vec = []
        idx = 0
        while len(vec) < self.embedding_dim:
            # take 4 bytes => 1 uint32 -> float in [0,1)
            chunk = h[idx % len(h):(idx % len(h)) + 4]
            val = int.from_bytes(chunk, 'little', signed=False)
            vec.append((val % 10000) / 10000.0)  # normalized pseudo-random
            idx += 4
        return torch.tensor(vec[:self.embedding_dim], dtype=torch.float)

    def encode(self, text: str):
        """
        Returns a 1D torch.Tensor embedding on CPU.
        If transformer is loaded, uses the model; otherwise uses deterministic hash fallback.
        """
        if text is None:
            text = ""

        if self.model is not None and self.tokenizer is not None:
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
            # mean pooling
            emb = outputs.last_hidden_state.mean(dim=1).squeeze(0).cpu()
            return emb
        elif self._use_hash_fallback:
            return self._hash_to_vector(text)
        else:
            # final fallback: zeros vector
            return torch.zeros(self.embedding_dim if self.embedding_dim > 0 else DEFAULT_EMBED_DIM, dtype=torch.float)
