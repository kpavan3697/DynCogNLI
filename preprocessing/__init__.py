# dynamic_commonsense_nlp/preprocessing/__init__.py
from .atomic_loader import load_atomic_tsv
from .conceptnet_loader import load_conceptnet


__all__ = ["load_atomic_tsv", "load_conceptnet"]
