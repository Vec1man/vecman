"""VECMAN - VQ-VAE based vector database for efficient text embeddings and retrieval."""

from .models.vqvae import VQVAE
from .utils.training import train_corpus
from .utils.retrieval import (
    embed_texts,
    save_jsonl,
    load_assets,
    retrieve,
    generate_answer
)

__version__ = "0.1.0"

__all__ = [
    "VQVAE",
    "train_corpus",
    "embed_texts",
    "save_jsonl",
    "load_assets",
    "retrieve",
    "generate_answer"
] 