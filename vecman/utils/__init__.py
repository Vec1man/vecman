"""VECMAN utility functions for training and retrieval."""

from .training import train_corpus, NPZStreamDataset
from .retrieval import (
    embed_texts,
    save_jsonl, 
    load_assets,
    retrieve,
    retrieve_vqvae,
    semantic_retrieve,
    semantic_retrieve_with_scores,
    generate_answer
)

__all__ = [
    # Training utilities
    "train_corpus",
    "NPZStreamDataset",
    
    # Retrieval utilities
    "embed_texts",
    "save_jsonl",
    "load_assets",
    "retrieve",
    "retrieve_vqvae", 
    "semantic_retrieve",
    "semantic_retrieve_with_scores",
    "generate_answer"
] 