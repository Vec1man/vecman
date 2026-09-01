"""VECMAN utility functions for training, embedding and retrieval."""

from .embedding import embed_texts, get_embedder
from .retrieval import (
    load_assets,
    retrieve,
    save_jsonl,
    semantic_retrieve,
    semantic_retrieve_with_scores,
)
from .training import NPZStreamDataset, train_corpus

__all__ = [
    "train_corpus",
    "NPZStreamDataset",
    "embed_texts",
    "get_embedder",
    "save_jsonl",
    "load_assets",
    "retrieve",
    "semantic_retrieve",
    "semantic_retrieve_with_scores",
]
