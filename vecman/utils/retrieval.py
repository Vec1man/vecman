"""VECMAN retrieval utilities.

The recommended API is :class:`vecman.VecmanIndex`. The functions here keep
the pre-v3 call shapes (`load_assets`, `retrieve`, ...) working, but they are
now correct: retrieval decompresses the stored codes once and searches in
the learned latent space — the corpus is never re-embedded per query.
"""

import json
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

from ..core.index import VecmanIndex
from ..models.vqvae import VQVAE
from .embedding import DEFAULT_EMBEDDING_MODEL, embed_texts, get_embedder

__all__ = [
    "embed_texts",
    "get_embedder",
    "save_jsonl",
    "load_assets",
    "retrieve",
    "semantic_retrieve",
    "semantic_retrieve_with_scores",
]


def save_jsonl(texts: List[str], path: str = "docs.jsonl") -> None:
    """Save texts to JSONL with ids (v3 index document format)."""
    if not texts:
        raise ValueError("texts list cannot be empty")
    with open(path, "w", encoding="utf-8") as f:
        for i, t in enumerate(texts):
            json.dump(
                {"id": i, "text": str(t) if t is not None else "",
                 "metadata": {}, "alive": True},
                f, ensure_ascii=False,
            )
            f.write("\n")


def load_assets(model_dir: Optional[str] = None) -> Tuple[VQVAE, np.ndarray, List[str]]:
    """Load a trained model directory (legacy tuple API).

    Returns:
        (vqvae model, codes array of shape (N, M), document texts)
    """
    index = VecmanIndex.load(Path(model_dir) if model_dir else Path.cwd())
    return index.model, index.codes, index.docs


def _index_from_parts(vqvae: VQVAE, codes: np.ndarray,
                      docs: List[str]) -> VecmanIndex:
    if codes.ndim == 1:
        raise ValueError(
            "Got a flat 1-D codes array (pre-v3 format). Re-train the model "
            "with vecman >= 3.0; the old single-code storage was lossy and "
            "overflowed for large codebooks."
        )
    if len(codes) != len(docs):
        raise ValueError(
            f"Codes length {len(codes)} doesn't match docs length {len(docs)}"
        )
    index = VecmanIndex(vqvae)
    index.codes = codes
    index.docs = list(docs)
    index.metadata = [{} for _ in docs]
    index.alive = np.ones(len(docs), dtype=bool)
    return index


def retrieve(vqvae: VQVAE,
             codes: np.ndarray,
             docs: List[str],
             q_vec: np.ndarray,
             k: int = 5,
             method: str = "auto",
             query_text: str = "",
             return_scores: bool = True) -> Union[List[str], Tuple[List[str], List[float]]]:
    """Retrieve the k most similar documents for a query embedding.

    Args:
        vqvae: Trained VQVAE.
        codes: Stored PQ codes of shape (N, M) — these ARE used: they are
            decompressed into the latent space and searched directly.
        docs: Document texts aligned with `codes`.
        q_vec: Query embedding in the model's input space.
        k: Number of documents to return.
        method: 'vqvae' searches the compressed latent space; 'semantic'
            re-embeds documents with sentence-transformers (slow, exact);
            'auto' means 'vqvae'.
        query_text: Query text (required for method='semantic').
        return_scores: Also return cosine similarity scores.

    Raises:
        ValueError: On invalid inputs — errors are raised, not silently
            converted into arbitrary results.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if len(docs) == 0:
        return ([], []) if return_scores else []

    if method == "semantic":
        if not query_text:
            raise ValueError("method='semantic' requires query_text")
        result_docs, scores = semantic_retrieve_with_scores(docs, query_text, k)
        return (result_docs, scores) if return_scores else result_docs

    if method not in ("vqvae", "auto", "hybrid"):
        raise ValueError(f"Unknown retrieval method: {method!r}")

    q_vec = np.asarray(q_vec, dtype=np.float32)
    if q_vec.size == 0:
        raise ValueError("Query vector cannot be empty")

    index = _index_from_parts(vqvae, codes, docs)
    results = index.search(q_vec, k=k)
    result_docs = [r.text for r in results]
    scores = [r.score for r in results]
    return (result_docs, scores) if return_scores else result_docs


def semantic_retrieve_with_scores(docs: List[str],
                                  query: str,
                                  k: int = 5,
                                  model_name: str = DEFAULT_EMBEDDING_MODEL
                                  ) -> Tuple[List[str], List[float]]:
    """Exact (uncompressed) cosine retrieval with sentence-transformers.

    Slower than the compressed index but useful as a quality baseline.
    """
    if not docs:
        raise ValueError("Documents list cannot be empty")
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    k = min(k, len(docs))

    query_emb = embed_texts([query], model_name)[0]
    doc_embs = embed_texts(docs, model_name)

    q_norm = np.linalg.norm(query_emb)
    d_norms = np.linalg.norm(doc_embs, axis=1)
    similarities = np.zeros(len(docs), dtype=np.float32)
    mask = (d_norms > 1e-12) & (q_norm > 1e-12)
    if mask.any():
        similarities[mask] = (doc_embs[mask] @ query_emb) / (d_norms[mask] * q_norm)

    top = np.argsort(similarities)[::-1][:k]
    return [docs[i] for i in top], [float(similarities[i]) for i in top]


def semantic_retrieve(docs: List[str], query: str, k: int = 5,
                      model_name: str = DEFAULT_EMBEDDING_MODEL) -> List[str]:
    """Exact semantic retrieval, documents only."""
    result_docs, _ = semantic_retrieve_with_scores(docs, query, k, model_name)
    return result_docs
