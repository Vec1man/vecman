"""VECMAN — learned embedding compression and compressed vector search.

VECMAN trains a product-quantized VQ-VAE on your embeddings and stores each
document as a handful of bytes, then searches directly in the compressed
latent space. Typical storage: 8 bytes/document vs 1536 bytes of float32
for 384-dim embeddings (192x smaller).

Quick start::

    from vecman import VQVAE, VecmanIndex, train_corpus

    out = train_corpus("corpus.npy", input_dim=384, epochs=10, device="cpu")
    index = VecmanIndex.load(out)          # codes decompressed once
    index.add_texts(["new document"])      # incremental adds, no retrain
    results = index.search("my query", k=5, filter={"lang": "en"})
"""

from .core.index import SearchResult, VecmanIndex
from .models.vqvae import VQVAE, ProductQuantizer, EMAVectorQuantizer
from .rag import generate_answer
from .utils.embedding import embed_texts, get_embedder
from .utils.retrieval import (
    load_assets,
    retrieve,
    save_jsonl,
    semantic_retrieve,
    semantic_retrieve_with_scores,
)
from .utils.training import train_corpus

__version__ = "3.0.0"

__all__ = [
    "VQVAE",
    "ProductQuantizer",
    "EMAVectorQuantizer",
    "VecmanIndex",
    "SearchResult",
    "train_corpus",
    "embed_texts",
    "get_embedder",
    "save_jsonl",
    "load_assets",
    "retrieve",
    "semantic_retrieve",
    "semantic_retrieve_with_scores",
    "generate_answer",
]
