"""LangChain VectorStore adapter for VECMAN.

Usage::

    from vecman import VecmanIndex
    from vecman.integrations.langchain import VecmanVectorStore

    index = VecmanIndex.load("my_index")
    store = VecmanVectorStore(index)
    store.similarity_search("query", k=5)

Requires ``langchain-core`` (``pip install langchain-core``); vecman itself
does not depend on it.
"""

from typing import Any, Iterable, List, Optional, Tuple

try:
    from langchain_core.documents import Document
    from langchain_core.vectorstores import VectorStore
except ImportError as e:  # pragma: no cover - exercised only without langchain
    raise ImportError(
        "langchain-core is required for the LangChain integration. "
        "Install it with `pip install langchain-core`."
    ) from e

from ..core.index import VecmanIndex


class VecmanVectorStore(VectorStore):
    """LangChain VectorStore backed by a :class:`VecmanIndex`."""

    def __init__(self, index: VecmanIndex, rerank: bool = False):
        self.index = index
        self.rerank = rerank

    # -- writes ----------------------------------------------------------

    def add_texts(self, texts: Iterable[str],
                  metadatas: Optional[List[dict]] = None,
                  **kwargs: Any) -> List[str]:
        ids = self.index.add_texts(list(texts), metadatas=metadatas)
        return [str(i) for i in ids]

    def delete(self, ids: Optional[List[str]] = None,
               **kwargs: Any) -> Optional[bool]:
        if not ids:
            return False
        removed = self.index.delete([int(i) for i in ids])
        return removed > 0

    # -- reads -----------------------------------------------------------

    def similarity_search_with_score(self, query: str, k: int = 4,
                                     filter: Optional[dict] = None,
                                     **kwargs: Any) -> List[Tuple[Document, float]]:
        results = self.index.search(query, k=k, filter=filter,
                                    rerank=self.rerank)
        return [
            (
                Document(
                    page_content=r.text,
                    metadata={**r.metadata, "id": r.id},
                ),
                r.score,
            )
            for r in results
        ]

    def similarity_search(self, query: str, k: int = 4,
                          filter: Optional[dict] = None,
                          **kwargs: Any) -> List[Document]:
        return [doc for doc, _ in
                self.similarity_search_with_score(query, k=k, filter=filter)]

    # -- constructors ----------------------------------------------------

    @classmethod
    def from_texts(cls, texts: List[str], embedding: Any = None,
                   metadatas: Optional[List[dict]] = None,
                   *, epochs: int = 100, device: str = "cpu",
                   num_subquantizers: int = 8,
                   **kwargs: Any) -> "VecmanVectorStore":
        """Train a fresh compressed index over the given texts.

        The ``embedding`` argument is accepted for interface compatibility
        but ignored: VECMAN embeds with its own sentence-transformers model
        so query-time encoding matches the trained compressor.
        """
        import tempfile

        import numpy as np

        from ..utils.embedding import embed_texts
        from ..utils.training import train_corpus

        vectors = embed_texts(texts)
        with tempfile.TemporaryDirectory() as tmp:
            corpus = f"{tmp}/corpus.npy"
            np.save(corpus, vectors)
            train_corpus(corpus, input_dim=vectors.shape[1], epochs=epochs,
                         num_subquantizers=num_subquantizers, device=device,
                         output_dir=tmp, store_embeddings=False)
            import json

            import torch

            from ..models.vqvae import VQVAE
            with open(f"{tmp}/vqvae_meta.json", "r", encoding="utf-8") as f:
                meta = json.load(f)
            model = VQVAE.from_config(meta)
            model.load_state_dict(torch.load(f"{tmp}/vqvae.pt",
                                             map_location="cpu"))
        index = VecmanIndex(model, device=device)
        index.add_vectors(vectors, texts, metadatas)
        return cls(index)
