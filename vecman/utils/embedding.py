"""Text embedding helpers with model caching.

sentence-transformers is imported lazily so that the compression core
(`vecman.models`, `vecman.core`) can be used with pre-computed vectors on
machines that don't have it installed.
"""

from typing import List

import numpy as np

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_MODEL_CACHE: dict = {}


def get_embedder(model_name: str = DEFAULT_EMBEDDING_MODEL):
    """Return a cached SentenceTransformer instance (loaded once per process)."""
    if model_name not in _MODEL_CACHE:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "sentence-transformers is required for text embedding. "
                "Install it with `pip install sentence-transformers`, or pass "
                "pre-computed vectors instead of raw texts."
            ) from e
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def embed_texts(texts: List[str], model_name: str = DEFAULT_EMBEDDING_MODEL,
                batch_size: int = 256) -> np.ndarray:
    """Embed texts with a sentence-transformers model.

    Args:
        texts: List of texts to embed. Empty/whitespace-only entries get a
            zero vector in their slot.
        model_name: sentence-transformers model name.
        batch_size: Encoding batch size.

    Returns:
        float32 array of shape (len(texts), embedding_dim).

    Raises:
        ValueError: If the list is empty or every text is blank.
    """
    if not texts:
        raise ValueError("texts list cannot be empty")

    valid = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
    if not valid:
        raise ValueError("All texts are empty or None")

    model = get_embedder(model_name)
    _, valid_texts = zip(*valid)
    embeddings = model.encode(
        list(valid_texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=len(valid_texts) > 1000,
    ).astype(np.float32)

    if len(valid) == len(texts):
        return embeddings

    full = np.zeros((len(texts), embeddings.shape[1]), dtype=np.float32)
    for row, (i, _) in enumerate(valid):
        full[i] = embeddings[row]
    return full
