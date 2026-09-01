"""VecmanIndex: a compressed vector index with CRUD, filtering and IVF.

This is the piece that makes the compression real: documents are stored as
product-quantizer codes (M small integers each). At load/refresh time the
codes are decompressed **once** into normalized latent vectors; every query
is then a single encoder pass plus one matrix multiplication — the corpus
is never re-embedded.

For large collections an IVF (inverted file) index clusters the latents so
queries only scan the closest `nprobe` clusters instead of the whole corpus.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch

from ..models.vqvae import VQVAE

# Below this many live documents a brute-force scan is faster than IVF.
IVF_MIN_SIZE = 10_000


def _matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Matrix multiply through torch. torch is already loaded for the model,
    and routing all BLAS work through one library avoids the duplicate
    OpenMP runtime crash seen when mixing numpy-MKL and torch (Anaconda)."""
    result = torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32)) @ \
        torch.from_numpy(np.ascontiguousarray(b, dtype=np.float32))
    return result.numpy()


@dataclass
class SearchResult:
    id: int
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


def _kmeans(x: np.ndarray, n_clusters: int, n_iter: int = 15,
            seed: int = 0) -> np.ndarray:
    """Spherical k-means (torch ops). Returns centroids (n_clusters, d)."""
    xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    generator = torch.Generator().manual_seed(seed)
    n = xt.shape[0]
    perm = torch.randperm(n, generator=generator)[: min(n_clusters, n)]
    centroids = xt[perm].clone()
    for _ in range(n_iter):
        # Assign each point to the nearest centroid (cosine on normalized data
        # is equivalent to maximizing the dot product).
        assign = torch.argmax(xt @ centroids.t(), dim=1)
        for c in range(centroids.shape[0]):
            members = xt[assign == c]
            if len(members) > 0:
                centroid = members.mean(dim=0)
                norm = centroid.norm()
                if norm > 1e-12:
                    centroids[c] = centroid / norm
    return centroids.numpy()


class VecmanIndex:
    """Compressed vector index backed by a trained VQ-VAE.

    Args:
        model: Trained VQVAE used to compress vectors and encode queries.
        embedding_model: sentence-transformers model name used when raw
            texts (rather than vectors) are added or queried.
    """

    def __init__(self, model: VQVAE, embedding_model: Optional[str] = None):
        from ..utils.embedding import DEFAULT_EMBEDDING_MODEL
        model.eval()
        self.model = model
        self.embedding_model = embedding_model or DEFAULT_EMBEDDING_MODEL
        self.codes = np.empty((0, model.num_subquantizers), dtype=model.codes_dtype)
        self.docs: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self.alive = np.empty(0, dtype=bool)
        # Derived state, rebuilt lazily.
        self._latents: Optional[np.ndarray] = None  # centered+normalized (N, lat_dim)
        self._latent_mean: Optional[np.ndarray] = None
        self._ivf_centroids: Optional[np.ndarray] = None
        self._ivf_assign: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ CRUD

    def __len__(self) -> int:
        return int(self.alive.sum())

    def add_vectors(self, vectors: np.ndarray, texts: Sequence[str],
                    metadatas: Optional[Sequence[Dict[str, Any]]] = None) -> List[int]:
        """Compress and add pre-computed embeddings. Returns assigned ids."""
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors[None, :]
        if vectors.shape[0] != len(texts):
            raise ValueError(
                f"Got {vectors.shape[0]} vectors but {len(texts)} texts"
            )
        if vectors.shape[1] != self.model.input_dim:
            raise ValueError(
                f"Vector dimension {vectors.shape[1]} doesn't match model "
                f"input_dim {self.model.input_dim}"
            )
        if metadatas is None:
            metadatas = [{} for _ in texts]
        if len(metadatas) != len(texts):
            raise ValueError("metadatas length must match texts length")

        new_codes = self.model.compress(torch.from_numpy(vectors))
        start = len(self.docs)
        self.codes = np.concatenate([self.codes, new_codes], axis=0)
        self.docs.extend(str(t) for t in texts)
        self.metadata.extend(dict(m) for m in metadatas)
        self.alive = np.concatenate([self.alive, np.ones(len(texts), dtype=bool)])
        self._invalidate()
        return list(range(start, start + len(texts)))

    def add_texts(self, texts: Sequence[str],
                  metadatas: Optional[Sequence[Dict[str, Any]]] = None) -> List[int]:
        """Embed, compress and add raw texts. Returns assigned ids."""
        from ..utils.embedding import embed_texts
        vectors = embed_texts(list(texts), self.embedding_model)
        return self.add_vectors(vectors, texts, metadatas)

    def delete(self, ids: Union[int, Sequence[int]]) -> int:
        """Soft-delete documents by id. Returns how many were removed."""
        if isinstance(ids, int):
            ids = [ids]
        removed = 0
        for i in ids:
            if 0 <= i < len(self.alive) and self.alive[i]:
                self.alive[i] = False
                removed += 1
        if removed:
            self._invalidate()
        return removed

    def update(self, doc_id: int, text: Optional[str] = None,
               vector: Optional[np.ndarray] = None,
               metadata: Optional[Dict[str, Any]] = None) -> None:
        """Update a document's text (re-compressing it), vector or metadata."""
        if not (0 <= doc_id < len(self.docs)) or not self.alive[doc_id]:
            raise KeyError(f"No live document with id {doc_id}")
        if text is not None and vector is None:
            from ..utils.embedding import embed_texts
            vector = embed_texts([text], self.embedding_model)[0]
        if vector is not None:
            vector = np.asarray(vector, dtype=np.float32)[None, :]
            self.codes[doc_id] = self.model.compress(torch.from_numpy(vector))[0]
            self._invalidate()
        if text is not None:
            self.docs[doc_id] = text
        if metadata is not None:
            self.metadata[doc_id] = dict(metadata)

    # ---------------------------------------------------------------- search

    def _invalidate(self) -> None:
        self._latents = None
        self._ivf_centroids = None
        self._ivf_assign = None

    def _ensure_latents(self) -> np.ndarray:
        """Decompress codes into centered, normalized latents (cached).

        Centering on the corpus mean matters: encoders can produce latents
        with a large shared offset, which is harmless for reconstruction but
        pushes every pairwise cosine toward 1. Removing the mean restores
        the discriminative angular structure.
        """
        if self._latents is None:
            if len(self.docs) == 0:
                self._latents = np.empty((0, self.model.lat_dim), dtype=np.float32)
                self._latent_mean = np.zeros(self.model.lat_dim, dtype=np.float32)
            else:
                lat = self.model.decompress(self.codes).cpu().numpy().astype(np.float32)
                live = self.alive if self.alive.any() else np.ones(len(lat), dtype=bool)
                self._latent_mean = lat[live].mean(axis=0)
                lat = lat - self._latent_mean
                norms = np.linalg.norm(lat, axis=1, keepdims=True)
                norms[norms < 1e-12] = 1.0
                self._latents = lat / norms
        return self._latents

    def _ensure_ivf(self) -> None:
        if self._ivf_centroids is not None:
            return
        latents = self._ensure_latents()
        live_count = len(self)
        n_clusters = max(8, int(np.sqrt(live_count)))
        live_idx = np.flatnonzero(self.alive)
        self._ivf_centroids = _kmeans(latents[live_idx], n_clusters)
        self._ivf_assign = np.argmax(_matmul(latents, self._ivf_centroids.T), axis=1)

    def encode_query(self, query: Union[str, np.ndarray]) -> np.ndarray:
        """Text or raw embedding -> centered, normalized latent vector."""
        if isinstance(query, str):
            from ..utils.embedding import embed_texts
            q_vec = embed_texts([query], self.embedding_model)[0]
        else:
            q_vec = np.asarray(query, dtype=np.float32)
        self._ensure_latents()  # makes sure the corpus latent mean exists
        q_lat = self.model.encode(torch.from_numpy(q_vec).float()).cpu().numpy()[0]
        q_lat = q_lat - self._latent_mean
        norm = np.linalg.norm(q_lat)
        if norm < 1e-12:
            raise ValueError("Query encoded to a zero vector")
        return (q_lat / norm).astype(np.float32)

    def search(self, query: Union[str, np.ndarray], k: int = 5,
               filter: Optional[Dict[str, Any]] = None,
               nprobe: int = 8) -> List[SearchResult]:
        """Search the index.

        Args:
            query: Query text, or a raw embedding in the model's input space.
            k: Number of results.
            filter: Optional metadata equality filter, e.g. {"lang": "en"}.
            nprobe: Number of IVF clusters to scan (large collections only).

        Returns:
            SearchResult list ordered by descending cosine similarity in the
            learned latent space.
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if len(self) == 0:
            return []

        q = self.encode_query(query)
        latents = self._ensure_latents()

        candidates = np.flatnonzero(self.alive)
        if filter:
            candidates = np.array(
                [i for i in candidates if self._matches(self.metadata[i], filter)],
                dtype=np.int64,
            )
            if candidates.size == 0:
                return []
        elif len(self) >= IVF_MIN_SIZE:
            self._ensure_ivf()
            cluster_scores = _matmul(self._ivf_centroids, q)
            top_clusters = np.argsort(cluster_scores)[::-1][:nprobe]
            in_probe = np.isin(self._ivf_assign, top_clusters)
            candidates = np.flatnonzero(in_probe & self.alive)
            if candidates.size == 0:
                candidates = np.flatnonzero(self.alive)

        scores = _matmul(latents[candidates], q)
        k = min(k, candidates.size)
        top = np.argpartition(scores, -k)[-k:]
        top = top[np.argsort(scores[top])[::-1]]

        return [
            SearchResult(
                id=int(candidates[i]),
                text=self.docs[candidates[i]],
                score=float(np.clip(scores[i], -1.0, 1.0)),
                metadata=self.metadata[candidates[i]],
            )
            for i in top
        ]

    @staticmethod
    def _matches(meta: Dict[str, Any], filter: Dict[str, Any]) -> bool:
        return all(meta.get(key) == value for key, value in filter.items())

    # --------------------------------------------------------------- storage

    def save(self, directory: Union[str, Path]) -> None:
        """Persist model weights, codes and documents to a directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), directory / "vqvae.pt")
        np.save(directory / "codes.npy", self.codes)
        with open(directory / "docs.jsonl", "w", encoding="utf-8") as f:
            for i, (text, meta, alive) in enumerate(
                zip(self.docs, self.metadata, self.alive)
            ):
                json.dump(
                    {"id": i, "text": text, "metadata": meta, "alive": bool(alive)},
                    f, ensure_ascii=False,
                )
                f.write("\n")
        meta = {
            "format_version": 3,
            "embedding_model": self.embedding_model,
            **self.model.config(),
        }
        with open(directory / "vqvae_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def load(cls, directory: Union[str, Path]) -> "VecmanIndex":
        """Load an index saved with :meth:`save`."""
        directory = Path(directory)
        required = ["vqvae_meta.json", "vqvae.pt", "codes.npy", "docs.jsonl"]
        missing = [f for f in required if not (directory / f).exists()]
        if missing:
            raise FileNotFoundError(f"Missing required files in {directory}: {missing}")

        with open(directory / "vqvae_meta.json", "r", encoding="utf-8") as f:
            meta = json.load(f)
        if meta.get("format_version", 0) < 3:
            raise ValueError(
                f"{directory} holds a pre-v3 VECMAN index, which used an "
                "incompatible storage format. Re-train with vecman >= 3.0."
            )

        model = VQVAE.from_config(meta)
        state = torch.load(directory / "vqvae.pt", map_location="cpu")
        model.load_state_dict(state)
        model.eval()

        index = cls(model, embedding_model=meta.get("embedding_model"))
        index.codes = np.load(directory / "codes.npy")
        with open(directory / "docs.jsonl", "r", encoding="utf-8") as f:
            alive_flags = []
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"docs.jsonl line {line_num} is not valid JSON: {e}"
                    ) from e
                index.docs.append(doc.get("text", ""))
                index.metadata.append(doc.get("metadata", {}))
                alive_flags.append(bool(doc.get("alive", True)))
        index.alive = np.array(alive_flags, dtype=bool)

        if index.codes.shape[0] != len(index.docs):
            raise ValueError(
                f"Corrupt index: {index.codes.shape[0]} code rows vs "
                f"{len(index.docs)} documents"
            )
        return index
