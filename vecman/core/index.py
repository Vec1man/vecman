"""VecmanIndex: a compressed vector index with CRUD, filtering, ANN and
two-stage reranking.

Search paths (chosen automatically, overridable via ``method=``):

* **latent** — codes are decompressed once into centered, normalized latent
  vectors (kept on ``self.device``); each query is one encoder pass plus one
  matrix multiplication.
* **adc** — asymmetric distance computation: the query builds a small
  ``(M, K)`` lookup table against the codebooks and document scores are M
  table lookups each. The full latent matrix is never materialized, so RAM
  stays proportional to the stored codes.

Candidate generation (``ann=``): flat scan, IVF (spherical k-means++), or an
HNSW graph. Optional **reranking** re-scores the top candidates against the
original embeddings (stored as float16) for near-exact quality, and
**hybrid** search fuses BM25 keyword scores with dense scores via
reciprocal-rank fusion.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from ..models.vqvae import VQVAE

# Below this many live documents a brute-force scan beats IVF.
IVF_MIN_SIZE = 10_000
# Above this many documents ADC is preferred over materialized latents.
ADC_MIN_SIZE = 50_000
# Chunk size for streaming decompression passes.
DECOMPRESS_CHUNK = 65_536

_FILTER_OPS = {"$in", "$nin", "$ne", "$gt", "$gte", "$lt", "$lte", "$contains"}


@dataclass
class SearchResult:
    id: int
    text: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)


def _kmeans(x: np.ndarray, n_clusters: int, n_iter: int = 15,
            seed: int = 0) -> np.ndarray:
    """Spherical k-means with k-means++ initialization (torch ops).

    Returns centroids of shape (n_clusters, d). All matmuls go through
    torch: mixing numpy-MKL and torch BLAS crashes on some Anaconda setups.
    """
    xt = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))
    generator = torch.Generator().manual_seed(seed)
    n = xt.shape[0]
    n_clusters = min(n_clusters, n)

    # k-means++ seeding: each new centroid is sampled proportionally to its
    # squared distance from the closest centroid chosen so far.
    first = int(torch.randint(0, n, (1,), generator=generator))
    chosen = [first]
    min_dist = 2.0 - 2.0 * (xt @ xt[first])  # squared distance on unit sphere
    for _ in range(1, n_clusters):
        weights = torch.clamp(min_dist, min=0)
        if float(weights.sum()) <= 0:
            idx = int(torch.randint(0, n, (1,), generator=generator))
        else:
            idx = int(torch.multinomial(weights, 1, generator=generator))
        chosen.append(idx)
        min_dist = torch.minimum(min_dist, 2.0 - 2.0 * (xt @ xt[idx]))
    centroids = xt[chosen].clone()

    for _ in range(n_iter):
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
        device: torch device for the model and search math ('cpu'/'cuda').
        store_embeddings: Keep original embeddings as float16 so searches
            can rerank against them (2 bytes/dim per doc). Disable to store
            only the compressed codes.
    """

    def __init__(self, model: VQVAE, embedding_model: Optional[str] = None,
                 device: str = "cpu", store_embeddings: bool = True):
        from ..utils.embedding import DEFAULT_EMBEDDING_MODEL
        model.eval()
        self.device = device
        self.model = model.to(device)
        self.embedding_model = embedding_model or DEFAULT_EMBEDDING_MODEL
        self.store_embeddings = store_embeddings
        self.codes = np.empty((0, model.num_subquantizers), dtype=model.codes_dtype)
        self.docs: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self.alive = np.empty(0, dtype=bool)
        self._embeddings: Optional[np.ndarray] = None  # (N, D) float16
        # Derived state, rebuilt lazily.
        self._invalidate()

    # ------------------------------------------------------------------ CRUD

    def __len__(self) -> int:
        return int(self.alive.sum())

    def to(self, device: str) -> "VecmanIndex":
        """Move the model and search math to another torch device."""
        self.device = device
        self.model = self.model.to(device)
        self._invalidate()
        return self

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

        new_codes = self.model.compress(
            torch.from_numpy(vectors).to(self.device))
        start = len(self.docs)
        self.codes = np.concatenate([self.codes, new_codes], axis=0)
        self.docs.extend(str(t) for t in texts)
        self.metadata.extend(dict(m) for m in metadatas)
        self.alive = np.concatenate([self.alive, np.ones(len(texts), dtype=bool)])
        if self.store_embeddings:
            f16 = vectors.astype(np.float16)
            if self._embeddings is None:
                self._embeddings = f16
            else:
                self._embeddings = np.concatenate(
                    [np.asarray(self._embeddings), f16], axis=0)
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
            self.codes[doc_id] = self.model.compress(
                torch.from_numpy(vector).to(self.device))[0]
            if self.store_embeddings and self._embeddings is not None:
                emb = np.asarray(self._embeddings)
                emb[doc_id] = vector[0].astype(np.float16)
                self._embeddings = emb
            self._invalidate()
        if text is not None:
            self.docs[doc_id] = text
            self._bm25 = None
        if metadata is not None:
            self.metadata[doc_id] = dict(metadata)

    def compact(self) -> Dict[int, int]:
        """Physically remove soft-deleted rows. Returns {old_id: new_id}."""
        keep = np.flatnonzero(self.alive)
        mapping = {int(old): new for new, old in enumerate(keep)}
        self.codes = self.codes[keep]
        self.docs = [self.docs[i] for i in keep]
        self.metadata = [self.metadata[i] for i in keep]
        if self._embeddings is not None:
            self._embeddings = np.asarray(self._embeddings)[keep]
        self.alive = np.ones(len(keep), dtype=bool)
        self._invalidate()
        return mapping

    # ------------------------------------------------------ derived caches

    def _invalidate(self) -> None:
        self._latents: Optional[torch.Tensor] = None      # centered+normalized
        self._latent_mean: Optional[torch.Tensor] = None  # (lat,)
        self._adc_ready = False
        self._codes_t: Optional[torch.Tensor] = None      # (N, M) long
        self._center_norms: Optional[torch.Tensor] = None  # (N,) ||z - mean||
        self._ivf_centroids: Optional[torch.Tensor] = None
        self._ivf_assign: Optional[np.ndarray] = None
        self._hnsw = None
        self._hnsw_ids: Optional[np.ndarray] = None
        self._bm25 = None

    def _decompress_chunks(self):
        for start in range(0, len(self.docs), DECOMPRESS_CHUNK):
            chunk = self.codes[start:start + DECOMPRESS_CHUNK]
            yield self.model.decompress(chunk).to(self.device)

    def _ensure_mean(self) -> torch.Tensor:
        """Corpus latent mean (streamed; no full latent matrix needed).

        Centering matters: encoders can emit latents with a large shared
        offset, harmless for reconstruction but pushing every pairwise
        cosine toward 1. Removing the mean restores angular structure.
        """
        if self._latent_mean is None:
            if len(self.docs) == 0:
                self._latent_mean = torch.zeros(
                    self.model.lat_dim, device=self.device)
            else:
                total = torch.zeros(self.model.lat_dim, device=self.device)
                for chunk in self._decompress_chunks():
                    total += chunk.sum(dim=0)
                self._latent_mean = total / len(self.docs)
        return self._latent_mean

    def _ensure_latents(self) -> torch.Tensor:
        """Materialize centered, normalized latents on self.device (cached)."""
        if self._latents is None:
            mean = self._ensure_mean()
            if len(self.docs) == 0:
                self._latents = torch.empty(
                    (0, self.model.lat_dim), device=self.device)
            else:
                parts = []
                for chunk in self._decompress_chunks():
                    centered = chunk - mean
                    norms = centered.norm(dim=1, keepdim=True).clamp_min(1e-12)
                    parts.append(centered / norms)
                self._latents = torch.cat(parts, dim=0)
        return self._latents

    def _ensure_adc(self) -> None:
        """Prepare ADC state: per-doc centered norms + codes tensor.

        Requires only O(N * M) integers plus O(N) floats — the latent
        matrix is streamed and discarded.
        """
        if self._adc_ready:
            return
        mean = self._ensure_mean()
        norms = []
        for chunk in self._decompress_chunks():
            norms.append((chunk - mean).norm(dim=1))
        self._center_norms = (
            torch.cat(norms) if norms
            else torch.empty(0, device=self.device)
        ).clamp_min(1e-12)
        self._codes_t = torch.from_numpy(
            np.ascontiguousarray(self.codes).astype(np.int64)).to(self.device)
        self._adc_ready = True

    def _adc_dot(self, q_direction: torch.Tensor,
                 candidates: torch.Tensor) -> torch.Tensor:
        """Uncentered dot(z_doc, q_direction) for candidate rows via lookup
        tables — M gathers per document instead of a full matmul."""
        vq = self.model.vq
        codes = self._codes_t[candidates]
        scores = torch.zeros(codes.shape[0], device=self.device)
        if self.model.quantizer_type == "rq":
            for stage, quantizer in enumerate(vq.quantizers):
                table = quantizer.codebook @ q_direction          # (K,)
                scores += table[codes[:, stage]]
        else:
            sub = vq.sub_dim
            for m, quantizer in enumerate(vq.quantizers):
                q_sub = q_direction[m * sub:(m + 1) * sub]
                table = quantizer.codebook @ q_sub                # (K,)
                scores += table[codes[:, m]]
        return scores

    def _ensure_ivf(self) -> None:
        if self._ivf_centroids is not None:
            return
        latents = self._ensure_latents()
        live_idx = np.flatnonzero(self.alive)
        n_clusters = max(8, int(np.sqrt(len(live_idx))))
        centroids = _kmeans(latents[live_idx].cpu().numpy(), n_clusters)
        self._ivf_centroids = torch.from_numpy(centroids).to(self.device)
        self._ivf_assign = torch.argmax(
            latents @ self._ivf_centroids.t(), dim=1).cpu().numpy()

    def _ensure_hnsw(self):
        if self._hnsw is None:
            from .hnsw import HNSW
            latents = self._ensure_latents()
            live = np.flatnonzero(self.alive)
            graph = HNSW(dim=self.model.lat_dim)
            vectors = latents[live].cpu().numpy()
            for row in range(vectors.shape[0]):
                graph.add(vectors[row])
            self._hnsw = graph
            self._hnsw_ids = live
        return self._hnsw

    def _ensure_bm25(self):
        if self._bm25 is None:
            from .bm25 import BM25
            self._bm25 = BM25(self.docs)
        return self._bm25

    # ---------------------------------------------------------------- query

    def _resolve_query_vec(self, query: Union[str, np.ndarray]) -> np.ndarray:
        if isinstance(query, str):
            from ..utils.embedding import embed_texts
            return embed_texts([query], self.embedding_model)[0]
        return np.asarray(query, dtype=np.float32)

    def _encode_centered(self, q_vec: np.ndarray) -> torch.Tensor:
        """Input-space vector -> centered, normalized latent on device."""
        mean = self._ensure_mean()
        q_lat = self.model.encode(
            torch.from_numpy(q_vec).float().to(self.device))[0]
        q_lat = q_lat - mean
        norm = q_lat.norm()
        if norm < 1e-12:
            raise ValueError("Query encoded to a zero vector")
        return q_lat / norm

    def encode_query(self, query: Union[str, np.ndarray]) -> np.ndarray:
        """Text or raw embedding -> centered, normalized latent vector."""
        return self._encode_centered(
            self._resolve_query_vec(query)).cpu().numpy()

    def _candidates(self, q_centered: torch.Tensor,
                    filter: Optional[Dict[str, Any]],
                    ann: str, nprobe: int, pool: int) -> np.ndarray:
        """Candidate document ids (numpy int64) for scoring."""
        live = np.flatnonzero(self.alive)
        if filter:
            return np.array(
                [i for i in live if self._matches(self.metadata[i], filter)],
                dtype=np.int64,
            )
        if ann == "auto":
            ann = "ivf" if len(self) >= IVF_MIN_SIZE else "flat"
        if ann == "hnsw":
            graph = self._ensure_hnsw()
            hits = graph.search(q_centered.cpu().numpy(),
                                k=min(pool, len(self._hnsw_ids)))
            if hits:
                return self._hnsw_ids[np.array([i for _, i in hits])]
            return live
        if ann == "ivf" and len(self) >= IVF_MIN_SIZE:
            self._ensure_ivf()
            cluster_scores = self._ivf_centroids @ q_centered
            top_clusters = torch.topk(
                cluster_scores, min(nprobe, cluster_scores.shape[0])
            ).indices.cpu().numpy()
            in_probe = np.isin(self._ivf_assign, top_clusters)
            candidates = np.flatnonzero(in_probe & self.alive)
            return candidates if candidates.size else live
        return live

    def _score(self, q_centered: torch.Tensor, candidates: np.ndarray,
               method: str) -> torch.Tensor:
        """Cosine scores in the centered latent space for candidate ids."""
        if method == "auto":
            method = "adc" if len(self) >= ADC_MIN_SIZE else "latent"
        cand_t = torch.from_numpy(candidates).to(self.device)
        if method == "adc":
            self._ensure_adc()
            mean = self._ensure_mean()
            dots = self._adc_dot(q_centered, cand_t)
            dots = dots - (mean @ q_centered)
            return dots / self._center_norms[cand_t]
        if method == "latent":
            latents = self._ensure_latents()
            return latents[cand_t] @ q_centered
        raise ValueError(f"Unknown scoring method: {method!r}")

    def _rerank(self, q_vec: np.ndarray, candidate_ids: np.ndarray,
                k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Exact cosine over the stored original embeddings for candidates."""
        if self._embeddings is None:
            raise RuntimeError(
                "Reranking requires stored embeddings "
                "(index built with store_embeddings=True)"
            )
        emb = torch.from_numpy(
            np.asarray(self._embeddings[candidate_ids], dtype=np.float32))
        q = torch.from_numpy(q_vec)
        scores = (emb @ q) / (
            emb.norm(dim=1).clamp_min(1e-12) * q.norm().clamp_min(1e-12))
        k = min(k, len(candidate_ids))
        top = torch.topk(scores, k)
        return candidate_ids[top.indices.numpy()], top.values.numpy()

    def search(self, query: Union[str, np.ndarray], k: int = 5,
               filter: Optional[Dict[str, Any]] = None,
               nprobe: int = 8,
               method: str = "auto",
               ann: str = "auto",
               rerank: bool = False,
               rerank_multiplier: int = 10,
               hybrid: bool = False,
               keyword_query: Optional[str] = None,
               rrf_k: int = 60) -> List[SearchResult]:
        """Search the index.

        Args:
            query: Query text, or a raw embedding in the model's input space.
            k: Number of results.
            filter: Metadata filter. Values are either plain (equality) or
                operator dicts: {"$in": [...]}, {"$nin": [...]}, {"$ne": v},
                {"$gt"/"$gte"/"$lt"/"$lte": number}, {"$contains": v}.
            nprobe: IVF clusters scanned (large, unfiltered searches).
            method: Scoring path — 'auto', 'latent' (materialized matrix) or
                'adc' (codebook lookup tables, lowest memory).
            ann: Candidate generation — 'auto', 'flat', 'ivf' or 'hnsw'.
            rerank: Re-score the top ``k * rerank_multiplier`` candidates
                against the stored original embeddings (near-exact quality).
            rerank_multiplier: Candidate pool factor for reranking.
            hybrid: Fuse BM25 keyword scores with dense scores via
                reciprocal-rank fusion. Uses the query text itself, or
                ``keyword_query`` when the dense query is a raw vector.
            keyword_query: Keyword text for hybrid search when ``query`` is
                a vector.
            rrf_k: RRF smoothing constant.

        Returns:
            SearchResult list ordered by descending score.
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if len(self) == 0:
            return []
        if hybrid:
            if isinstance(query, str):
                keyword_query = keyword_query or query
            elif not keyword_query:
                raise ValueError(
                    "hybrid search with a vector query requires keyword_query"
                )

        q_vec = self._resolve_query_vec(query)
        q_centered = self._encode_centered(q_vec)
        pool = max(k * rerank_multiplier, 50) if (rerank or hybrid) else k

        candidates = self._candidates(q_centered, filter, ann, nprobe, pool)
        if candidates.size == 0:
            return []

        scores_t = self._score(q_centered, candidates, method)
        top_n = min(pool, candidates.size)
        top = torch.topk(scores_t, top_n)
        dense_ids = candidates[top.indices.cpu().numpy()]
        dense_scores = top.values.cpu().numpy()

        if hybrid:
            dense_ids, dense_scores = self._fuse_bm25(
                keyword_query, dense_ids, candidates, pool, rrf_k)

        if rerank:
            out_k = min(k, dense_ids.size)
            dense_ids, dense_scores = self._rerank(q_vec, dense_ids, out_k)

        return self._results(dense_ids[:k], dense_scores[:k],
                             clip=not (rerank or hybrid))

    def _fuse_bm25(self, query_text: str, dense_ids: np.ndarray,
                   candidates: np.ndarray, pool: int,
                   rrf_k: int) -> Tuple[np.ndarray, np.ndarray]:
        bm25 = self._ensure_bm25()
        keyword_scores = bm25.scores(query_text)
        allowed = set(int(i) for i in candidates)
        keyword_order = [
            int(i) for i in np.argsort(keyword_scores)[::-1]
            if int(i) in allowed and keyword_scores[i] > 0
        ][:pool]
        fused: Dict[int, float] = {}
        for rank, doc_id in enumerate(dense_ids.tolist()):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
        for rank, doc_id in enumerate(keyword_order):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (rrf_k + rank + 1)
        ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
        ids = np.array([doc_id for doc_id, _ in ordered], dtype=np.int64)
        vals = np.array([score for _, score in ordered], dtype=np.float32)
        return ids, vals

    def _results(self, ids: np.ndarray, scores: np.ndarray,
                 clip: bool = True) -> List[SearchResult]:
        return [
            SearchResult(
                id=int(i),
                text=self.docs[int(i)],
                score=float(np.clip(s, -1.0, 1.0)) if clip else float(s),
                metadata=self.metadata[int(i)],
            )
            for i, s in zip(ids, scores)
        ]

    # -------------------------------------------- vector-space selection ops

    def search_batch(self, queries: Sequence[Union[str, np.ndarray]],
                     k: int = 5, **kwargs) -> List[List[SearchResult]]:
        """Search many queries at once. Text queries are embedded in one
        batched encoder call; remaining options match :meth:`search`."""
        texts = [q for q in queries if isinstance(q, str)]
        embedded: Dict[int, np.ndarray] = {}
        if texts:
            from ..utils.embedding import embed_texts
            text_vecs = embed_texts(texts, self.embedding_model)
            cursor = 0
            for pos, q in enumerate(queries):
                if isinstance(q, str):
                    embedded[pos] = text_vecs[cursor]
                    cursor += 1
        results = []
        for pos, q in enumerate(queries):
            vec = embedded.get(pos, q)
            results.append(self.search(np.asarray(vec, dtype=np.float32),
                                       k=k, **kwargs))
        return results

    def find_similar(self, doc_id: int, k: int = 5,
                     filter: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """More-like-this: nearest neighbours of a stored document, searched
        directly in the compressed latent space (no re-embedding)."""
        if not (0 <= doc_id < len(self.docs)) or not self.alive[doc_id]:
            raise KeyError(f"No live document with id {doc_id}")
        latents = self._ensure_latents()
        q_centered = latents[doc_id]
        candidates = self._candidates(q_centered, filter, "flat", 8, k + 1)
        candidates = candidates[candidates != doc_id]
        if candidates.size == 0:
            return []
        scores_t = self._score(q_centered, candidates, "latent")
        top = torch.topk(scores_t, min(k, candidates.size))
        ids = candidates[top.indices.cpu().numpy()]
        return self._results(ids, top.values.cpu().numpy())

    def range_search(self, query: Union[str, np.ndarray], min_score: float,
                     filter: Optional[Dict[str, Any]] = None,
                     method: str = "auto") -> List[SearchResult]:
        """Select every document whose latent-space cosine similarity to the
        query is at least ``min_score`` (descending order)."""
        if len(self) == 0:
            return []
        q_centered = self._encode_centered(self._resolve_query_vec(query))
        candidates = self._candidates(q_centered, filter, "flat", 8, len(self))
        if candidates.size == 0:
            return []
        scores_t = self._score(q_centered, candidates, method)
        mask = scores_t >= min_score
        selected = candidates[mask.cpu().numpy()]
        selected_scores = scores_t[mask]
        order = torch.argsort(selected_scores, descending=True).cpu().numpy()
        return self._results(selected[order],
                             selected_scores.cpu().numpy()[order])

    # --------------------------------------------------------------- filter

    @classmethod
    def _matches(cls, meta: Dict[str, Any], filter: Dict[str, Any]) -> bool:
        return all(cls._match_one(meta.get(key), cond)
                   for key, cond in filter.items())

    @staticmethod
    def _match_one(value: Any, cond: Any) -> bool:
        if not isinstance(cond, dict):
            return value == cond
        for op, rhs in cond.items():
            if op not in _FILTER_OPS:
                raise ValueError(f"Unknown filter operator: {op!r}")
            try:
                if op == "$in" and value not in rhs:
                    return False
                if op == "$nin" and value in rhs:
                    return False
                if op == "$ne" and value == rhs:
                    return False
                if op == "$gt" and not (value is not None and value > rhs):
                    return False
                if op == "$gte" and not (value is not None and value >= rhs):
                    return False
                if op == "$lt" and not (value is not None and value < rhs):
                    return False
                if op == "$lte" and not (value is not None and value <= rhs):
                    return False
                if op == "$contains" and (value is None or rhs not in value):
                    return False
            except TypeError:
                return False
        return True

    # --------------------------------------------------------------- storage

    def save(self, directory: Union[str, Path]) -> None:
        """Persist model weights, codes, documents and (optionally) the
        original embeddings to a directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), directory / "vqvae.pt")
        np.save(directory / "codes.npy", self.codes)
        if self._embeddings is not None:
            np.save(directory / "embeddings.f16.npy",
                    np.asarray(self._embeddings, dtype=np.float16))
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
    def load(cls, directory: Union[str, Path],
             device: str = "cpu") -> "VecmanIndex":
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

        index = cls(model, embedding_model=meta.get("embedding_model"),
                    device=device)
        index.codes = np.load(directory / "codes.npy")
        emb_path = directory / "embeddings.f16.npy"
        if emb_path.exists():
            index._embeddings = np.load(emb_path, mmap_mode="r")
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
